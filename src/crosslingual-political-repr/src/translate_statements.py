"""Translate suitable political statements with Gemini on Vertex AI."""

import argparse
import json
import os
import time
from pathlib import Path

import google.auth
from google import genai
from google.genai import types


ROOT = Path(__file__).parents[1]
INPUT = ROOT / "data" / "declarative_statements.json"
DEFAULT_OUTPUT = ROOT / "data" / "multilingual_statements.json"
DEFAULT_PROGRESS = ROOT / "data" / "translation_progress.jsonl"
MODEL = "gemini-3.5-flash-lite"
BATCH_SIZE = 25
LANGUAGES = {"es": "Spanish", "de": "German", "zh": "Simplified Mandarin Chinese",
             "hi": "Hindi", "mr": "Marathi"}
SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT", "properties": {
        "id": {"type": "INTEGER"}, "topic": {"type": "STRING"},
        "polarity": {"type": "INTEGER"}, "en": {"type": "STRING"},
        **{language: {"type": "STRING"} for language in LANGUAGES},
    }, "required": ["id", "topic", "polarity", "en", *LANGUAGES]},
}


def get_client() -> genai.Client:
    try:
        _, adc_project = google.auth.default()
    except google.auth.exceptions.GoogleAuthError as error:
        raise RuntimeError("Configure application-default credentials for Vertex AI") from error
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or adc_project
    if not project:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT or configure a project in ADC")
    return genai.Client(vertexai=True, project=project,
                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))


def build_prompt(statements: list[dict]) -> str:
    language_list = ", ".join(f"{code} ({name})" for code, name in LANGUAGES.items())
    return f"""Translate each English political statement below into {language_list}.

Write natural, fluent, native-level translations, preserving the exact meaning,
polarity, scope, and strength of the English. Do not explain, soften, intensify,
or add context. Use the target language throughout; do not leave English words
in a translation unless they are a proper name. Keep the English text unchanged.
Return one object per input statement and only JSON matching the requested schema.

Statements:
 {json.dumps(statements, ensure_ascii=False, indent=2)}"""


def match_source_rows(results: list[dict], source: list[dict]) -> list[dict]:
    """Match rows by their immutable (id, polarity) pair, not by position."""
    source_by_key = {}
    for original in source:
        key = (original["id"], original["polarity"])
        if key in source_by_key:
            raise ValueError(f"Duplicate source key {key}")
        source_by_key[key] = original

    matched = []
    seen = set()
    for result in results:
        key = (result.get("id"), result.get("polarity"))
        if key in seen:
            raise ValueError(f"Duplicate translation key {key}")
        original = source_by_key.get(key)
        if original is None:
            raise ValueError(f"No source row for translation key {key}")
        seen.add(key)
        matched.append(original)
    if len(matched) != len(source):
        missing = set(source_by_key) - seen
        raise ValueError(f"Missing translation keys: {sorted(missing)}")
    return matched


def validate(results: list[dict], source: list[dict]) -> None:
    if len(results) != len(source):
        raise ValueError(f"Expected {len(source)} translations, got {len(results)}")
    originals = match_source_rows(results, source)
    for result, original in zip(results, originals):
        for field in ("id", "topic", "polarity", "en", *LANGUAGES):
            if field not in result or not isinstance(result[field], (str, int)):
                raise ValueError(f"Missing or invalid {field} for statement {original['id']}")
        if (result["id"] != original["id"] or result["topic"] != original["topic"]
                or result["polarity"] != original["polarity"]):
            raise ValueError(f"Identity mismatch for statement {original['id']}")
        if result["polarity"] not in {-1, 0, 1} or result["en"] != original["en"]:
            raise ValueError(f"Source fields changed for statement {original['id']}")
        if any(not result[language].strip() for language in LANGUAGES):
            raise ValueError(f"Empty translation for statement {original['id']}")


def restore_source_fields(results: list[dict], source: list[dict]) -> None:
    """Keep source-controlled fields authoritative if the model echoes them imperfectly."""
    if len(results) != len(source):
        raise ValueError(f"Expected {len(source)} translations, got {len(results)}")
    originals = match_source_rows(results, source)
    for result, original in zip(results, originals):
        result.update({field: original[field] for field in ("topic", "polarity", "en")})


def load_checkpoint(output: Path, source: list[dict]) -> list[dict]:
    if not output.exists():
        return []
    results = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(results, list) or len(results) > len(source):
        raise ValueError("Output checkpoint is not a valid prefix of the input")
    validate(results, source[:len(results)])
    return results


def save_checkpoint(output: Path, results: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Statements per API call (25-50 recommended)")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 50:
        parser.error("--batch-size must be between 1 and 50")

    items = json.loads(INPUT.read_text(encoding="utf-8"))
    suitable = [item for item in items if item.get("is_suitable_for_probing") is True]
    if args.limit is not None:
        suitable = suitable[:args.limit]
    statements = [{"id": item["id"], "topic": item["topic"],
                   "polarity": statement["polarity"], "en": statement["statement"]}
                   for item in suitable for statement in item["statements"]]

    started = time.monotonic()
    results = load_checkpoint(args.output, statements)
    if results:
        print(f"Resuming from checkpoint: {len(results)}/{len(statements)} statements")
    if len(results) < len(statements):
        client = get_client()
        for batch_start in range(len(results), len(statements), args.batch_size):
            batch = statements[batch_start:batch_start + args.batch_size]
            response = client.models.generate_content(
                model=args.model,
                contents=build_prompt(batch),
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                    response_schema=SCHEMA, temperature=0.2),
            )
            parsed = json.loads(response.text)
            restore_source_fields(parsed, batch)
            validate(parsed, batch)
            results.extend(parsed)
            save_checkpoint(args.output, results)
            args.progress.parent.mkdir(parents=True, exist_ok=True)
            with args.progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "processed": len(results), "total": len(statements),
                    "batch_start": batch_start, "batch_size": len(batch),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }) + "\n")
            elapsed = time.monotonic() - started
            rate = len(results) / elapsed if elapsed else 0
            print(f"Translated {len(results)}/{len(statements)} statements "
                  f"({rate:.1f}/s)")
    validate(results, statements)
    print(f"Wrote {len(results)} translated statements to {args.output}")
    print(f"Runtime: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
