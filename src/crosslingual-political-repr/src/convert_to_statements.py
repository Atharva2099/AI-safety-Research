"""Convert global-opinion questions to paired declarative statements with Gemini."""

import argparse
import json
import os
from pathlib import Path

import google.auth
from google import genai
from google.genai import types


ROOT = Path(__file__).parents[1]
INPUT = ROOT / "data" / "raw_global_opinions.json"
OUTPUT = ROOT / "data" / "declarative_statements.json"
PROGRESS = ROOT / "data" / "conversion_progress.jsonl"
MODEL = "gemini-3.5-flash-lite"
BATCH_SIZE = 50

SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT", "properties": {
        "id": {"type": "INTEGER"}, "question": {"type": "STRING"},
        "topic": {"type": "STRING"}, "is_suitable_for_probing": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"}, "statements": {"type": "ARRAY", "items": {
            "type": "OBJECT", "properties": {
                "polarity": {"type": "INTEGER"},
                "ideology_tag": {"type": "STRING"}, "statement": {"type": "STRING"},
            }, "required": ["polarity", "ideology_tag", "statement"]
        }}
    }, "required": ["id", "question", "topic", "is_suitable_for_probing", "reason", "statements"]},
}


def get_client() -> genai.Client:
    try:
        _, adc_project = google.auth.default()
    except google.auth.exceptions.GoogleAuthError as error:
        raise RuntimeError("Configure application-default credentials for Vertex AI") from error
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or adc_project
    if not project_id:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT or configure a project in ADC")
    return genai.Client(vertexai=True, project=project_id,
                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))


def validate_batch(result: list[dict], questions: list[dict]) -> None:
    if len(result) != len(questions):
        raise ValueError(f"Expected {len(questions)} results, got {len(result)}")
    allowed = {-1, 0, 1}
    for record, question in zip(result, questions):
        if record["id"] != question["id"]:
            raise ValueError(f"Result identity mismatch for question {question['id']}")
        # The source question is authoritative; models occasionally echo it with
        # harmless whitespace or punctuation changes.
        record["question"] = question["question"]
        for statement in record["statements"]:
            if statement["polarity"] not in allowed or not statement["statement"].strip():
                raise ValueError(f"Invalid statement for question {question['id']}")


def build_prompt(questions: list[dict]) -> str:
    return f"""For each question, identify its policy topic and decide whether it is suitable for ideological stance probing.

Mark a question unsuitable with an empty statements list when it asks about a politician or party rating, a vague evaluation/perception/feeling/trust judgment, a factual assessment or prediction, or lacks explicit substantive policy alternatives. Do not infer alternatives from general political knowledge.

For suitable questions, write standalone assertive declarative statements directly paraphrasing the explicit substantive alternatives. Label opposing alternatives +1 and -1. Include polarity 0 only when a substantive middle or neutral option is present; never invent one. Preserve qualifiers and scope, and do not add actors, goals, consequences, values, or policy details. Keep statements symmetric and neutral. Return one result per input, preserving id and question exactly, and return only JSON matching the schema.

Questions:
{json.dumps(questions, ensure_ascii=False, indent=2)}"""


def load_checkpoint(questions: list[dict]) -> list[dict]:
    if not OUTPUT.exists():
        return []
    results = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if not isinstance(results, list) or len(results) > len(questions):
        raise ValueError("Output checkpoint is not a valid prefix of the input")
    for result, question in zip(results, questions):
        if result.get("id") != question["id"]:
            raise ValueError(f"Output checkpoint does not match question {question['id']}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    questions = json.loads(INPUT.read_text(encoding="utf-8"))
    client = get_client()
    results = load_checkpoint(questions)
    start = len(results)
    if start:
        print(f"Resuming from checkpoint: {start}/{len(questions)}")
    for start in range(start, len(questions), BATCH_SIZE):
        batch = [{"id": q["id"], "question": q["question"], "options": q.get("options", [])}
                 for q in questions[start : start + BATCH_SIZE]]
        response = client.models.generate_content(
            model=args.model, contents=build_prompt(batch),
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                                response_schema=SCHEMA, temperature=0.2),
        )
        parsed = json.loads(response.text)
        validate_batch(parsed, batch)
        results.extend(parsed)
        progress = {"processed": len(results), "batch_start": start,
                    "batch_size": len(batch), "suitable": sum(r["is_suitable_for_probing"] for r in parsed),
                    "statements": sum(len(r["statements"]) for r in parsed)}
        with PROGRESS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(progress) + "\n")
        OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Processed {len(results)}/{len(questions)}")
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    suitable = sum(r["is_suitable_for_probing"] for r in results)
    print(f"Wrote {len(results)} records: {suitable} suitable, {len(results) - suitable} unsuitable, "
          f"{sum(len(r['statements']) for r in results)} statements")


if __name__ == "__main__":
    main()
