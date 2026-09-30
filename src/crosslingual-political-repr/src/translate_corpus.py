"""Stage 3 of dataset curation: translate the sampled corpus into all six languages.

Every item exists in en, de, es, zh, hi and mr. An item's own language is native text; the other five
are translations of it. Items are keyed by item_id, so a sentence stays identifiable in every language.

Resumable: each finished item is appended to a progress file and skipped on the next run.

    export GEMINI_API_KEY=...            # or configure Vertex ADC
    uv run --with google-genai python -m src.translate_corpus --version v1 --source en

Writes data/rile_<version>/translations.jsonl (progress, one row per item) and rewrites
text_<lang>.jsonl for all six languages. Manifesto-derived text: gitignored, never redistribute.
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {"en": "English", "es": "Spanish", "de": "German",
             "zh": "Simplified Mandarin Chinese", "hi": "Hindi", "mr": "Marathi"}
MODEL = "gemini-3.8-flash"  # v1 corpus was translated with gemini-3.5-flash-lite; each row records its own model
BATCH = 20


def client():
    """Vertex first, matching src/translate_statements.py, so usage draws on the project's GCP credits.
    A bare API key bills separately, so it is used only when USE_GEMINI_API_KEY=1 is set explicitly."""
    from google import genai
    if os.environ.get("USE_GEMINI_API_KEY") == "1" and os.environ.get("GEMINI_API_KEY"):
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"]), "api_key"
    import google.auth
    _, project = google.auth.default()
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or project
    return genai.Client(vertexai=True, project=project,
                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global")), "vertex"


def prompt_for(source_lang, targets, rows):
    names = ", ".join(f"{c} ({LANGUAGES[c]})" for c in targets)
    return f"""Translate each {LANGUAGES[source_lang]} political sentence below into {names}.

These are sentences from real party election manifestos. Translate them faithfully:

- Preserve the political position exactly. Never soften, strengthen, negate, or reverse a stance.
- Preserve scope and hedging ("some", "all", "may", "must") exactly as written.
- Keep proper nouns, party names, institutions and place names recognisable.
- Some inputs are clause fragments taken from longer sentences. Translate the fragment as it stands;
  do not complete it, and do not add context that is not there.
- Write natural, fluent, native-level text in each target language. Do not leave source-language words
  in a translation unless they are a proper name.
- Return one object per input, echoing item_id unchanged, and only JSON matching the schema.

Sentences:
{json.dumps(rows, ensure_ascii=False, indent=1)}"""


def schema_for(targets):
    return {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"item_id": {"type": "STRING"}, **{t: {"type": "STRING"} for t in targets}},
        "required": ["item_id", *targets]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--source", required=True, choices=sorted(LANGUAGES), help="source language to translate FROM")
    p.add_argument("--limit", type=int, default=None, help="translate at most this many items (for a trial run)")
    p.add_argument("--sleep", type=float, default=0.5)
    args = p.parse_args()

    from google.genai import types
    d = ROOT / "data" / f"rile_{args.version}"
    targets = [c for c in LANGUAGES if c != args.source]
    items = {r["item_id"]: r for r in map(json.loads, (d / "items.jsonl").read_text().splitlines())
             if r["source_lang"] == args.source}
    text = {r["item_id"]: r["text"] for r in
            map(json.loads, (d / f"text_{args.source}.jsonl").read_text().splitlines())}

    progress_path = d / "translations.jsonl"
    done = set()
    if progress_path.exists():
        done = {json.loads(l)["item_id"] for l in progress_path.read_text().splitlines()}
    todo = [i for i in items if i not in done]
    already = len(items) - len(todo)  # count before --limit, or the message misreports a fresh run
    if args.limit:
        todo = todo[:args.limit]
    print(f"source {args.source}: {len(items)} items, {already} already done, {len(todo)} to do now")
    if not todo:
        return

    cli, auth = client()
    print(f"auth: {auth} | model: {MODEL} | targets: {','.join(targets)}")
    config = types.GenerateContentConfig(response_mime_type="application/json",
                                         response_schema=schema_for(targets), temperature=0)
    ok = fail = 0
    with progress_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), BATCH):
            chunk = [{"item_id": i, "text": text[i]} for i in todo[start:start + BATCH]]
            try:
                resp = cli.models.generate_content(model=MODEL, contents=prompt_for(args.source, targets, chunk),
                                                   config=config)
                got = {r["item_id"]: r for r in json.loads(resp.text)}
            except Exception as e:
                fail += len(chunk)
                print(f"  batch at {start}: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(5)
                continue
            for c in chunk:
                r = got.get(c["item_id"])
                if not r or any(not str(r.get(t, "")).strip() for t in targets):
                    fail += 1
                    continue
                row = {"item_id": c["item_id"], "source_lang": args.source, args.source: c["text"],
                       **{t: r[t] for t in targets}, "model": MODEL}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok += 1
            out.flush()
            print(f"  {min(start + BATCH, len(todo))}/{len(todo)}  ok={ok} fail={fail}")
            time.sleep(args.sleep)
    print(f"\ndone: {ok} translated, {fail} failed (rerun to retry failures)")


if __name__ == "__main__":
    main()
