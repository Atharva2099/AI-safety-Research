"""Stage 3c of dataset curation: retranslate only the (item, language) pairs the judge flagged.

Dropping an item everywhere because one language mangled one word would be expensive and unnecessary:
the other five translations are fine and the political label is unaffected. So we retranslate just the
flagged language, telling the model what the judge objected to, and re-judge afterwards. An item is
dropped only if a language still fails after a repair attempt.

Fixes are appended to fixes.jsonl rather than rewriting translations.jsonl, so the original output and
every repair attempt stay on the record.

    export GEMINI_API_KEY=...
    uv run --with google-genai python -m src.fix_translations --version v1

Writes data/rile_<version>/fixes.jsonl. Re-judge afterwards with src/vet_translations.py.
"""
import argparse
import collections
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {"en": "English", "es": "Spanish", "de": "German",
             "zh": "Simplified Mandarin Chinese", "hi": "Hindi", "mr": "Marathi"}
MODEL = "gemini-3.7-flash"  # stronger than the bulk translator (flash-lite): these are the hard cases
# Repairs are graded by src/vet_translations.py, whose judge is gemini-3.8-flash: a different model,
# and a newer one, so the checker is at least as capable as the writer rather than marking its own work.
BATCH = 10
RETRIES = 6


def client():
    """Vertex first, so usage draws on the project's GCP credits. A bare API key bills separately,
    so it is used only when explicitly requested with USE_GEMINI_API_KEY=1."""
    from google import genai
    if os.environ.get("USE_GEMINI_API_KEY") == "1" and os.environ.get("GEMINI_API_KEY"):
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    import google.auth
    _, project = google.auth.default()
    return genai.Client(vertexai=True, project=os.getenv("GOOGLE_CLOUD_PROJECT") or project,
                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))


def build_prompt(target, rows):
    return f"""Retranslate these political sentences into {LANGUAGES[target]}. A reviewer rejected the
previous attempt for each one; the objection is given so you can avoid repeating it.

Requirements:
- Preserve the political position exactly. Never negate, soften, strengthen or reverse a stance.
- Fix the specific problem the reviewer identified.
- Keep proper nouns, place names and institutions correct; do not confuse similar names
  (for example Australia and Austria).
- Some inputs are clause fragments from longer sentences. Keep them as fragments.
- Write natural, fluent {LANGUAGES[target]}. Leave no source-language words except proper names,
  and no stray characters from other scripts.

Return only JSON matching the schema, echoing item_id unchanged.

Items:
{json.dumps(rows, ensure_ascii=False, indent=1)}"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=3, help="parallel API calls; the project quota 429s above ~4")
    args = p.parse_args()

    from google.genai import types
    d = ROOT / "data" / f"rile_{args.version}"
    translations = {json.loads(l)["item_id"]: json.loads(l)
                    for l in (d / "translations.jsonl").read_text().splitlines()}
    # Both judging passes: the first over all items, the second over repaired ones (which re-reads the
    # languages that were not repaired and sometimes flags a new problem there). Later verdicts win.
    verdicts = {}
    for name in ("vetting.jsonl", "vetting_fixes.jsonl"):
        path = d / name
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                verdicts[r["item_id"]] = r
    verdicts = list(verdicts.values())

    fixes_path = d / "fixes.jsonl"
    already = set()
    if fixes_path.exists():
        already = {(json.loads(l)["item_id"], json.loads(l)["lang"]) for l in fixes_path.read_text().splitlines()}

    todo = collections.defaultdict(list)  # target language -> items needing repair
    for v in verdicts:
        for lang, d_ in v["verdicts"].items():
            if d_.get("verdict") != "ok" and (v["item_id"], lang) not in already:
                todo[lang].append((v["item_id"], d_.get("verdict"), d_.get("note", "")))
    total = sum(len(v) for v in todo.values())
    print(f"flagged pairs to repair: {total} " + str({k: len(v) for k, v in sorted(todo.items())}))
    if not total:
        return

    cli = client()
    schema = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "item_id": {"type": "STRING"}, "translation": {"type": "STRING"}},
        "required": ["item_id", "translation"]}}
    config = types.GenerateContentConfig(response_mime_type="application/json",
                                         response_schema=schema, temperature=0)
    batches = []
    for lang, entries in sorted(todo.items()):
        if args.limit:
            entries = entries[:args.limit]
        batches += [(lang, entries[k:k + BATCH]) for k in range(0, len(entries), BATCH)]

    def repair(batch):
        """One API call for one language. Returns (rows_to_write, n_failed, error)."""
        lang, chunk = batch
        payload = [{"item_id": i,
                    "source_language": translations[i]["source_lang"],
                    "source_text": translations[i][translations[i]["source_lang"]],
                    "english": translations[i]["en"],
                    "rejected_translation": translations[i][lang],
                    "reviewer_objection": f"{verdict}: {note}"} for i, verdict, note in chunk]
        # The project quota returns 429s under load; back off and retry rather than losing the batch.
        got, err = None, None
        for attempt in range(RETRIES):
            try:
                resp = cli.models.generate_content(model=MODEL, contents=build_prompt(lang, payload),
                                                   config=config)
                got = {r["item_id"]: r["translation"] for r in json.loads(resp.text)}
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:100]}"
                if attempt < RETRIES - 1:
                    time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1.5))
        if got is None:
            return [], len(chunk), f"{lang}: {err}"
        rows_out, missing = [], 0
        for i, verdict, note in chunk:
            new = got.get(i, "").strip()
            if not new:
                missing += 1
                continue
            rows_out.append({"item_id": i, "lang": lang, "text": new,
                             "previous": translations[i][lang], "was": verdict, "model": MODEL})
        return rows_out, missing, None

    done = fail = n_batches = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, fixes_path.open("a", encoding="utf-8") as out:
        for rows_out, missing, err in pool.map(repair, batches):
            n_batches += 1
            fail += missing
            if err:
                print(f"  batch failed: {err}")
            for row in rows_out:  # written from this thread only, so no lock is needed
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                done += 1
            out.flush()
            if n_batches % 5 == 0 or n_batches == len(batches):
                print(f"  {n_batches}/{len(batches)} batches  fixed={done} failed={fail}")
    print(f"\nrepaired {done}, failed {fail}. Re-judge the repaired pairs before using them.")


if __name__ == "__main__":
    main()
