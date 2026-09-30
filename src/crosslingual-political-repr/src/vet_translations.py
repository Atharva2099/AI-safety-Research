"""Stage 3b of dataset curation: check that translations kept the political stance.

A stronger model than the translator reads the source sentence next to all five translations and flags
any that drift. The failure that matters most is a flipped stance: a sentence that argues for something
in English and against it in Hindi would be a labelled error in the corpus, not just a clumsy phrase.

Because the corpus must stay parallel, an item flagged in ANY language is dropped from ALL languages.
Flags are written, not applied; `--apply` is a separate deliberate step.

    export GEMINI_API_KEY=...
    uv run --with google-genai python -m src.vet_translations --version v1          # judge pass
    uv run python -m src.vet_translations --version v1 --sample hi mr               # human check sheets

Writes data/rile_<version>/vetting.jsonl (resumable) and samples to data/rile_<version>/sample_<lang>.tsv.
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
JUDGE_MODEL = "gemini-3.8-flash"  # must differ from the model that produced the translation being judged
BATCH = 10
RETRIES = 6
ISSUES = ("ok", "stance_flipped", "meaning_changed", "not_translated", "unintelligible")


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


def build_prompt(rows, source_lang, targets):
    return f"""You are checking machine translations of sentences taken from real party election manifestos.

For each item you are given the original {LANGUAGES[source_lang]} sentence and its translations into
{", ".join(LANGUAGES[t] for t in targets)}. Judge each translation separately and report one verdict per language:

- "ok": the translation carries the same political position, scope and strength as the original.
- "stance_flipped": it argues the opposite, negates something the original asserted, or drops a negation.
- "meaning_changed": the topic or claim differs materially, or hedging/scope changed ("some" to "all").
- "not_translated": it is still largely in the source language, or was left untranslated.
- "unintelligible": it is not fluent enough in the target language to be read as a real sentence.

Judge only fidelity to the original. Do NOT judge whether you agree with the politics, and do not
penalise a sentence for being a clause fragment: many inputs are fragments and should stay fragments.
Be strict about stance_flipped; that is the error that would corrupt the dataset's labels.

Return only JSON matching the schema, echoing item_id unchanged, one object per input item.

Items:
{json.dumps(rows, ensure_ascii=False, indent=1)}"""


def schema_for(targets):
    verdict = {"type": "OBJECT", "properties": {
        "verdict": {"type": "STRING", "enum": list(ISSUES)}, "note": {"type": "STRING"}},
        "required": ["verdict"]}
    return {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "item_id": {"type": "STRING"}, **{t: verdict for t in targets}},
        "required": ["item_id", *targets]}}


def load_translations(version):
    d = ROOT / "data" / f"rile_{version}"
    return [json.loads(l) for l in (d / "translations.jsonl").read_text().splitlines()]


def write_samples(version, langs, n, seed=20260917):
    """Side-by-side sheets for a human to read. TSV so it opens in any spreadsheet."""
    d = ROOT / "data" / f"rile_{version}"
    rows = load_translations(version)
    rng = random.Random(seed)
    for lang in langs:
        pick = rng.sample(rows, min(n, len(rows)))
        # English is included as a pivot: every item exists in English, so a reader who does not read
        # the source language can still check the translation against something.
        lines = ["item_id\tsource_lang\tsource_text\tenglish\t%s_translation\tverdict_write_ok_or_problem" % lang]
        for r in pick:
            clean = lambda s: s.replace("\t", " ").replace("\n", " ")
            lines.append(f"{r['item_id']}\t{r['source_lang']}\t{clean(r[r['source_lang']])}\t"
                         f"{clean(r['en'])}\t{clean(r[lang])}\t")
        path = d / f"sample_{lang}.tsv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(pick)} rows)")


def write_flagged(version, langs):
    """Every flagged translation for these languages, with the judge's objection, for a human to review
    before any repair is run. The last column is for the reader's own verdict."""
    d = ROOT / "data" / f"rile_{version}"
    tr = {json.loads(l)["item_id"]: json.loads(l) for l in (d / "translations.jsonl").read_text().splitlines()}
    verdicts = [json.loads(l) for l in (d / "vetting.jsonl").read_text().splitlines()]
    collected = []
    for lang in langs:
        lines = ["item_id\tverdict\tjudge_note\tsource_lang\tsource_text\tenglish\t"
                 f"{lang}_translation\tyour_verdict_agree_or_disagree"]
        n = 0
        for v in verdicts:
            verdict = v["verdicts"].get(lang, {})
            if verdict.get("verdict", "ok") == "ok":
                continue
            r = tr[v["item_id"]]
            clean = lambda s: str(s).replace("\t", " ").replace("\n", " ")
            lines.append("\t".join([v["item_id"], verdict["verdict"], clean(verdict.get("note", "")),
                                    r["source_lang"], clean(r[r["source_lang"]]), clean(r["en"]),
                                    clean(r[lang]), ""]))
            n += 1
        path = d / f"flagged_{lang}.tsv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(f"wrote {path} ({n} flagged)")
        for v in verdicts:
            verdict = v["verdicts"].get(lang, {})
            if verdict.get("verdict", "ok") != "ok":
                collected.append((verdict["verdict"], lang, v["item_id"], verdict, tr[v["item_id"]]))

    # One Markdown document for all languages, worst verdict first so the label-corrupting cases
    # are read before the cosmetic ones.
    order = ["stance_flipped", "unintelligible", "not_translated", "meaning_changed"]
    kinds = [o for o in order if any(c[0] == o for c in collected)]
    kinds += sorted({c[0] for c in collected} - set(kinds))
    md = [f"# Flagged translations ({len(collected)} across {', '.join(LANGUAGES[l] for l in langs)})", "",
          "Mark each one: **FIX** if the reviewer is right, **KEEP** if the reviewer is wrong.",
          "Stance flips come first: those are the ones that would put a wrong label in the dataset.", "",
          "| Section | Count |", "|---|---:|"]
    md += [f"| {k.replace('_', ' ').title()} | {sum(1 for c in collected if c[0] == k)} |" for k in kinds]
    md += [""]
    for kind in kinds:
        entries = [c for c in collected if c[0] == kind]
        md += [f"## {kind.replace('_', ' ').title()} ({len(entries)})", ""]
        for idx, (_, lang, item_id, verdict, r) in enumerate(entries, 1):
            md += [f"**{idx}. {LANGUAGES[lang]}** · `{item_id}` — reviewer: {verdict.get('note') or '(no note)'}", "",
                   f"- **English:** {r['en']}", f"- **{LANGUAGES[lang]}:** {r[lang]}"]
            if r["source_lang"] != "en":
                md += [f"- **Original ({LANGUAGES[r['source_lang']]}):** {r[r['source_lang']]}"]
            md += ["- **Your verdict:** ", ""]
    md_path = d / "flagged_all.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {md_path} ({len(collected)} entries)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--flagged", nargs="*", default=None, help="export flagged translations for review")
    p.add_argument("--sample", nargs="*", default=None, help="write human check sheets for these languages")
    p.add_argument("--sample-size", type=int, default=50)
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8, help="parallel API calls; lower this if the API rate-limits")
    p.add_argument("--only", nargs="*", default=None, help="re-judge just these item ids, ignoring existing verdicts")
    p.add_argument("--rejudge-fixes", action="store_true",
                   help="re-check items repaired by src/fix_translations.py, reading the repaired text")
    args = p.parse_args()

    if args.flagged is not None:
        write_flagged(args.version, args.flagged or ["hi", "mr"])
        return
    if args.sample is not None:
        write_samples(args.version, args.sample or ["hi", "mr"], args.sample_size)
        return

    from google.genai import types
    d = ROOT / "data" / f"rile_{args.version}"
    rows = {r["item_id"]: r for r in load_translations(args.version)}
    out_path = d / "vetting.jsonl"
    if args.rejudge_fixes:
        # Judge the repaired text, not the original: overlay fixes.jsonl and re-check only those items.
        # Written to a separate file so the first-pass verdicts stay on the record.
        fixed = collections.defaultdict(dict)
        for line in (d / "fixes.jsonl").read_text().splitlines():
            f = json.loads(line)
            fixed[f["item_id"]][f["lang"]] = f["text"]
        for item_id, langs in fixed.items():
            rows[item_id] = {**rows[item_id], **langs}
        rows = {k: v for k, v in rows.items() if k in fixed}
        out_path = d / "vetting_fixes.jsonl"
        print(f"re-judging {len(rows)} repaired items with {args.model}")
    done = {json.loads(l)["item_id"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    if args.only:
        # Re-judge these ids even though a verdict exists: their text changed since it was written
        # (a later repair pass). The new verdict is appended, and the last one for an id wins.
        done -= set(args.only)
        rows = {k: v for k, v in rows.items() if k in set(args.only)}
    todo = [i for i in rows if i not in done]
    print(f"{len(rows)} items, {len(done)} already judged, {len(todo)} to do")
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        return

    cli = client()
    # Batch within a source language: one prompt cannot mix sources, since the target set differs.
    batches = []
    for src in sorted({rows[i]["source_lang"] for i in todo}):
        ids = [i for i in todo if rows[i]["source_lang"] == src]
        batches += [(src, ids[k:k + BATCH]) for k in range(0, len(ids), BATCH)]

    def judge(batch):
        """One API call. Returns (rows_to_write, n_failed); raises nothing."""
        src, chunk_ids = batch
        targets = [t for t in LANGUAGES if t != src]
        payload = [{"item_id": i, "source": rows[i][src], **{t: rows[i][t] for t in targets}}
                   for i in chunk_ids]
        config = types.GenerateContentConfig(response_mime_type="application/json",
                                             response_schema=schema_for(targets), temperature=0)
        # The project's quota is shared across workers, so 429s are expected under parallelism.
        # Back off and retry rather than dropping the batch; jitter keeps workers from retrying in lockstep.
        last = None
        for attempt in range(RETRIES):
            try:
                resp = cli.models.generate_content(model=args.model,
                                                   contents=build_prompt(payload, src, targets), config=config)
                got = {r["item_id"]: r for r in json.loads(resp.text)}
                break
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:100]}"
                if attempt == RETRIES - 1:
                    return [], len(chunk_ids), last
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1.5))
        else:
            return [], len(chunk_ids), last
        out_rows, missing = [], 0
        for i in chunk_ids:
            r = got.get(i)
            if not r:
                missing += 1
                continue
            verdicts = {t: r[t] for t in targets if t in r}
            bad = {t: v for t, v in verdicts.items() if v.get("verdict") != "ok"}
            out_rows.append({"item_id": i, "source_lang": src, "verdicts": verdicts,
                             "keep": not bad, "judge_model": args.model})
        return out_rows, missing, None

    ok = flagged = failed = done_batches = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a", encoding="utf-8") as out:
        for out_rows, missing, err in pool.map(judge, batches):
            done_batches += 1
            failed += missing
            if err:
                print(f"  batch failed: {err}")
            for row in out_rows:  # written from this thread only, so no lock is needed
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok += row["keep"]
                flagged += not row["keep"]
            out.flush()
            if done_batches % 10 == 0 or done_batches == len(batches):
                print(f"  {done_batches}/{len(batches)} batches  ok={ok} flagged={flagged} failed={failed}")
    print(f"\njudged: {ok} clean, {flagged} flagged, {failed} failed (rerun to retry)")


if __name__ == "__main__":
    main()
