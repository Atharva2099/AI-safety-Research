"""Stage 1 of dataset curation: clean a raw Manifesto pool into a per-language pool.

Rules, in order, each counted in the report:
  1. base      keep >= MIN_WORDS words and a numeric cmp_code inside the RILE index
  2. mojibake  repair Windows-1252 bytes rendered as literal <92> / <96> tags (English only in practice)
  3. conflict  drop every copy of a text whose duplicates disagree on the left/right label (no correct answer)
  4. dedup     keep one copy of each remaining repeated text (campaign slogans repeat across manifestos)
  5. fragment  drop mid-clause fragments: starts lowercase AND has no terminal . ! ?
  6. language  keep only sentences where langdetect and lingua BOTH say the target language

    uv run --with langdetect --with lingua-language-detector python -m src.clean_pools \
        --input data/manifesto_raw.json --lang en

Writes data/clean_<lang>.jsonl, data/dropped_<lang>.jsonl (every removal with a reason) and
artifacts/results/clean_<lang>_report.json. Manifesto text: gitignored, never redistribute.
"""
import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RIGHT = {"104", "201", "203", "305", "401", "402", "407", "414", "505", "601", "603", "605", "606"}
LEFT = {"103", "105", "106", "107", "403", "404", "406", "412", "413", "504", "506", "701", "202"}
MIN_WORDS = 5
TAG = re.compile(r"<([0-9a-fA-F]{2})>")
LINGUA = {"en": "ENGLISH", "de": "GERMAN", "es": "SPANISH"}


def code(row):
    return str(row.get("cmp_code")).split(".")[0]


def repair(text):
    """<92> and friends are single Windows-1252 bytes printed as hex; decode them back to real characters."""
    def sub(m):
        try:
            return bytes([int(m.group(1), 16)]).decode("cp1252")
        except (UnicodeDecodeError, ValueError):
            return m.group(0)
    return TAG.sub(sub, text)


def is_fragment(text):
    t = text.strip()
    return bool(t) and t[:1].islower() and t[-1:] not in ".!?"


def detectors(lang):
    """langdetect plus lingua; a sentence is kept only if both agree on the target language."""
    from langdetect import DetectorFactory, detect
    from lingua import Language, LanguageDetectorBuilder
    DetectorFactory.seed = 0
    names = ["ENGLISH", "GERMAN", "SPANISH", "CATALAN", "PORTUGUESE", "FRENCH", "BASQUE", "ITALIAN", "DUTCH"]
    lingua_det = LanguageDetectorBuilder.from_languages(*[getattr(Language, n) for n in names]).build()

    def check(text):
        try:
            a = detect(text)
        except Exception:
            a = "?"
        b = lingua_det.detect_language_of(text)
        b = b.name if b is not None else "?"
        return a == lang, b == LINGUA[lang], a, b
    return check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--lang", required=True, choices=sorted(LINGUA))
    p.add_argument("--party-prefix", default=None, help="country code prefix, e.g. 41 Germany, 33 Spain")
    args = p.parse_args()

    rows = json.loads(args.input.read_text())
    report = {"input": str(args.input), "lang": args.lang, "party_prefix": args.party_prefix,
              "rules": {}, "artefacts": {}}
    dropped = []

    def drop(rs, reason):
        for r in rs:
            dropped.append({"reason": reason, "manifesto_id": r.get("manifesto_id"),
                            "cmp_code": r.get("cmp_code"), "text": r.get("text")})

    # 1. base
    kept = [r for r in rows
            if len(str(r.get("text") or "").split()) >= MIN_WORDS and code(r) in (RIGHT | LEFT)
            and (args.party_prefix is None or str(r.get("party")).startswith(args.party_prefix))]
    report["rules"]["base"] = {"in": len(rows), "out": len(kept)}

    # 2. mojibake
    n_fixed = 0
    for r in kept:
        found = TAG.findall(r["text"])
        if found:
            n_fixed += 1
            for f in found:
                report["artefacts"][f"<{f}>"] = report["artefacts"].get(f"<{f}>", 0) + 1
            r["text"] = repair(r["text"])
    report["rules"]["mojibake_repaired"] = {"sentences_fixed": n_fixed}

    # 3. label-conflicting duplicates
    by_text = collections.defaultdict(list)
    for r in kept:
        by_text[r["text"].strip()].append(r)
    conflicted = {t for t, v in by_text.items() if len({1 if code(x) in RIGHT else 0 for x in v}) > 1}
    drop([r for r in kept if r["text"].strip() in conflicted], "label_conflict")
    kept = [r for r in kept if r["text"].strip() not in conflicted]
    report["rules"]["label_conflict"] = {"texts": len(conflicted),
                                         "sentences_dropped": sum(len(by_text[t]) for t in conflicted),
                                         "out": len(kept)}

    # 4. dedup, keeping the first copy in a deterministic order
    kept.sort(key=lambda r: (str(r.get("manifesto_id")), str(r.get("date")), r["text"]))
    seen, unique = set(), []
    for r in kept:
        t = r["text"].strip()
        (unique.append(r), seen.add(t)) if t not in seen else drop([r], "duplicate_text")
    report["rules"]["dedup"] = {"extra_copies_removed": len(kept) - len(unique), "out": len(unique)}

    # 5. fragments
    frags = [r for r in unique if is_fragment(r["text"])]
    drop(frags, "fragment")
    unique = [r for r in unique if not is_fragment(r["text"])]
    report["rules"]["fragment"] = {"dropped": len(frags), "out": len(unique)}

    # 6. language agreement
    check = detectors(args.lang)
    final, disagree = [], 0
    for r in unique:
        ok_a, ok_b, a, b = check(r["text"])
        if ok_a and ok_b:
            final.append(r)
        else:
            disagree += int(ok_a != ok_b)
            drop([{**r, "detected": f"langdetect={a},lingua={b}"}], "language")
    report["rules"]["language"] = {"dropped": len(unique) - len(final),
                                   "detectors_disagreed": disagree, "out": len(final)}

    for r in final:
        r["label"] = int(code(r) in RIGHT)
    counts = collections.Counter(r["label"] for r in final)
    report["final"] = {"n": len(final), "right": counts[1], "left": counts[0],
                       "manifestos": len({r.get("manifesto_id") for r in final}),
                       "parties": len({r.get("partyname") for r in final})}

    (ROOT / "data" / f"clean_{args.lang}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final), encoding="utf-8")
    (ROOT / "data" / f"dropped_{args.lang}.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dropped), encoding="utf-8")
    (ROOT / "artifacts" / "results" / f"clean_{args.lang}_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for name, v in report["rules"].items():
        print(f"  {name:24s} {v}")
    print(f"\n{args.lang}: {report['final']}")


if __name__ == "__main__":
    main()
