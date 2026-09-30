"""Stage 2 of dataset curation: sample the cleaned pools into a versioned, parallel-ready corpus.

Two slices, drawn from the cleaned per-language pools:
  main axis   TARGET per side from each source language, capped per category so no topic dominates
  paired      the three genuine opposite pairs (military, welfare, protectionism), en+de only,
              because Spanish has 4 welfare-limit sentences in total

Held-out categories are fixed HERE, before any probe is trained, so they cannot be chosen later to
flatter a result. They are tagged, not removed: they belong in the corpus but not in probe training.

    uv run python -m src.sample_corpus --version v1

Layout (data/rile_<version>/): items.jsonl holds ids, labels and metadata with NO text, so it can be
published under the Manifesto licence; text_<lang>.jsonl holds the sentences and stays gitignored.
"""
import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RIGHT = {"104", "201", "203", "305", "401", "402", "407", "414", "505", "601", "603", "605", "606"}
LEFT = {"103", "105", "106", "107", "403", "404", "406", "412", "413", "504", "506", "701", "202"}
EXCLUDED_SUBCODES = {"202.2", "605.2", "703.2"}  # MPDS2024a codebook p. 10
# Fixed before any training: two categories per side, none of them part of a paired issue.
HELD_OUT = {"601": "national way of life", "603": "traditional morality",   # right
            "506": "education expansion", "107": "internationalism"}        # left
PAIRS = {"military": ("105", "104"), "welfare": ("504", "505"), "protectionism": ("406", "407")}
PAIR_QUOTA = {"military": 200, "welfare": 200, "protectionism": 150}
PAIR_SOURCES = ("en", "de")
SEED = 20260917


def code(r):
    return str(r["cmp_code"]).split(".")[0]


def item_id(r):
    """Stable and reproducible: the same sentence in the same document always gets the same id."""
    key = f"{r.get('manifesto_id')}|{r['cmp_code']}|{r['text'].strip()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def load(lang):
    rows = [json.loads(line) for line in (ROOT / "data" / f"clean_{lang}.jsonl").read_text().splitlines()]
    rows = [r for r in rows if str(r["cmp_code"]) not in EXCLUDED_SUBCODES]
    for r in rows:
        r["source_lang"], r["code"], r["item_id"] = lang, code(r), item_id(r)
    return rows


def take(rows, n, rng):
    """n rows sampled without replacement, deterministic given the seed."""
    if len(rows) <= n:
        return list(rows)
    return [rows[i] for i in rng.choice(len(rows), n, replace=False)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--target", type=int, default=1000, help="main-axis sentences per side per source language")
    p.add_argument("--cap", type=int, default=150, help="max per category per side per source language")
    args = p.parse_args()

    rng = np.random.default_rng(SEED)
    out_dir = ROOT / "data" / f"rile_{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen, report = {}, {"main": {}, "paired": {}}

    # ---- main axis: per source language, per side, capped per category ----
    for lang in ("en", "de", "es"):
        rows = load(lang)
        for label in (1, 0):
            side = [r for r in rows if r["label"] == label]
            by_cat = collections.defaultdict(list)
            for r in side:
                by_cat[r["code"]].append(r)
            pool = []
            for cat in sorted(by_cat):
                pool += take(by_cat[cat], args.cap, rng)
            picked = take(pool, args.target, rng)
            for r in picked:
                chosen[r["item_id"]] = r
            report["main"][f"{lang}_{'right' if label else 'left'}"] = {
                "picked": len(picked), "available": len(side), "categories": len(by_cat)}

    # ---- paired subset: top up each issue to its quota from English and German ----
    for issue, (left_code, right_code) in PAIRS.items():
        for cat, label in ((left_code, 0), (right_code, 1)):
            have = [r for r in chosen.values() if r["code"] == cat and r["source_lang"] in PAIR_SOURCES]
            need = PAIR_QUOTA[issue] - len(have)
            if need > 0:
                extra = [r for lang in PAIR_SOURCES for r in load(lang)
                         if r["code"] == cat and r["item_id"] not in chosen]
                for r in take(extra, need, rng):
                    chosen[r["item_id"]] = r
            final = sum(1 for r in chosen.values() if r["code"] == cat and r["source_lang"] in PAIR_SOURCES)
            report["paired"][f"{issue}_{'right' if label else 'left'}"] = {
                "quota": PAIR_QUOTA[issue], "have": final, "met": final >= PAIR_QUOTA[issue]}

    items = sorted(chosen.values(), key=lambda r: r["item_id"])
    for r in items:
        r["pair_issue"] = next((i for i, (a, b) in PAIRS.items() if r["code"] in (a, b)), None)
        r["held_out"] = r["code"] in HELD_OUT

    # ---- write: metadata without text, text separately per language ----
    meta_fields = ("item_id", "source_lang", "label", "cmp_code", "code", "pair_issue", "held_out",
                   "manifesto_id", "party", "partyname", "date")
    (out_dir / "items.jsonl").write_text(
        "".join(json.dumps({k: r.get(k) for k in meta_fields}, ensure_ascii=False) + "\n" for r in items),
        encoding="utf-8")
    for lang in ("en", "de", "es"):
        native = [r for r in items if r["source_lang"] == lang]
        (out_dir / f"text_{lang}.jsonl").write_text(
            "".join(json.dumps({"item_id": r["item_id"], "text": r["text"], "is_native": True},
                               ensure_ascii=False) + "\n" for r in native), encoding="utf-8")

    words = {lab: [len(r["text"].split()) for r in items if r["label"] == lab] for lab in (1, 0)}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip() or "unknown"
    manifest = {
        "version": args.version, "built": "2026-09-17", "seed": SEED, "code_commit": commit,
        "target_per_side_per_lang": args.target, "cap_per_category": args.cap,
        "held_out_categories": HELD_OUT, "pair_quota": PAIR_QUOTA, "pair_sources": list(PAIR_SOURCES),
        "n_items": len(items),
        "by_source_lang": dict(collections.Counter(r["source_lang"] for r in items)),
        "by_label": {"right": sum(r["label"] for r in items), "left": sum(1 - r["label"] for r in items)},
        "held_out_items": sum(r["held_out"] for r in items),
        "paired_items": sum(r["pair_issue"] is not None for r in items),
        "length_words": {"right_mean": round(float(np.mean(words[1])), 2),
                         "left_mean": round(float(np.mean(words[0])), 2),
                         "right_median": float(np.median(words[1])), "left_median": float(np.median(words[0]))},
        "report": report,
        "file_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in sorted(out_dir.glob("*.jsonl"))},
        "note": "items.jsonl carries no sentence text and may be published; text_*.jsonl may not.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "artifacts" / "results" / f"rile_{args.version}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: manifest[k] for k in
                      ("n_items", "by_source_lang", "by_label", "held_out_items", "paired_items", "length_words")},
                     indent=1))
    print("\npaired quotas:", json.dumps(report["paired"], indent=1))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
