"""Build one language's left-right probing set from Manifesto data, keyed by the official RILE index.

RILE (Manifesto Project Dataset codebook, MPDS2024a s3.6 "Programmatic dimensions", p.30):
  right = per104 + per201 + per203 + per305 + per401 + per402 + per407 + per414 + per505
        + per601 + per603 + per605 + per606
  left  = per103 + per105 + per106 + per107 + per403 + per404 + per406 + per412 + per413
        + per504 + per506 + per701 + per202
Categories outside both lists are dropped (RILE ignores them).

Pipeline: cap each category so no topic dominates -> tone-match right vs left on cardiff valence.
The validator gate (independent tone models on the finished set) is run separately.

    uv run --with torch --with transformers --with sentencepiece --with protobuf python -m src.build_rile_set

Writes data/manifesto_<lang>_rile.json (gitignored: Manifesto data must not be redistributed).
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from src.metrics import NEGATION_BY_LANG, auc
from src.tone import tone_probs, valence

ROOT = Path(__file__).resolve().parents[1]
RIGHT = {"104", "201", "203", "305", "401", "402", "407", "414", "505", "601", "603", "605", "606"}
LEFT = {"103", "105", "106", "107", "403", "404", "406", "412", "413", "504", "506", "701", "202"}
EXCLUDED_SUBCODES = {"202.2", "605.2", "703.2"}  # MPDS2024a codebook p. 10
MIN_WORDS, SEED = 5, 20260915


def detect_lang(text):
    """Detected language of one sentence, or '?' when langdetect cannot decide."""
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0  # langdetect is randomised; fix it so builds are reproducible
    try:
        return detect(text)
    except Exception:
        return "?"


def load_rile_rows(path, party_prefix=None):
    """RILE-keyed sentences in a fixed order (must match the order used when tone scores were cached).

    party_prefix selects one country out of a multi-country file (Manifesto party ids start with the
    country code: 41 = Germany, 33 = Spain, 51 = Great Britain, 61 = United States).
    """
    rows = [r for r in json.loads(Path(path).read_text())
            if len(str(r.get("text") or "").split()) >= MIN_WORDS
            and str(r.get("cmp_code")) not in EXCLUDED_SUBCODES
            and str(r.get("cmp_code")).split(".")[0] in (RIGHT | LEFT)
            and (party_prefix is None or str(r.get("party")).startswith(party_prefix))]
    for r in rows:
        r["code"] = str(r["cmp_code"]).split(".")[0]
        r["label"] = int(r["code"] in RIGHT)
    return rows


def report(tag, rows, tone, lang="en"):
    y = np.array([r["label"] for r in rows])
    v = np.array([tone[id(r)] for r in rows])
    pattern = NEGATION_BY_LANG[lang]  # an English pattern on German text would report a meaningless 0.000
    neg = np.array([bool(pattern.search(r["text"])) for r in rows])
    print(f"{tag:<18}right {int(y.sum()):>6}  left {int((1 - y).sum()):>6}  "
          f"tone AUC {auc(y, v):.3f}  negation gap {neg[y == 1].mean() - neg[y == 0].mean():+.3f}")
    return {"n_right": int(y.sum()), "n_left": int((1 - y).sum()), "tone_auc": auc(y, v),
            "negation_gap": float(neg[y == 1].mean() - neg[y == 0].mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cap", type=int, default=800, help="max sentences per category, so no topic dominates")
    p.add_argument("--bins", type=int, default=10, help="tone bins used for matching")
    p.add_argument("--input", type=Path, default=ROOT / "data" / "manifesto_raw.json")
    p.add_argument("--lang", default="en", help="language tag for the output file")
    p.add_argument("--party-prefix", default=None, help="country code prefix, e.g. 41 for Germany, 33 for Spain")
    p.add_argument("--match-model", default="cardiff", help="tone model to match on; use one competent in this language")
    p.add_argument("--cache-key", default=None, help="tone cache key; defaults to one per language and filter setting")
    p.add_argument("--no-lang-filter", action="store_true", help="keep sentences whose detected language differs")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out_path = args.out or ROOT / "data" / f"manifesto_{args.lang}_rile.json"
    # The cache is keyed to an exact sentence list, so filtering needs its own key (a stale key would
    # trip the text-hash assert in tone_probs rather than silently misalign scores).
    args.cache_key = args.cache_key or f"rile_{args.lang}{'' if args.no_lang_filter else '_filtered'}"

    rng = np.random.default_rng(SEED)
    rows = load_rile_rows(args.input, args.party_prefix)
    if not args.no_lang_filter:
        # Country != language: Spain's manifestos include Catalan and Galician parties, the UK set
        # picked up Bloc Quebecois. Language is the variable under test, so off-language rows must go.
        before = len(rows)
        rows = [r for r in rows if detect_lang(r["text"]) == args.lang]
        print(f"language filter: kept {len(rows)} of {before} sentences detected as '{args.lang}'")
    probs, names = tone_probs(args.match_model, args.cache_key, [r["text"] for r in rows])
    tone = {id(r): float(v) for r, v in zip(rows, valence(args.match_model, probs, names))}
    out = {"rile_source": "MPDS2024a codebook s3.6 p.30", "cap": args.cap, "bins": args.bins,
           "match_model": args.match_model, "stages": {}}
    out["stages"]["raw"] = report("raw RILE", rows, tone, args.lang)

    # 1. cap each category
    capped = []
    for c, group in collections.defaultdict(list, {c: [r for r in rows if r["code"] == c]
                                                   for c in sorted({r["code"] for r in rows})}).items():
        capped += list(rng.choice(group, args.cap, replace=False)) if len(group) > args.cap else group
    out["stages"]["capped"] = report(f"capped at {args.cap}", capped, tone, args.lang)

    # 2. tone-match right against left, bin by bin
    edges = np.linspace(-1, 1, args.bins + 1)
    right = [r for r in capped if r["label"] == 1]
    left = [r for r in capped if r["label"] == 0]
    matched = []
    for k in range(len(edges) + 1):
        a = [r for r in right if np.digitize(tone[id(r)], edges) == k]
        b = [r for r in left if np.digitize(tone[id(r)], edges) == k]
        m = min(len(a), len(b))
        matched += list(rng.permutation(a)[:m]) + list(rng.permutation(b)[:m])
    out["stages"]["matched"] = report("tone-matched", matched, tone, args.lang)

    by_cat = collections.Counter(r["code"] for r in matched)
    out["category_counts"] = dict(sorted(by_cat.items()))
    out["n_manifestos"] = len({r["manifesto_id"] for r in matched})
    out["n_parties"] = len({r["partyname"] for r in matched})
    print(f"\n{len(matched)} sentences | {out['n_parties']} parties | {out['n_manifestos']} manifestos")
    print("largest categories:", by_cat.most_common(6))

    out["lang"] = args.lang
    out_path.write_text(json.dumps(
        [{"text": r["text"], "label": r["label"], "cmp_code": r["cmp_code"], "party": r["party"],
          "partyname": r["partyname"], "date": r["date"], "manifesto_id": r["manifesto_id"]} for r in matched],
        indent=1) + "\n", encoding="utf-8")
    (ROOT / "artifacts" / "results" / f"manifesto_{args.lang}_rile_build.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
