"""Tone validator gate: can a sentiment model that took no part in the matching still separate the sides?

Matching guarantees only that the matching model is blind to the label. The gate re-tests with
independent models trained on other domains, per language. A set passes when no independent model
sits further than THRESHOLD from AUC 0.50 in either direction (being wrong in reverse is equally
exploitable). Failures are reported, not retried with a different model.

    # versioned corpus (items.jsonl + text_<lang>.jsonl)
    uv run --with torch --with transformers --with sentencepiece --with protobuf \
        python -m src.validator_gate --version v1 --langs en de es

    # single-file sets from the earlier build (data/manifesto_<lang>_rile.json)
    uv run ... python -m src.validator_gate --legacy --langs en de es

Writes artifacts/results/rile_gate_<tag>.json. Reads cached tone scores; a cache miss re-scores.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.metrics import RES, ROOT, auc
from src.tone import TONE_MODELS, VALIDATORS_BY_LANG, tone_probs, valence

THRESHOLD = 0.05


def load_legacy(lang):
    rows = json.loads((ROOT / "data" / f"manifesto_{lang}_rile.json").read_text())
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def load_versioned(version, lang):
    d = ROOT / "data" / f"rile_{version}"
    labels = {r["item_id"]: r["label"] for r in map(json.loads, (d / "items.jsonl").read_text().splitlines())}
    rows = [json.loads(l) for l in (d / f"text_{lang}.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["item_id"] in labels]
    return [r["text"] for r in rows], np.array([labels[r["item_id"]] for r in rows])


def gate(lang, texts, y, cache_key, matched_on):
    """AUC per model; the verdict ignores the matching model, which is blind by construction."""
    res = {}
    for key in [matched_on] + [v for v in VALIDATORS_BY_LANG[lang] if v != matched_on]:
        probs, names = tone_probs(key, cache_key, texts)
        res[key] = float(auc(y, valence(key, probs, names)))
        tag = "  (matched on)" if key == matched_on else ""
        print(f"  {key:<12} {TONE_MODELS[key][0]:<56} {res[key]:.3f}{tag}")
    worst = max(abs(a - 0.5) for k, a in res.items() if k != matched_on)
    verdict = "PASS" if worst < THRESHOLD else "FAIL"
    print(f"  worst independent validator: {worst:.3f} from 0.50 -> {verdict}\n")
    return {"n": len(texts), "matched_on": matched_on, "aucs": res,
            "worst_gap": worst, "threshold": THRESHOLD, "verdict": verdict}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["en", "de", "es"])
    p.add_argument("--version", default="v1")
    p.add_argument("--legacy", action="store_true", help="read data/manifesto_<lang>_rile.json instead")
    p.add_argument("--matched-on", default=None,
                   help="model used for matching (default: spanish for es, cardiff otherwise)")
    p.add_argument("--cache-suffix", default=None, help="tone cache key suffix; defaults to the source tag")
    args = p.parse_args()

    tag = "legacy" if args.legacy else args.version
    out = {"source": tag, "threshold": THRESHOLD, "langs": {}}
    for lang in args.langs:
        texts, y = load_legacy(lang) if args.legacy else load_versioned(args.version, lang)
        matched_on = args.matched_on or ("spanish" if lang == "es" else "cardiff")
        key = f"{lang}_rile_{args.cache_suffix or tag}"
        print(f"=== {lang}: {len(texts)} sentences (right {int(y.sum())} / left {int((1 - y).sum())}) ===")
        out["langs"][lang] = gate(lang, texts, y, key, matched_on)

    path = RES / f"rile_gate_{tag}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("verdicts:", {k: v["verdict"] for k, v in out["langs"].items()}, f"-> {path.name}")


if __name__ == "__main__":
    main()
