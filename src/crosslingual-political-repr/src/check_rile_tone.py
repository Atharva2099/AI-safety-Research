"""Summarize sentiment-model tone by language and RILE label.

    uv run --with torch --with transformers --with sentencepiece --with protobuf \
      python -m src.check_rile_tone

Scores all six languages in one Cardiff model load and saves aggregate statistics only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.metrics import auc, RES
from src.tone import score_texts, valence

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
MODEL_KEY = "cardiff"


def summarize(rows, values):
    labels = np.array([row["label"] for row in rows])
    scores = np.asarray(values)
    left, right = labels == 0, labels == 1
    mean = lambda mask: float(scores[mask].mean()) if mask.any() else None
    return {
        "n": len(rows), "n_left": int(left.sum()), "n_right": int(right.sum()),
        "mean_left": mean(left), "mean_right": mean(right),
        "tone_auc": auc(labels, scores) if left.any() and right.any() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/rile_v1/final_translations.jsonl")
    parser.add_argument("--out", type=Path, default=RES / "rile_v1_tone.json")
    parser.add_argument("--limit", type=int, help="score only the first rows for a quick smoke check")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    pairs = [(row, lang, row[lang]) for row in rows for lang in LANGUAGES]
    probs, names = score_texts(MODEL_KEY, [text for _, _, text in pairs])
    scores = valence(MODEL_KEY, probs, names)

    by_lang = {}
    for lang in LANGUAGES:
        positions = [i for i, (_, item_lang, _) in enumerate(pairs) if item_lang == lang]
        lang_rows = [pairs[i][0] for i in positions]
        lang_scores = scores[positions]
        by_lang[lang] = summarize(lang_rows, lang_scores)
        if lang in ("en", "de", "es"):
            native = [i for i in positions if pairs[i][0]["source_lang"] == lang]
            translated = [i for i in positions if pairs[i][0]["source_lang"] != lang]
            by_lang[lang]["native"] = summarize([pairs[i][0] for i in native], scores[native])
            by_lang[lang]["translated"] = summarize([pairs[i][0] for i in translated], scores[translated])

    result = {
        "model": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "model_revision": "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8",
        "n_items": len(rows), "languages": by_lang,
        "note": "zh and mr are exploratory because Cardiff training-language coverage is limited.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_items": len(rows), "languages": list(by_lang), "output": str(args.out)}))


if __name__ == "__main__":
    main()
