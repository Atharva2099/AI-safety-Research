"""Compute a strict out-of-fold cross-lingual probe-transfer matrix."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer

LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]


def load_data(path: Path) -> dict[str, list[dict]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    pairs = {}
    for record in records:
        if record.get("polarity") in {-1, 1}:
            pairs.setdefault(record["id"], []).append(record)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("Each question must have exactly two polarity records")
    return {language: [
        {"id": question_id, "text": record[language], "label": record["polarity"]}
        for question_id, pair in pairs.items() for record in sorted(pair, key=lambda x: x["polarity"])
    ] for language in LANGUAGES}


def extract(model, tokenizer, texts: list[str], layer: int, device: torch.device) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), 32):
        batch = tokenizer(texts[start:start + 32], return_tensors="pt", padding=True,
                          truncation=True).to(device)
        with torch.inference_mode():
            hidden = model(**batch, output_hidden_states=True).hidden_states[layer + 1]
        last = batch["attention_mask"].sum(dim=1) - 1
        vectors.append(hidden[torch.arange(hidden.size(0), device=device), last].float().cpu().numpy())
    return np.concatenate(vectors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True, help="Transformer block index")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "multilingual_statements.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data = load_data(args.data)
    ids = [item["id"] for item in data["en"]]
    if any([item["id"] for item in data[language]] != ids for language in LANGUAGES):
        raise ValueError("Languages are not strictly paired and aligned")
    folds = list(GroupKFold(n_splits=5).split(ids, groups=ids))
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    # Left padding shifts real tokens' RoPE positions by the pad count and
    # corrupts activations (hit Gemma: 86% -> 63%). Right padding keeps real
    # tokens at positions 0..L-1, matching single-sequence extraction.
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    features = {language: extract(model, tokenizer, [x["text"] for x in data[language]], args.layer, device)
                for language in LANGUAGES}
    labels = {language: np.array([x["label"] for x in data[language]]) for language in LANGUAGES}

    matrix = []
    for source in LANGUAGES:
        row = []
        for target in LANGUAGES:
            truth, predicted = [], []
            for train, test in folds:
                probe = LogisticRegression(C=1.0, max_iter=1000).fit(features[source][train], labels[source][train])
                predicted.extend(probe.predict(features[target][test]))
                truth.extend(labels[target][test])
            row.append(float(np.mean(np.asarray(predicted) == np.asarray(truth))))
        matrix.append(row)

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    output = ROOT / "artifacts" / "results" / f"cross_lingual_matrix_{slug}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": args.model, "layer": args.layer,
                                  "languages": LANGUAGES, "matrix": matrix}, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"model": args.model, "layer": args.layer, "matrix": matrix}))


if __name__ == "__main__":
    main()
