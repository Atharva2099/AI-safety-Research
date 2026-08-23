"""Extract layerwise activations and probe political polarity."""

import argparse
import json
import re
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError: StratifiedGroupKFold = None
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_statements(path: Path, limit: int | None) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    statements = [
        {"question_id": record["id"], "statement": statement["statement"], "polarity": statement["polarity"]}
        for record in records if record.get("is_suitable_for_probing") is True
        for statement in record.get("statements", [])
        if statement.get("polarity") in {1, -1}
    ]
    return statements[:limit] if limit is not None else statements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--data", default=str(ROOT / "data" / "declarative_statements.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    args = parser.parse_args()

    statements = load_statements(Path(args.data), args.limit)
    labels = [item["polarity"] for item in statements]
    if len(statements) < 10 or min(labels.count(-1), labels.count(1)) < 5:
        raise ValueError("Need at least 5 statements of each polarity for 5-fold CV")
    device = choose_device(args.device)
    dtype = (torch.bfloat16 if torch.cuda.is_available() and device.type == "cuda"
             and torch.cuda.is_bf16_supported() else
             torch.float16 if device.type == "mps" or device.type == "cuda" else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    activations: list[list] = []
    with torch.no_grad():
        for item in statements:
            inputs = tokenizer(item["statement"], return_tensors="pt", truncation=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            activations.append([hidden[0, -1, :].float().cpu().numpy() for hidden in outputs.hidden_states])

    X = [[activations[row][layer] for row in range(len(statements))]
         for layer in range(len(activations[0]))]
    groups = [item["question_id"] for item in statements]
    try:
        if StratifiedGroupKFold is None:
            raise ImportError
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(splitter.split(X[0], labels, groups))
    except (ImportError, ValueError):
        folds = list(GroupKFold(n_splits=5).split(X[0], labels, groups))
    output_path = Path(args.output_dir) / "results" / f"probe_results_{re.sub(r'[^A-Za-z0-9_.-]+', '_', args.model)}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            print("Layer                 Accuracy    F1")
            print("--------------------  ----------  ----------")
            for layer, features in enumerate(X):
                predictions = [None] * len(labels)
                sample_folds = [None] * len(labels)
                accuracies, f1_scores = [], []
                for fold, (train, test) in enumerate(folds):
                    probe = LogisticRegression(C=1.0, max_iter=1000)
                    probe.fit([features[i] for i in train], [labels[i] for i in train])
                    predicted = probe.predict([features[i] for i in test])
                    accuracies.append(accuracy_score([labels[i] for i in test], predicted))
                    f1_scores.append(f1_score([labels[i] for i in test], predicted, pos_label=1))
                    for index, prediction in zip(test, predicted):
                        predictions[index] = int(prediction)
                        sample_folds[index] = fold
                name = "Embedding Layer" if layer == 0 else f"Block {layer - 1}"
                accuracy, f1 = sum(accuracies) / 5, sum(f1_scores) / 5
                print(f"{name:<20}  {accuracy:>10.4f}  {f1:>10.4f}")
                for index, item in enumerate(statements):
                    result = {"layer": name, "question_id": item["question_id"], "statement": item["statement"],
                              "true_polarity": item["polarity"], "predicted_polarity": predictions[index],
                              "fold": sample_folds[index], "accuracy": accuracy, "f1": f1}
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
