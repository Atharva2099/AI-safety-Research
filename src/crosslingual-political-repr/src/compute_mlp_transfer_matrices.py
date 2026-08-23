"""Compute 6x6 cross-lingual probe-transfer matrices using a 2-layer MLP probe."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer

LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]

PEAK_LAYERS = {
    "allenai/Olmo-3-7B-Instruct": 17,
    "mistralai/Ministral-8B-Instruct-2410": 31,
    "google/gemma-2-9b-it": 23,
    "Qwen/Qwen3.5-9B": 14,
}


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_dataset(path: Path):
    records = json.loads(path.read_text(encoding="utf-8"))
    pairs = {}
    for r in records:
        if r.get("polarity") in {-1, 1}:
            pairs.setdefault(r["id"], []).append(r)
    if any(len(p) != 2 for p in pairs.values()):
        raise ValueError("Each question must have exactly two polarity records")
    return {
        lang: [
            {"id": q_id, "text": r[lang], "label": 1 if r["polarity"] == 1 else 0}
            for q_id, pair in pairs.items()
            for r in sorted(pair, key=lambda x: x["polarity"])
        ]
        for lang in LANGUAGES
    }


def extract_single_layer(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    collected = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.inference_mode():
            outputs = model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        last = batch["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(last.size(0), device=device)
        state = hidden_states[layer + 1]
        vectors = state[batch_idx, last].float().cpu().numpy()
        collected.append(vectors)

    return np.concatenate(collected, axis=0)


def train_and_eval_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
) -> float:
    """Train MLP probe on GPU and evaluate test accuracy."""
    in_dim = X_train.shape[1]
    model = MLPProbe(in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_te)
        preds = (torch.sigmoid(test_logits) >= 0.5).long().cpu().numpy()

    return float(np.mean(preds == y_test))


def evaluate_mlp_matrix(
    features_by_lang: dict[str, np.ndarray],
    labels_by_lang: dict[str, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> list[list[float]]:
    matrix = []
    for source in LANGUAGES:
        row = []
        for target in LANGUAGES:
            fold_accs = []
            for train_idx, test_idx in folds:
                acc = train_and_eval_mlp(
                    features_by_lang[source][train_idx],
                    labels_by_lang[source][train_idx],
                    features_by_lang[target][test_idx],
                    labels_by_lang[target][test_idx],
                    device,
                )
                fold_accs.append(acc)
            row.append(float(np.mean(fold_accs)))
        matrix.append(row)
    return matrix


def evaluate_model(model_name: str, layer: int, data: dict, folds: list, device: torch.device, results_dir: Path):
    print(f"\n=======================================================")
    print(f"Evaluating MLP Probing on {model_name} at Block {layer}")
    print(f"=======================================================")
    
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    batch_sz = 8 if "Qwen" in model_name or "gemma" in model_name else 16
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    
    features = {}
    labels = {}
    for lang in LANGUAGES:
        print(f"  Extracting features ({lang})...")
        texts = [x["text"] for x in data[lang]]
        features[lang] = extract_single_layer(model, tokenizer, texts, layer, device, batch_size=batch_sz)
        labels[lang] = np.array([x["label"] for x in data[lang]])
        
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("  Training and evaluating 6x6 MLP transfer matrix...")
    matrix = evaluate_mlp_matrix(features, labels, folds, device)
    
    output_file = results_dir / f"mlp_cross_lingual_matrix_{slug}.json"
    record = {
        "model": model_name,
        "layer": layer,
        "probe_type": "2_layer_mlp",
        "languages": LANGUAGES,
        "matrix": matrix,
    }
    output_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "multilingual_statements.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    data = load_dataset(args.data)
    ids = [item["id"] for item in data["en"]]
    folds = list(GroupKFold(n_splits=5).split(ids, groups=ids))
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device in {"auto", "cuda"} else "cpu")
    results_dir = ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_run = (
        PEAK_LAYERS.items()
        if args.model == "all"
        else [(args.model, PEAK_LAYERS[args.model])]
    )
    
    for model_name, layer in models_to_run:
        evaluate_model(model_name, layer, data, folds, device, results_dir)
        
    print("\nAll MLP cross-lingual transfer matrices successfully computed!")


if __name__ == "__main__":
    main()
