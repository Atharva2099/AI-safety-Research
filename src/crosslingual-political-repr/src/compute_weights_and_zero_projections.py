"""Compute probe weight cosine similarity matrices and 0-label neutral projections."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]

# Primary peak layer for each model
PEAK_LAYERS = {
    "allenai/Olmo-3-7B-Instruct": 17,
    "mistralai/Ministral-8B-Instruct-2410": 31,
    "google/gemma-2-9b-it": 23,
    "Qwen/Qwen3.5-9B": 14,
}


def load_dataset(path: Path):
    records = json.loads(path.read_text(encoding="utf-8"))
    
    # Paired +1 / -1 for training
    paired_pos = {lang: [] for lang in LANGUAGES}
    paired_neg = {lang: [] for lang in LANGUAGES}
    zero_items = {lang: [] for lang in LANGUAGES}
    
    for r in records:
        pol = r.get("polarity")
        if pol == 1:
            for lang in LANGUAGES:
                paired_pos[lang].append({"id": r["id"], "text": r[lang], "label": 1})
        elif pol == -1:
            for lang in LANGUAGES:
                paired_neg[lang].append({"id": r["id"], "text": r[lang], "label": -1})
        elif pol == 0:
            for lang in LANGUAGES:
                zero_items[lang].append({"id": r["id"], "text": r[lang], "label": 0})
                
    train_data = {}
    for lang in LANGUAGES:
        items = paired_pos[lang] + paired_neg[lang]
        items.sort(key=lambda x: (x["id"], x["label"]))
        train_data[lang] = items
        
    return train_data, zero_items


def extract_single_layer(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract final-token activations for one layer."""
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
        state = hidden_states[layer + 1]  # hidden_states[0] is embedding
        vectors = state[batch_idx, last].float().cpu().numpy()
        collected.append(vectors)

    return np.concatenate(collected, axis=0)


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def evaluate_model(model_name: str, layer: int, train_data: dict, zero_data: dict, device: torch.device, results_dir: Path):
    print(f"\n=======================================================")
    print(f"Evaluating {model_name} at Block {layer}")
    print(f"=======================================================")
    
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    batch_sz = 8 if "Qwen" in model_name or "gemma" in model_name else 16
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    
    # 1. Extract training activations (+1 and -1)
    train_features = {}
    train_labels = {}
    for lang in LANGUAGES:
        print(f"  Extracting train features ({lang})...")
        texts = [x["text"] for x in train_data[lang]]
        train_features[lang] = extract_single_layer(model, tokenizer, texts, layer, device, batch_size=batch_sz)
        train_labels[lang] = np.array([x["label"] for x in train_data[lang]])
        
    # 2. Extract zero-label activations (neutral)
    zero_features = {}
    for lang in LANGUAGES:
        print(f"  Extracting zero-label features ({lang})...")
        texts = [x["text"] for x in zero_data[lang]]
        zero_features[lang] = extract_single_layer(model, tokenizer, texts, layer, device, batch_size=batch_sz)
        
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 3. Fit linear probes and extract weight vectors W_lang
    probes = {}
    weights = {}
    biases = {}
    for lang in LANGUAGES:
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(train_features[lang], train_labels[lang])
        probes[lang] = clf
        weights[lang] = clf.coef_[0]  # shape (D,)
        biases[lang] = float(clf.intercept_[0])
        
    # 4. Compute 6x6 Cosine Similarity Matrix between probe weights
    cos_matrix = []
    for l1 in LANGUAGES:
        row = []
        for l2 in LANGUAGES:
            sim = cosine_sim(weights[l1], weights[l2])
            row.append(sim)
        cos_matrix.append(row)
        
    # Extract unique off-diagonal pairs
    pair_similarities = {}
    for i in range(len(LANGUAGES)):
        for j in range(i + 1, len(LANGUAGES)):
            pair_key = f"{LANGUAGES[i]}_{LANGUAGES[j]}"
            pair_similarities[pair_key] = cos_matrix[i][j]
            
    cos_output_file = results_dir / f"probe_weight_cosine_matrix_{slug}.json"
    cos_record = {
        "model": model_name,
        "layer": layer,
        "languages": LANGUAGES,
        "cosine_matrix": cos_matrix,
        "unique_pair_similarities": pair_similarities,
        "off_diagonal_mean": float(np.mean(list(pair_similarities.values()))),
    }
    cos_output_file.write_text(json.dumps(cos_record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {cos_output_file.name} (Off-diagonal mean cos = {cos_record['off_diagonal_mean']:.4f})")
    
    # 5. Compute 0-label neutral calibration projection analysis
    # Decision function: score = (X · W) + b
    # For each language probe, project -1, 0, +1 statements from all languages
    projection_results = {}
    for probe_lang in LANGUAGES:
        w = weights[probe_lang]
        b = biases[probe_lang]
        
        projection_results[probe_lang] = {}
        for eval_lang in LANGUAGES:
            # Positive statements (+1)
            pos_idx = np.where(train_labels[eval_lang] == 1)[0]
            neg_idx = np.where(train_labels[eval_lang] == -1)[0]
            
            pos_scores = (np.dot(train_features[eval_lang][pos_idx], w) + b).tolist()
            neg_scores = (np.dot(train_features[eval_lang][neg_idx], w) + b).tolist()
            zero_scores = (np.dot(zero_features[eval_lang], w) + b).tolist()
            
            projection_results[probe_lang][eval_lang] = {
                "neg_mean": float(np.mean(neg_scores)),
                "neg_std": float(np.std(neg_scores)),
                "zero_mean": float(np.mean(zero_scores)),
                "zero_std": float(np.std(zero_scores)),
                "pos_mean": float(np.mean(pos_scores)),
                "pos_std": float(np.std(pos_scores)),
                "raw_neg_scores": neg_scores,
                "raw_zero_scores": zero_scores,
                "raw_pos_scores": pos_scores,
            }
            
    proj_output_file = results_dir / f"neutral_zero_projection_{slug}.json"
    proj_record = {
        "model": model_name,
        "layer": layer,
        "languages": LANGUAGES,
        "projections": projection_results,
    }
    proj_output_file.write_text(json.dumps(proj_record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {proj_output_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "multilingual_statements.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    train_data, zero_data = load_dataset(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device in {"auto", "cuda"} else "cpu")
    results_dir = ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_run = (
        PEAK_LAYERS.items()
        if args.model == "all"
        else [(args.model, PEAK_LAYERS[args.model])]
    )
    
    for model_name, layer in models_to_run:
        evaluate_model(model_name, layer, train_data, zero_data, device, results_dir)
        
    print("\nWeight cosine matrices and 0-label projections successfully computed!")


if __name__ == "__main__":
    main()
