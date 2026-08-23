"""Compute 6x6 cross-lingual probe-transfer matrices for all unique peak layers."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer

LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIGS = {
    "allenai/Olmo-3-7B-Instruct": {
        "unique_layers": [10, 15, 17, 18],
        "layer_anchors": {
            10: ["es"],
            15: ["hi"],
            17: ["en", "de", "zh"],
            18: ["mr"],
        },
    },
    "mistralai/Ministral-8B-Instruct-2410": {
        "unique_layers": [9, 20, 31, 32, 33],
        "layer_anchors": {
            9: ["hi"],
            20: ["zh"],
            31: ["en", "de"],
            32: ["es"],
            33: ["mr"],
        },
    },
    "google/gemma-2-9b-it": {
        "unique_layers": [13, 15, 18, 22, 23, 24, 25],
        "layer_anchors": {
            13: ["mr"],
            15: ["en"],
            18: ["zh"],
            22: ["de"],
            23: ["RQ1-en-peak"],
            24: ["hi"],
            25: ["es"],
        },
    },
    "Qwen/Qwen3.5-9B": {
        "unique_layers": [12, 14, 16],
        "layer_anchors": {
            12: ["zh"],
            14: ["en", "es", "de"],
            16: ["hi"],
        },
        "scan_mr_all_layers": True,
    },
}


def load_data(path: Path) -> dict[str, list[dict]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    pairs = {}
    for record in records:
        if record.get("polarity") in {-1, 1}:
            pairs.setdefault(record["id"], []).append(record)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("Each question must have exactly two polarity records")
    return {
        language: [
            {"id": question_id, "text": record[language], "label": record["polarity"]}
            for question_id, pair in pairs.items()
            for record in sorted(pair, key=lambda x: x["polarity"])
        ]
        for language in LANGUAGES
    }


def extract_layers(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    device: torch.device,
    batch_size: int = 8,
) -> dict[int, np.ndarray]:
    """Extract final-token activations for multiple layers in a single pass."""
    collected = {l: [] for l in layers}
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
        for layer in layers:
            state = hidden_states[layer + 1]  # hidden_states[0] is embedding
            vectors = state[batch_idx, last].float().cpu().numpy()
            collected[layer].append(vectors)

    return {layer: np.concatenate(collected[layer], axis=0) for layer in layers}


def _fit_single_pair(source_feats, source_labels, target_feats, target_labels, train_idx, test_idx):
    probe = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs").fit(
        source_feats[train_idx], source_labels[train_idx]
    )
    preds = probe.predict(target_feats[test_idx])
    return (preds == target_labels[test_idx]).tolist()


def evaluate_6x6_matrix(
    features_by_lang: dict[str, np.ndarray],
    labels_by_lang: dict[str, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[list[float]]:
    """Compute the 6x6 out-of-fold transfer accuracy matrix in parallel."""
    tasks = []
    task_keys = []
    for s_idx, source in enumerate(LANGUAGES):
        for t_idx, target in enumerate(LANGUAGES):
            for train, test in folds:
                tasks.append((
                    features_by_lang[source], labels_by_lang[source],
                    features_by_lang[target], labels_by_lang[target],
                    train, test
                ))
                task_keys.append((s_idx, t_idx))

    results = Parallel(n_jobs=4, batch_size=5)(
        delayed(_fit_single_pair)(*args) for args in tasks
    )

    matrix = [[0.0] * len(LANGUAGES) for _ in LANGUAGES]
    pair_correct = {}
    for (s_idx, t_idx), bool_list in zip(task_keys, results):
        pair_correct.setdefault((s_idx, t_idx), []).extend(bool_list)

    for (s_idx, t_idx), matches in pair_correct.items():
        matrix[s_idx][t_idx] = float(np.mean(matches))

    return matrix


def _fit_single_layer_mr(mr_feats_layer, mr_labels, train, test):
    probe = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs").fit(
        mr_feats_layer[train], mr_labels[train]
    )
    preds = probe.predict(mr_feats_layer[test])
    return (preds == mr_labels[test]).tolist()


def run_model(model_name: str, config: dict, data: dict, folds: list, device: torch.device, results_dir: Path):
    print(f"\n=======================================================")
    print(f"Running model: {model_name}")
    print(f"=======================================================")
    
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    
    target_layers = list(config["unique_layers"])
    layer_anchors = dict(config.get("layer_anchors", {}))
    
    # If we need to scan MR across all layers first (e.g. for Qwen 3.5)
    if config.get("scan_mr_all_layers"):
        num_layers = getattr(model.config, "num_hidden_layers", None) or getattr(getattr(model.config, "text_config", None), "num_hidden_layers", None) or 32
        all_layers = list(range(num_layers))
        print(f"Scanning MR in-language across all {num_layers} blocks in parallel...")
        mr_features = extract_layers(model, tokenizer, [x["text"] for x in data["mr"]], all_layers, device)
        mr_labels = np.array([x["label"] for x in data["mr"]])
        
        mr_tasks = []
        mr_task_keys = []
        for l in all_layers:
            for train, test in folds:
                mr_tasks.append((mr_features[l], mr_labels, train, test))
                mr_task_keys.append(l)
                
        mr_results = Parallel(n_jobs=4, batch_size=5)(
            delayed(_fit_single_layer_mr)(*args) for args in mr_tasks
        )
        
        mr_layer_accs = {}
        layer_matches = {}
        for l, bool_list in zip(mr_task_keys, mr_results):
            layer_matches.setdefault(l, []).extend(bool_list)
        for l, matches in layer_matches.items():
            acc = float(np.mean(matches))
            mr_layer_accs[l] = acc
            print(f"  MR Block {l}: {acc*100:.2f}%")
        
        best_mr_layer = max(mr_layer_accs.items(), key=lambda x: x[1])[0]
        print(f"--> MR Peak for {model_name} is Block {best_mr_layer} ({mr_layer_accs[best_mr_layer]*100:.2f}%)")
        
        if best_mr_layer not in target_layers:
            target_layers.append(best_mr_layer)
            layer_anchors[best_mr_layer] = ["mr"]
        else:
            layer_anchors[best_mr_layer].append("mr")
    
    target_layers = sorted(set(target_layers))
    print(f"Extracting features for unique layers: {target_layers}")
    
    features_by_layer_lang = {l: {} for l in target_layers}
    labels_by_lang = {l: np.array([x["label"] for x in data[l]]) for l in LANGUAGES}
    
    for language in LANGUAGES:
        print(f"  Extracting {language}...")
        lang_layer_feats = extract_layers(
            model, tokenizer, [x["text"] for x in data[language]], target_layers, device
        )
        for l in target_layers:
            features_by_layer_lang[l][language] = lang_layer_feats[l]
            
    # Now compute and save 6x6 matrix for each unique layer
    for layer in target_layers:
        anchors = layer_anchors.get(layer, [])
        print(f"\nComputing 6x6 matrix for Block {layer} in parallel (anchors: {anchors})...")
        matrix = evaluate_6x6_matrix(features_by_layer_lang[layer], labels_by_lang, folds)
        
        out_file = results_dir / f"cross_lingual_matrix_{slug}_block{layer}.json"
        record = {
            "model": model_name,
            "layer": layer,
            "anchoring_languages": anchors,
            "languages": LANGUAGES,
            "matrix": matrix,
        }
        out_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {out_file.name}")
        
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", help="Specific model or 'all'")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "multilingual_statements.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    data = load_data(args.data)
    ids = [item["id"] for item in data["en"]]
    folds = list(GroupKFold(n_splits=5).split(ids, groups=ids))
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device in {"auto", "cuda"} else "cpu")
    results_dir = ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_run = (
        MODEL_CONFIGS.items()
        if args.model == "all"
        else [(args.model, MODEL_CONFIGS[args.model])]
    )
    
    for model_name, config in models_to_run:
        run_model(model_name, config, data, folds, device, results_dir)
        
    print("\nAll peak-layer matrices successfully computed!")


if __name__ == "__main__":
    main()
