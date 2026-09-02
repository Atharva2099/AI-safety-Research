"""Compute 6x6 cross-lingual transfer matrices with frozen, low-parameter MLPs."""

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]

PEAK_LAYERS = {
    "allenai/Olmo-3-7B-Instruct": 17,
    "mistralai/Ministral-8B-Instruct-2410": 31,
    "google/gemma-2-9b-it": 23,
    "Qwen/Qwen3.5-9B": 14,
}


SEEDS = (0, 1, 2, 3, 4)
HIDDEN_DIM = 8
MAX_EPOCHS = 200
PATIENCE = 15
INNER_VALIDATION_FRACTION = 0.2
INNER_SPLIT_SEED_OFFSET = 10_000
MIN_VALIDATION_IMPROVEMENT = 0.0


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FittedProbe:
    """The exact preprocessing and probe object used for all target languages."""

    def __init__(self, model: MLPProbe, scaler: StandardScaler):
        self.model = model
        self.scaler = scaler


def set_seed(seed: int) -> None:
    """Set all relevant RNGs; CUDA kernels can still have hardware caveats."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parameter_count(in_dim: int, hidden_dim: int = HIDDEN_DIM) -> int:
    return (in_dim * hidden_dim + hidden_dim) + (hidden_dim + 1)


def grouped_inner_split(train_idx: np.ndarray, groups: np.ndarray, outer_fold: int):
    """Make one initialization-independent, complete-group validation split."""
    train_groups = np.unique(groups[train_idx])
    rng = np.random.default_rng(INNER_SPLIT_SEED_OFFSET + outer_fold)
    shuffled = rng.permutation(train_groups)
    n_validation = max(1, int(round(len(train_groups) * INNER_VALIDATION_FRACTION)))
    validation_groups = set(shuffled[:n_validation].tolist())
    is_validation = np.isin(groups[train_idx], list(validation_groups))
    return train_idx[~is_validation], train_idx[is_validation]


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")


def _fit_epochs(model, X, y, device, epochs, lr, weight_decay, optimizer=None):
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    _require_finite("training features", X)
    _require_finite("training labels", y)
    model.train()
    last_loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(_tensor(X, device)), _tensor(y, device))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return last_loss


def _metrics(probe: FittedProbe, X: np.ndarray, y: np.ndarray, device: torch.device):
    _require_finite("evaluation features", X)
    _require_finite("evaluation labels", y)
    scaled = probe.scaler.transform(X)
    _require_finite("scaled features", scaled)
    probe.model.eval()
    with torch.inference_mode():
        logits = probe.model(_tensor(scaled, device))
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, _tensor(y, device)
        ).item()
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite validation or evaluation loss")
        predictions = (logits >= 0).long().cpu().numpy()
    return {
        "loss": float(loss),
        "accuracy": float(np.mean(predictions == y)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }


def fit_probe(
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    groups: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_fold: int,
    seed: int,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
) -> tuple[FittedProbe, dict]:
    """Select epochs on an inner source-only split, then refit on outer train."""
    _require_finite("outer training features", X_outer_train)
    _require_finite("outer training labels", y_outer_train)
    inner_train_idx, inner_validation_idx = grouped_inner_split(
        outer_train_idx, groups, outer_fold
    )
    inner_scaler = StandardScaler().fit(X_outer_train[inner_train_idx])
    X_inner = inner_scaler.transform(X_outer_train[inner_train_idx])
    _require_finite("scaled inner training features", X_inner)

    set_seed(seed)
    inner_model = MLPProbe(X_outer_train.shape[1]).to(device)
    inner_probe = FittedProbe(inner_model, inner_scaler)
    inner_optimizer = torch.optim.AdamW(inner_model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_loss = None
    best_epoch = None
    stale_epochs = 0
    validation_history = []
    termination_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        training_loss = _fit_epochs(inner_model, X_inner, y_outer_train[inner_train_idx], device, 1, lr, weight_decay, inner_optimizer)
        validation = _metrics(inner_probe, X_outer_train[inner_validation_idx], y_outer_train[inner_validation_idx], device)
        if not np.isfinite(training_loss) or not np.isfinite(validation["loss"]):
            raise FloatingPointError("non-finite training or validation loss")
        validation_history.append(validation["loss"])
        if (best_loss is None or
                validation["loss"] < best_loss - MIN_VALIDATION_IMPROVEMENT):
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(inner_model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                termination_reason = "early_stopping"
                break

    if best_state is None or best_epoch is None or best_loss is None:
        raise FloatingPointError("no finite validation checkpoint was found")
    inner_model.load_state_dict(best_state)
    source_validation = _metrics(inner_probe, X_outer_train[inner_validation_idx], y_outer_train[inner_validation_idx], device)
    source_inner_train = _metrics(inner_probe, X_outer_train[inner_train_idx], y_outer_train[inner_train_idx], device)

    # This is a fresh scaler/model, deliberately fit only after epoch selection.
    final_scaler = StandardScaler().fit(X_outer_train[outer_train_idx])
    final_X = final_scaler.transform(X_outer_train[outer_train_idx])
    _require_finite("scaled final training features", final_X)
    set_seed(seed)
    final_model = MLPProbe(X_outer_train.shape[1]).to(device)
    final_training_loss = _fit_epochs(final_model, final_X, y_outer_train[outer_train_idx], device, best_epoch, lr, weight_decay)
    if not np.isfinite(final_training_loss):
        raise FloatingPointError("non-finite final refit loss")
    final_probe = FittedProbe(final_model.eval(), final_scaler)
    source_training = _metrics(
        final_probe, X_outer_train[outer_train_idx], y_outer_train[outer_train_idx], device
    )
    return final_probe, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "termination_reason": termination_reason,
        "final_training_loss": final_training_loss,
        "source_training": source_training,
        "source_inner_train": source_inner_train,
        "source_validation": source_validation,
        "inner_train_indices": inner_train_idx.tolist(),
        "inner_validation_indices": inner_validation_idx.tolist(),
        "inner_validation_groups": int(len(np.unique(groups[inner_validation_idx]))),
        "validation_history_length": len(validation_history),
    }


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


def evaluate_mlp_matrix(
    features_by_lang: dict[str, np.ndarray],
    labels_by_lang: dict[str, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    seeds: tuple[int, ...] = SEEDS,
    shuffled_labels: bool = False,
    control_seed: int = 1729,
    progress: bool = True,
    fit_fn: Callable = fit_probe,
) -> dict:
    """Fit one frozen source probe per fold/seed and score it on every target."""
    cell_metrics = {source: {target: [] for target in LANGUAGES} for source in LANGUAGES}
    training_runs = []
    for source in LANGUAGES:
        source_groups = np.repeat(np.arange(len(features_by_lang[source]) // 2), 2)
        for outer_fold, (train_idx, test_idx) in enumerate(folds):
            y_train = labels_by_lang[source].copy()
            if shuffled_labels:
                # Each question keeps exactly one 0 and one 1, but its orientation
                # is random. This destroys stance alignment without target leakage.
                rng = np.random.default_rng(control_seed + outer_fold)
                for group in np.unique(source_groups[train_idx]):
                    pair = train_idx[source_groups[train_idx] == group]
                    if rng.integers(2):
                        y_train[pair] = y_train[pair[::-1]]
            for seed in (control_seed,) if shuffled_labels else seeds:
                if progress:
                    kind = "control" if shuffled_labels else "probe"
                    print(f"  {kind}: source={source} fold={outer_fold + 1}/{len(folds)} seed={seed}")
                probe, fit_metadata = fit_fn(
                    features_by_lang[source], y_train, source_groups, train_idx,
                    outer_fold, seed, device,
                )
                training_runs.append({
                    "source": source, "outer_fold": outer_fold, "seed": seed,
                    **fit_metadata,
                    "parameter_count": parameter_count(features_by_lang[source].shape[1]),
                })
                # No fitting occurs inside this target loop: `probe` is unchanged.
                for target in LANGUAGES:
                    metrics = _metrics(probe, features_by_lang[target][test_idx], labels_by_lang[target][test_idx], device)
                    cell_metrics[source][target].append({
                        "outer_fold": outer_fold, "seed": seed,
                        "accuracy": metrics["accuracy"], "f1": metrics["f1"],
                    })

    effective_seeds = [control_seed] if shuffled_labels else list(seeds)
    aggregation = summarize_accuracy(cell_metrics, effective_seeds, len(folds))
    return {
        "matrix": aggregation["mean_accuracy"],
        "pooled_fold_seed_accuracy_sd": aggregation["pooled_fold_seed_accuracy_sd"],
        "seed_level_accuracy_mean": aggregation["seed_level_accuracy_mean"],
        "seed_level_accuracy_sd": aggregation["seed_level_accuracy_sd"],
        "fold_level_accuracy_mean": aggregation["fold_level_accuracy_mean"],
        "fold_level_accuracy_sd": aggregation["fold_level_accuracy_sd"],
        "cell_metrics": cell_metrics,
        "training_runs": training_runs,
        "training_count": len(training_runs),
        "seed_list": effective_seeds,
        "initialization_seeds": effective_seeds,
        "initialization_seed_count": len(effective_seeds),
        "label_shuffle_seed": control_seed if shuffled_labels else None,
        "shuffled_label_control": shuffled_labels,
    }


def summarize_accuracy(cell_metrics: dict, seeds: tuple[int, ...], fold_count: int) -> dict:
    """Return means and SDs with the aggregation unit named explicitly."""
    means, pooled_sd, seed_means, seed_level_sd = [], [], [], []
    fold_means, fold_level_sd = [], []
    for source in LANGUAGES:
        mean_row, pooled_row, seed_mean_row, seed_sd_row = [], [], [], []
        fold_mean_row, fold_sd_row = [], []
        for target in LANGUAGES:
            entries = cell_metrics[source][target]
            values = np.array([entry["accuracy"] for entry in entries], dtype=float)
            mean_row.append(float(np.mean(values)))
            pooled_row.append(float(np.std(values)))
            by_seed = [np.mean([entry["accuracy"] for entry in entries if entry["seed"] == seed]) for seed in seeds]
            by_fold = [np.mean([entry["accuracy"] for entry in entries if entry["outer_fold"] == fold]) for fold in range(fold_count)]
            seed_mean_row.append(float(np.mean(by_seed)))
            seed_sd_row.append(float(np.std(by_seed)))
            fold_mean_row.append(float(np.mean(by_fold)))
            fold_sd_row.append(float(np.std(by_fold)))
        means.append(mean_row)
        pooled_sd.append(pooled_row)
        seed_means.append(seed_mean_row)
        fold_means.append(fold_mean_row)
        # These are appended as separate rows below to keep the returned shape clear.
        seed_level_sd.append(seed_sd_row)
        fold_level_sd.append(fold_sd_row)
    return {
        "mean_accuracy": means,
        "pooled_fold_seed_accuracy_sd": pooled_sd,
        "seed_level_accuracy_mean": seed_means,
        "seed_level_accuracy_sd": seed_level_sd,
        "fold_level_accuracy_mean": fold_means,
        "fold_level_accuracy_sd": fold_level_sd,
    }


def build_record(model_name: str, layer: int, features: dict[str, np.ndarray], folds: list,
                 primary: dict, control: dict | None = None) -> dict:
    """Build the self-describing artifact record without fitting or extraction."""
    return {
        "model": model_name,
        "layer": layer,
        "schema_version": 4,
        "probe_type": "one_hidden_layer_mlp",
        "languages": LANGUAGES,
        "matrix": primary["matrix"],
        "pooled_fold_seed_accuracy_sd": primary["pooled_fold_seed_accuracy_sd"],
        "seed_level_accuracy_mean": primary["seed_level_accuracy_mean"],
        "seed_level_accuracy_sd": primary["seed_level_accuracy_sd"],
        "fold_level_accuracy_mean": primary["fold_level_accuracy_mean"],
        "fold_level_accuracy_sd": primary["fold_level_accuracy_sd"],
        "aggregation": {
            "matrix": "mean accuracy over all outer-fold/initialization-seed entries",
            "pooled_fold_seed_accuracy_sd": "population SD over all fold/seed accuracies; not a confidence interval",
            "seed_level_accuracy_mean": "mean of the five seed-level means, where each seed-level mean averages folds",
            "seed_level_accuracy_sd": "population SD across the five seed-level means; not a confidence interval",
            "fold_level_accuracy_mean": "mean of the five fold-level means, where each fold-level mean averages seeds",
            "fold_level_accuracy_sd": "population SD across the five fold-level means; not a confidence interval",
        },
        "outer_splitter": {
            "name": "GroupKFold", "n_splits": len(folds),
            "grouped_by": "question ID",
            "folds": [{"train_indices": train.tolist(), "test_indices": test.tolist()}
                      for train, test in folds],
        },
        "architecture": {"input_standardization": "source_only", "hidden_dim": HIDDEN_DIM,
                          "activation": "ReLU", "layers": "Linear(in_dim, 8) -> ReLU -> Linear(8, 1)",
                          "input_dim": int(features[LANGUAGES[0]].shape[1]),
                          "parameter_count": parameter_count(features[LANGUAGES[0]].shape[1]),
                          "parameter_count_formula": "8 * in_dim + 17", "output": "one logit",
                          "preprocessing": "inner scaler on source inner-training rows; fresh final scaler on all source outer-training rows"},
        "training": {"loss": "BCEWithLogitsLoss", "optimizer": "AdamW", "lr": 1e-3,
                      "weight_decay": 1e-2, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                      "full_batch": True, "outer_folds": len(folds), "seeds": list(SEEDS),
                      "inner_validation_fraction": INNER_VALIDATION_FRACTION,
                      "inner_split_seed_rule": "NumPy default_rng(10000 + outer_fold)",
                      "selection_metric": "source validation BCEWithLogitsLoss",
                      "improvement_rule": "strict improvement with min_delta=0.0",
                      "final_refit": "fresh scaler and model fit on full outer-training source rows for selected epoch count",
                      "metric_semantics": {
                          "source_inner_train": "epoch-selection model, inner-training source rows",
                          "source_validation": "epoch-selection model, held-out source validation rows",
                          "source_training": "final fresh-refit model, all source outer-training rows",
                      },
                      "determinism": "Python/NumPy/torch CPU/CUDA seeds set; CUDA may retain kernel caveats"},
        "primary": {k: primary[k] for k in ("cell_metrics", "training_runs", "training_count",
                                              "seed_list", "initialization_seeds", "initialization_seed_count")},
        "control": control,
    }


def evaluate_model(model_name: str, layer: int, data: dict, folds: list, device: torch.device,
                   results_dir: Path, run_control: bool = False, control_seed: int = 1729):
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
    primary = evaluate_mlp_matrix(features, labels, folds, device)
    control = None
    if run_control:
        print("  Running bounded shuffled-label control...")
        control = evaluate_mlp_matrix(features, labels, folds, device,
                                       shuffled_labels=True, control_seed=control_seed)
    
    output_file = results_dir / f"mlp_cross_lingual_matrix_{slug}.json"
    record = build_record(model_name, layer, features, folds, primary, control)
    output_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "multilingual_statements.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shuffled-label-control", action="store_true")
    parser.add_argument("--control-seed", type=int, default=1729)
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
        evaluate_model(model_name, layer, data, folds, device, results_dir,
                       run_control=args.shuffled_label_control,
                       control_seed=args.control_seed)
        
    print("\nAll MLP cross-lingual transfer matrices successfully computed!")


if __name__ == "__main__":
    main()
