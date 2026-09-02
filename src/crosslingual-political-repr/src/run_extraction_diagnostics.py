"""Final RQ2 representation conditions, tokenizer diagnostics, and probes."""

import argparse
import gzip
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
CONDITIONS = ("current_raw", "current_l2", "content_raw", "content_l2", "mean_raw", "mean_l2", "stripped_raw", "stripped_l2")
STRATEGIES = ("current", "content", "mean", "stripped")
TARGETS = {
    "allenai/Olmo-3-7B-Instruct": 17,
    "mistralai/Ministral-8B-Instruct-2410": 31,
    "google/gemma-2-9b-it": 23,
    "Qwen/Qwen3.5-9B": 12,
}
MAX_ITER = 1000
SOLVER_TOL = 1e-4
C_CANDIDATES = (.0001, .001, .01, .1, 1, 10)


def load_data(path: Path):
    records = json.loads(path.read_text(encoding="utf-8"))
    pairs = {}
    for row in records:
        if row.get("polarity") in (-1, 1):
            pairs.setdefault(row["id"], []).append(row)
    if len(pairs) != 580 or any(len(v) != 2 for v in pairs.values()):
        raise ValueError("Expected 580 paired question IDs")
    if any({r["polarity"] for r in v} != {-1, 1} for v in pairs.values()):
        raise ValueError("Pairs must have opposite labels")
    out = {l: [] for l in LANGUAGES}
    for qid, pair in pairs.items():
        for row in sorted(pair, key=lambda r: r["polarity"]):
            for language in LANGUAGES:
                out[language].append({"question_id": qid, "statement": row[language], "label": int(row["polarity"] == 1)})
    ids = [x["question_id"] for x in out["en"]]
    if any([x["question_id"] for x in out[l]] != ids for l in LANGUAGES):
        raise ValueError("Language records are not aligned")
    return out


def _is_punctuation(text: str) -> bool:
    """True if non-empty text consists entirely of Unicode punctuation characters (categories P*)."""
    cleaned = text.strip()
    return bool(cleaned) and all(unicodedata.category(c).startswith("P") for c in cleaned)


def _token_surface(tokenizer, token_str: str) -> str:
    """Extract clean surface text from a token string, stripping BPE/SentencePiece marker artifacts."""
    return tokenizer.convert_tokens_to_string([token_str]).strip()


def _strip_terminal_punctuation(text: str) -> str:
    """Remove trailing whitespace, then trailing Unicode punctuation (categories P*), then whitespace."""
    stripped = text.rstrip()
    end = len(stripped)
    while end > 0 and unicodedata.category(stripped[end - 1]).startswith("P"):
        end -= 1
    return stripped[:end].rstrip()


def _token_strings(tokenizer, ids):
    return tokenizer.convert_ids_to_tokens([int(x) for x in ids])


def tokenize_diagnostics(tokenizer, text: str, max_length=None):
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = list(encoded["input_ids"])
    if max_length is not None and len(ids) > max_length:
        raise ValueError("Statement would be truncated")
    tokens = _token_strings(tokenizer, ids)
    special = set(getattr(tokenizer, "all_special_ids", []))
    attention = np.asarray(encoded.get("attention_mask", [1] * len(ids)))
    surfaces = [_token_surface(tokenizer, tok) for tok in tokens]
    content = [i for i, (tid, surf, mask) in enumerate(zip(ids, surfaces, attention))
               if mask and tid not in special and not _is_punctuation(surf)]
    if not content:
        raise AssertionError("Statement has no content-bearing tokens")
    final = text.rstrip()[-1:] or ""
    current_position = int(np.flatnonzero(attention)[-1])
    terminal_token = tokens[current_position]
    terminal_surface = surfaces[current_position]
    last_content_pos = content[-1]
    last_content_surface = surfaces[last_content_pos]
    return {"token_count": int(sum(attention)), "chars_per_token": len(text) / max(1, int(sum(attention))),
            "special_count": sum(int(tid in special) for tid in ids),
            "unknown_count": sum(int(tid == getattr(tokenizer, "unk_token_id", None)) for tid in ids),
            "byte_fallback": sum(int("byte" in tok.lower() or tok.startswith("<0x")) for tok in tokens),
            "truncated": False, "terminal_punctuation": final,
            "terminal_is_punctuation": _is_punctuation(terminal_surface),
            "terminal_content_position": last_content_pos,
            "terminal_content_token_id": int(ids[last_content_pos]),
            "terminal_content_is_punctuation": _is_punctuation(last_content_surface),
            "content_positions": content, "current_position": current_position,
            "terminal_token_id": int(ids[current_position]), "tokens": tokens}


def l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise AssertionError("Cannot L2-normalize a non-finite or zero vector")
    return vector / norm


def extract_conditions(model, tokenizer, texts, layer, device="cpu", max_length=None, batch_size=16):
    """Extract all eight conditions; no input is silently truncated."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tokenizer.padding_side = "right"
    encoded = [tokenizer(t, add_special_tokens=True, truncation=False) for t in texts]
    diagnostics = [tokenize_diagnostics(tokenizer, t, max_length) for t in texts]
    stripped = [_strip_terminal_punctuation(t) for t in texts]
    stripped_encoded = [tokenizer(t, add_special_tokens=True, truncation=False) for t in stripped]
    if max_length is not None and any(len(x["input_ids"]) > max_length for x in stripped_encoded):
        raise ValueError("Stripped statement would be truncated")
    pad = tokenizer.pad_token_id
    result = {name: [] for name in CONDITIONS}
    torch = __import__("torch")
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            stop = min(start + batch_size, len(texts))
            batch = tokenizer.pad(encoded[start:stop], padding=True, return_tensors="pt")
            stripped_batch = tokenizer.pad(stripped_encoded[start:stop], padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            stripped_batch = {k: v.to(device) for k, v in stripped_batch.items()}
            hidden = model(**batch, output_hidden_states=True).hidden_states[layer + 1]
            stripped_hidden = model(**stripped_batch, output_hidden_states=True).hidden_states[layer + 1]
            attention = batch["attention_mask"]
            current = attention.sum(dim=1) - 1
            stripped_position = stripped_batch["attention_mask"].sum(dim=1) - 1
            content_lengths = torch.tensor(
                [len(d["content_positions"]) for d in diagnostics[start:stop]],
                device=device,
            )
            max_content = int(content_lengths.max())
            content_indices = torch.zeros((stop - start, max_content), dtype=torch.long, device=device)
            content_mask = torch.zeros((stop - start, max_content), dtype=torch.bool, device=device)
            for batch_row, diag in enumerate(diagnostics[start:stop]):
                positions = torch.tensor(diag["content_positions"], dtype=torch.long, device=device)
                content_indices[batch_row, :len(positions)] = positions
                content_mask[batch_row, :len(positions)] = True
            hidden_float = hidden.float()
            gathered_content = hidden_float.gather(
                1, content_indices.unsqueeze(-1).expand(-1, -1, hidden_float.shape[-1])
            )
            mean = (gathered_content * content_mask.unsqueeze(-1)).sum(dim=1) / content_lengths[:, None]
            row_indices = torch.arange(stop - start, device=device)
            last_content_idx = content_lengths - 1
            terminal_content_positions = content_indices[row_indices, last_content_idx]
            batch_vectors = {
                "current": hidden_float[row_indices, current],
                "content": hidden_float[row_indices, terminal_content_positions],
                "mean": mean,
                "stripped": stripped_hidden.float()[row_indices, stripped_position],
            }
            for batch_row, (row, diag) in enumerate(zip(range(start, stop), diagnostics[start:stop])):
                vectors = {k: v[batch_row].cpu().numpy() for k, v in batch_vectors.items()}
                row_stripped_position = int(stripped_position[batch_row])
                diag["raw_vector_norms"] = {k: float(np.linalg.norm(v)) for k, v in vectors.items()}
                diag["stripped_position"] = row_stripped_position
                diag["padding_token_id"] = pad
                for strategy, vector in vectors.items():
                    result[strategy + "_raw"].append(vector)
                    result[strategy + "_l2"].append(l2_normalize(vector))
            del hidden, stripped_hidden, hidden_float, gathered_content, mean
    return {k: np.asarray(v) for k, v in result.items()}, diagnostics


def grouped_folds(question_ids, n_splits=5):
    ids = np.asarray(question_ids)
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(ids)), groups=ids))


def select_c(X, y, groups, train_idx, candidates=C_CANDIDATES):
    """Mean inner-validation accuracy over grouped folds; ties resolve to the smallest C."""
    sorted_candidates = sorted(candidates)
    inner = list(GroupKFold(n_splits=3).split(X[train_idx], y[train_idx], groups=groups[train_idx]))
    best_c, best_score = None, -np.inf
    inner_fits = []
    for c in sorted_candidates:
        fold_scores = []
        for a, b in inner:
            clf = LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER, tol=SOLVER_TOL)
            clf.fit(X[train_idx][a], y[train_idx][a])
            n_iter = int(np.max(clf.n_iter_))
            fold_scores.append(clf.score(X[train_idx][b], y[train_idx][b]))
            inner_fits.append({"C": float(c), "n_iter": n_iter, "converged": bool(n_iter < MAX_ITER)})
        score = float(np.mean(fold_scores))
        if score > best_score:
            best_c, best_score = c, score
    return best_c, inner_fits


def probe_records(features_by_language, labels_by_language, question_ids_by_language, lane="fair",
                  source_language="en", norm_only=False, conditions=None):
    """Fit once per source/fold/condition and emit one auditable row per target prediction."""
    if lane == "legacy":
        conditions = ("current_raw",)
    elif conditions is None:
        conditions = CONDITIONS
    source_ids = np.asarray(question_ids_by_language[source_language])
    folds = grouped_folds(source_ids)
    records = []
    fits = []
    for condition, source_X in features_by_language[source_language].items():
        if condition not in conditions:
            continue
        if norm_only and condition not in {f"{strategy}_raw" for strategy in STRATEGIES}:
            continue
        X = np.log(np.linalg.norm(source_X, axis=1) + 1e-8)[:, None] if norm_only else source_X
        for fold, (train, test) in enumerate(folds):
            if lane == "legacy":
                c, inner_fits = 1.0, []
            else:
                c, inner_fits = select_c(X, labels_by_language[source_language], source_ids, train)
            feature_mode = "norm_only" if norm_only else "full"
            fit_id = f"{source_language}:{condition}:{lane}:{feature_mode}:fold{fold}"
            probe = LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER, tol=SOLVER_TOL).fit(
                X[train], labels_by_language[source_language][train])
            n_iter = int(np.max(probe.n_iter_))
            fits.append({"fit_id": fit_id, "source_language": source_language, "condition": condition,
                         "lane": lane, "norm_only": norm_only, "selected_C": c, "n_iter": n_iter,
                         "converged": bool(n_iter < MAX_ITER), "inner_fits_converged": all(f["converged"] for f in inner_fits)})
            for target in LANGUAGES:
                target_features = (np.log(np.linalg.norm(features_by_language[target][condition], axis=1) + 1e-8)[:, None]
                                   if norm_only else features_by_language[target][condition])
                target_ids = question_ids_by_language[target]
                target_X = target_features[test]
                scores = probe.decision_function(target_X)
                predictions = probe.predict(target_X)
                probabilities = probe.predict_proba(target_X)[:, 1]
                records.extend({"condition": condition, "source_language": source_language, "target_language": target,
                                "lane": lane, "feature_mode": feature_mode,
                                "row_index": int(row_idx), "question_id": int(qid),
                                "truth": int(truth), "pred": int(pred), "proba": float(proba), "decision_score": float(score),
                                "fold": fold, "selected_C": c, "fit_id": fit_id, "norm_only": norm_only}
                               for row_idx, qid, truth, pred, proba, score in zip(
                                   test, target_ids[test], labels_by_language[target][test], predictions, probabilities, scores))
    return records, fits


def validate_layer(model: str, layer: int):
    if model in TARGETS and layer != TARGETS[model]:
        raise ValueError(f"{model} is pinned to layer {TARGETS[model]}, got {layer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "multilingual_statements.json")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--lanes", default="legacy,fair", help="Comma-separated: legacy,fair (smoke tests may narrow this)")
    parser.add_argument("--conditions", default=",".join(CONDITIONS), help="Fair-lane condition subset for smoke tests")
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="Optional .npz path; enables probe-only retry without re-extraction")
    args = parser.parse_args()
    validate_layer(args.model, args.layer)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.output is None:
        model_slug = args.model.replace("/", "_")
        args.output = ROOT / "artifacts" / "results" / f"extraction_diagnostics_{model_slug}.json"
    lanes = tuple(l.strip() for l in args.lanes.split(",") if l.strip())
    fair_conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    unknown = [c for c in fair_conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    import sklearn
    import transformers
    import torch
    import hashlib
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    data = load_data(args.data)
    probe_environment = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "scikit_learn": sklearn.__version__,
        "probe": {"solver": "lbfgs", "max_iter": MAX_ITER, "tol": SOLVER_TOL,
                  "C_candidates": list(C_CANDIDATES), "tie_rule": "smallest C"},
    }
    all_results = {"model": args.model, "layer": args.layer, "languages": LANGUAGES,
                   "schema_version": 2, "conditions_version": 2,
                   "data_hash": data_hash,
                   "conditions": {}, "predictions": [], "fits": [], "probe_environment": probe_environment}
    feature_by_language = {}
    if args.feature_cache is not None and args.feature_cache.exists():
        loaded = np.load(args.feature_cache, allow_pickle=False)
        cached_meta = json.loads(str(loaded["metadata"]))
        if (cached_meta.get("model") != args.model or cached_meta.get("layer") != args.layer
                or cached_meta.get("data_hash") != data_hash
                or cached_meta.get("conditions_version") != 2):
            raise ValueError(f"Feature cache fingerprint mismatch: {cached_meta} vs {args.model}:{args.layer}:{data_hash[:8]}")
        feature_by_language = {l: {c: loaded[f"{l}:{c}"] for c in CONDITIONS} for l in LANGUAGES}
        all_results["conditions"] = cached_meta["conditions"]
        all_results["diagnostics"] = json.loads(str(loaded["diagnostics"]))
        all_results["extraction_environment"] = cached_meta.get("extraction_environment")
    else:
        if not (args.device == "cuda" and torch.cuda.is_bf16_supported()):
            raise RuntimeError("bf16 CUDA is required for this diagnostic; refusing fp32/fp16 fallback")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "right"
        model_kwargs = {"torch_dtype": torch.bfloat16}
        model = AutoModel.from_pretrained(args.model, **model_kwargs).to(args.device).eval()
        extraction_environment = {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "dtype": "bfloat16",
            "model_revision": getattr(model.config, "_commit_hash", "unknown"),
            "model": args.model,
        }
        all_results["extraction_environment"] = extraction_environment
        for language in LANGUAGES:
            features, diagnostics = extract_conditions(model, tokenizer, [x["statement"] for x in data[language]], args.layer,
                                                        args.device, tokenizer.model_max_length, args.batch_size)
            feature_by_language[language] = features
            all_results["conditions"][language] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                                    for k, v in features.items()}
            all_results.setdefault("diagnostics", {})[language] = diagnostics
        if args.feature_cache is not None:
            args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
            flat = {f"{l}:{c}": v for l in LANGUAGES for c, v in feature_by_language[l].items()}
            meta_json = json.dumps({"model": args.model, "layer": args.layer, "data_hash": data_hash,
                                    "conditions_version": 2, "conditions": all_results["conditions"],
                                    "extraction_environment": extraction_environment})
            diag_json = json.dumps(all_results["diagnostics"], ensure_ascii=False)
            flat["metadata"] = np.array(meta_json)
            flat["diagnostics"] = np.array(diag_json)
            np.savez_compressed(args.feature_cache, **flat)
    labels = {l: np.asarray([x["label"] for x in data[l]]) for l in LANGUAGES}
    ids_by_language = {l: np.asarray([x["question_id"] for x in data[l]]) for l in LANGUAGES}
    for source in LANGUAGES:
        if "legacy" in lanes:
            records, fits = probe_records(feature_by_language, labels, ids_by_language, "legacy", source,
                                          conditions=("current_raw",))
            all_results["predictions"].extend(records)
            all_results["fits"].extend(fits)
        if "fair" in lanes:
            records, fits = probe_records(feature_by_language, labels, ids_by_language, "fair", source,
                                          conditions=fair_conditions)
            all_results["predictions"].extend(records)
            all_results["fits"].extend(fits)
            records, fits = probe_records(feature_by_language, labels, ids_by_language, "fair", source,
                                          norm_only=True, conditions=fair_conditions)
            all_results["predictions"].extend(records)
            all_results["fits"].extend(fits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(all_results, ensure_ascii=False)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    if str(args.output).endswith(".gz"):
        with gzip.open(tmp, "wt", compresslevel=6) as fh:
            fh.write(payload)
    else:
        tmp.write_text(payload, encoding="utf-8")
    tmp.replace(args.output)


if __name__ == "__main__":
    main()
