"""Layerwise in-language and zero-shot cross-lingual political probes."""

from __future__ import annotations

import argparse
from collections import Counter
import warnings
from datetime import datetime, timezone
import gzip
import json
import re
import time
import uuid
from pathlib import Path

LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")


def choose_device(name: str) -> torch.device:
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(path: Path, languages: list[str]) -> dict[str, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for record in json.loads(path.read_text(encoding="utf-8")):
        if record.get("polarity") in {-1, 1}:
            grouped.setdefault(record["id"], []).append(record)
    paired = []
    for question_id, records in grouped.items():
        if len(records) != 2 or {r["polarity"] for r in records} != {-1, 1}:
            raise ValueError(f"Question {question_id} is not a strict +/-1 pair")
        if any(not record.get(language) for record in records for language in languages):
            raise ValueError(f"Question {question_id} is missing a language statement")
        paired.append((question_id, records))
    if len(paired) < 5:
        raise ValueError("At least five paired questions are required for 5-fold CV")
    return {
        language: [
            {"question_id": question_id, "statement": record[language],
             "polarity": record["polarity"]}
            for question_id, records in paired for record in sorted(records, key=lambda r: r["polarity"])
        ]
        for language in languages
    }


def load_rile_data(path: Path, languages: list[str]):
    """Load separate RILE development and held-out-category items."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    seen = set()
    data = {language: [] for language in languages}
    held_out = {language: [] for language in languages}
    held_out_count = 0
    for row in rows:
        item_id = row["item_id"]
        if item_id in seen:
            raise ValueError(f"Duplicate RILE item_id: {item_id}")
        seen.add(item_id)
        if (row.get("label") not in {0, 1} or not row.get("manifesto_id")
                or not isinstance(row.get("held_out"), bool)):
            raise ValueError(f"Invalid RILE label or manifesto_id for {item_id}")
        if any(not row.get(language) for language in languages):
            raise ValueError(f"RILE item {item_id} is missing a language statement")
        if row.get("held_out"):
            held_out_count += 1
            for language in languages:
                held_out[language].append({
                    "item_id": item_id, "manifesto_id": row["manifesto_id"],
                    "statement": row[language], "label": row["label"],
                    "source_lang": row.get("source_lang"), "code": row.get("code"),
                    "cmp_code": row.get("cmp_code"),
                })
            continue
        for language in languages:
            data[language].append({
                "item_id": item_id, "manifesto_id": row["manifesto_id"],
                "statement": row[language], "label": row["label"],
                "source_lang": row.get("source_lang"), "code": row.get("code"),
                "cmp_code": row.get("cmp_code"),
            })
    if not data[languages[0]]:
        raise ValueError("No RILE development items found")
    return data, held_out, held_out_count


def canonical_folds(question_ids: list[int]) -> list[tuple[set[int], set[int]]]:
    from sklearn.model_selection import GroupKFold

    unique_ids = list(dict.fromkeys(question_ids))
    if len(unique_ids) < 5:
        raise ValueError("At least five unique question IDs are required")
    return [
        ({unique_ids[i] for i in train}, {unique_ids[i] for i in test})
        for train, test in GroupKFold(n_splits=5).split(unique_ids, groups=unique_ids)
    ]


def rile_folds(manifesto_ids: list[str]) -> list[tuple[set[str], set[str]]]:
    counts = Counter(manifesto_ids)
    groups = sorted(counts, key=lambda group: (-counts[group], group))
    if len(groups) < 5:
        raise ValueError("At least five RILE manifesto IDs are required for 5-fold CV")
    test_groups = [set() for _ in range(5)]
    fold_sizes = [0] * 5
    for group in groups:
        fold = min(range(5), key=fold_sizes.__getitem__)
        test_groups[fold].add(group)
        fold_sizes[fold] += counts[group]
    all_groups = set(groups)
    return [(all_groups - test, test) for test in test_groups]


def extract(model, tokenizer, items: list[dict], device: torch.device) -> list[list]:
    import torch

    vectors = []
    with torch.inference_mode():
        for item in items:
            inputs = tokenizer(item["statement"], return_tensors="pt", truncation=True).to(device)
            hidden = model(**inputs, output_hidden_states=True).hidden_states
            vectors.append([state[0, -1].float().cpu().numpy() for state in hidden])
    return [[vectors[row][layer] for row in range(len(items))]
            for layer in range(len(vectors[0]))]


def indices(items: list[dict], group_ids: set, group_field: str = "question_id") -> list[int]:
    return [i for i, item in enumerate(items) if item[group_field] in group_ids]


def model_dimensions(config) -> tuple[int, int, int]:
    """Return hidden-state count, width, and transformer block count."""
    configs = [getattr(config, "text_config", None), config]
    block_count = next((getattr(item, "num_hidden_layers", None) for item in configs
                        if item is not None and getattr(item, "num_hidden_layers", None) is not None), None)
    hidden_size = next((getattr(item, "hidden_size", None) or getattr(item, "d_model", None)
                        for item in configs if item is not None
                        and (getattr(item, "hidden_size", None) or getattr(item, "d_model", None))), None)
    if block_count is None or hidden_size is None:
        raise ValueError("Model config must define num_hidden_layers and hidden_size or d_model")
    return int(block_count) + 1, int(hidden_size), int(block_count)


def rile_result_paths(output_dir: Path, model_slug: str, split: str,
                      dataset: str = "rile_v1", protocol: str | None = None) -> tuple[Path, Path]:
    suffix = "" if split == "development" else "_heldout"
    protocol_part = f"_{protocol}" if protocol else ""
    stem = output_dir / f"{dataset}{protocol_part}_{model_slug}{suffix}"
    return stem.with_name(stem.name + "_summary.json"), stem.with_name(stem.name + "_predictions.jsonl.gz")


def progress_logger(path: Path, run_id: str, model: str, context: dict | None = None):
    run_started = time.monotonic()
    def log(event: str, **fields) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                  "model": model, "event": event,
                  "elapsed_seconds": round(time.monotonic() - run_started, 2), **fields,
                  **(context or {})}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return log


def extract_rile_memmap(model, tokenizer, items: list[dict], device, path: Path,
                        batch_size: int, progress=None):
    """Write final-token vectors for every model layer to a float16 memmap."""
    import numpy as np
    import torch

    layer_count, hidden_size, _ = model_dimensions(model.config)
    vectors = np.memmap(path, mode="w+", dtype=np.float16,
                        shape=(len(items), layer_count, hidden_size))
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Batched RILE extraction requires a tokenizer pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            stop = min(start + batch_size, len(items))
            batch = tokenizer([item["statement"] for item in items[start:stop]],
                              return_tensors="pt", padding=True, truncation=True).to(device)
            hidden_states = model(**batch, output_hidden_states=True).hidden_states
            last = batch["attention_mask"].sum(dim=1) - 1
            row = torch.arange(last.size(0), device=device)
            vectors[start:stop] = np.stack([
                state[row, last].float().cpu().numpy() for state in hidden_states
            ], axis=1).astype(np.float16)
            if progress and (stop == len(items) or stop % (batch_size * 100) == 0):
                progress(stop, len(items))
    vectors.flush()
    return vectors


def nested_group_folds(items: list[dict], count: int):
    """Balance whole manifestos across folds, within the supplied partition."""
    groups = Counter(item["manifesto_id"] for item in items)
    if len(groups) < count:
        raise ValueError(f"Need {count} manifesto groups; found {len(groups)}")
    bins = [set() for _ in range(count)]
    sizes = [0] * count
    for group in sorted(groups, key=lambda key: (-groups[key], key)):
        slot = min(range(count), key=sizes.__getitem__)
        bins[slot].add(group)
        sizes[slot] += groups[group]
    return [(indices(items, set(groups) - group, "manifesto_id"),
             indices(items, group, "manifesto_id")) for group in bins]


def budgeted_sample(items: list[dict], limit: int, seed: int):
    """Proportionally sample outer-training strata, preserving their original row order."""
    import numpy as np

    strata = {}
    for index, item in enumerate(items):
        key = (item["manifesto_id"], item["label"], str(item.get("cmp_code") or item.get("code")))
        strata.setdefault(key, []).append(index)
    count = min(limit, len(items))
    rng = np.random.default_rng(seed)
    quotas = {key: len(rows) * count / len(items) for key, rows in strata.items()}
    allocation = {key: int(quotas[key]) for key in strata}
    remainder = count - sum(allocation.values())
    for key in sorted(strata, key=lambda key: (-(quotas[key] - allocation[key]), key))[:remainder]:
        allocation[key] += 1
    selected = [index for key, rows in strata.items()
                for index in rng.choice(rows, allocation[key], replace=False)]
    return sorted(selected)


def budgeted_folds(items: list[dict], count: int, first_only: bool):
    folds = nested_group_folds(items, count)
    folds = folds[:1] if first_only else folds
    if any({items[i]["label"] for i in train} != {0, 1} or
           {items[i]["label"] for i in valid} != {0, 1} for train, valid in folds):
        raise ValueError("Budgeted grouped screening requires both labels in each train and validation fold")
    return folds


def budgeted_screen(items: list[dict], limit: int, fold_count: int, first_only: bool, seed: int):
    for attempt in range(100):
        sample = budgeted_sample(items, limit, seed + attempt)
        subset = [items[index] for index in sample]
        try:
            folds = budgeted_folds(subset, fold_count, first_only)
            return sample, folds, seed + attempt
        except ValueError:
            continue
    raise ValueError("Could not form label-complete budgeted grouped folds in 100 deterministic samples")


def probe_fit(x, y, max_iter: int, stats: list[dict], context: dict, progress=None):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        probe = LogisticRegression(C=1.0, max_iter=max_iter).fit(
            np.asarray(x, dtype=np.float32), y)
    record = {**context, "seconds": round(time.monotonic() - started, 3),
              "n_iter": int(probe.n_iter_[0]),
              "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught)}
    stats.append(record)
    if progress:
        progress("fit_complete", **record)
    return probe


def _worker_screen_layer(layer, features, labels, folds, max_iter):
    """Score one layer across fixed grouped folds in an isolated worker process."""
    import os
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from threadpoolctl import threadpool_limits

    fit_rows, truth, guesses = [], [], []
    with threadpool_limits(limits=1):
        for fold_index, (train, valid) in enumerate(folds):
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = LogisticRegression(C=1.0, max_iter=max_iter).fit(
                    np.asarray(features[train, layer, :], dtype=np.float32), labels[train])
            fit_rows.append({"fold": fold_index, "seconds": round(time.monotonic() - started, 3),
                             "n_iter": int(model.n_iter_[0]),
                             "convergence_warning": any(issubclass(w.category, ConvergenceWarning)
                                                         for w in caught)})
            guesses.extend(model.predict(np.asarray(features[valid, layer, :], dtype=np.float32)).tolist())
            truth.extend(labels[valid].tolist())
    return {"layer_index": layer, "balanced_accuracy": float(balanced_accuracy_score(truth, guesses)),
            "pid": os.getpid(), "fits": fit_rows}


def _process_tree_rss_bytes(root_pid: int) -> int:
    """Sum resident memory for a Linux process and its current descendants."""
    import os

    if not Path("/proc").is_dir():
        raise RuntimeError("The worker pilot's process-tree RSS measurement requires Linux /proc")
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = 0
    pending = [root_pid]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        proc = Path("/proc") / str(pid)
        try:
            statm = (proc / "statm").read_text().split()
            total += int(statm[1]) * page_size
            children = (proc / "task" / str(pid) / "children").read_text().split()
            pending.extend(int(child) for child in children)
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
    return total


def run_worker_pilot(args, features, items, outer_folds, output_path, progress):
    """Benchmark deterministic budgeted layer screens using outer-fold training data only."""
    import os
    import threading
    import numpy as np
    from joblib import Parallel, delayed

    if args.pilot_workers_max_iter < 1:
        raise ValueError("--pilot-workers-max-iter must be positive")
    train_outer, _ = outer_folds[0]
    train_items = [items[i] for i in train_outer]
    labels = np.asarray([item["label"] for item in items], dtype=np.int8)
    stage_inputs = {}
    for name, limit, fold_count, first_only in (("screen_1250", 1250, 5, True),
                                                 ("screen_2000", 2000, 2, False)):
        sampled, folds, used_seed = budgeted_screen(
            train_items, limit, fold_count, first_only, 20260928)
        sampled_outer = np.asarray(train_outer)[sampled]
        fixed_folds = [(sampled_outer[train], sampled_outer[valid]) for train, valid in folds]
        stage_inputs[name] = {"sampled": sampled, "folds": fixed_folds, "seed": used_seed}

    results = {"status": "partial", "dataset": "rile_v2", "protocol": "budgeted_source_peak",
               "pilot": "worker_timing", "model": args.model,
               "input_path": str(args.data_path.resolve()), "language": "en", "outer_fold": 0,
               "outer_training_items": len(train_outer), "max_iter": args.pilot_workers_max_iter,
               "C": 1.0,
               "feature_preprocessing": "raw float32", "blas_threads_per_fit": 1,
               "workers": [1, 2, 4, 6], "stages": {}, "no_outer_test_metrics": True}
    for worker_count in results["workers"]:
        candidates = list(range(features.shape[1]))
        worker_result = {"n_jobs": worker_count, "stages": {}}
        for stage_name in ("screen_1250", "screen_2000"):
            stage = stage_inputs[stage_name]
            if stage_name == "screen_2000":
                candidates = worker_result["stages"]["screen_1250"]["top12"]
            started = time.monotonic()
            progress("worker_stage_start", stage=stage_name, n_jobs=worker_count,
                     candidate_layers=len(candidates), sample_items=len(stage["sampled"]))
            memory_done = threading.Event()
            memory_peak = [_process_tree_rss_bytes(os.getpid())]

            def sample_memory():
                while not memory_done.wait(0.1):
                    memory_peak[0] = max(memory_peak[0], _process_tree_rss_bytes(os.getpid()))

            memory_thread = threading.Thread(target=sample_memory, daemon=True)
            memory_thread.start()
            try:
                scored = Parallel(n_jobs=worker_count, backend="loky")(
                    delayed(_worker_screen_layer)(layer, features, labels, stage["folds"],
                                                  args.pilot_workers_max_iter)
                    for layer in candidates)
            finally:
                memory_peak[0] = max(memory_peak[0], _process_tree_rss_bytes(os.getpid()))
                memory_done.set()
                memory_thread.join()
            scored.sort(key=lambda row: (-row["balanced_accuracy"], row["layer_index"]))
            stage_record = {"sample_items": len(stage["sampled"]),
                            "sample_manifestos": len({train_items[i]["manifesto_id"]
                                                       for i in stage["sampled"]}),
                            "sampling_seed": stage["seed"],
                            "grouped_validation_folds": len(stage["folds"]),
                            "wall_seconds": round(time.monotonic() - started, 3),
                            "peak_process_tree_rss_bytes": memory_peak[0],
                            "peak_rss_measurement": "sampled sum of parent and descendant RSS from /proc every 0.1 seconds",
                            "peak_rss_sampling_interval_seconds": 0.1,
                            "layer_scores": [{"layer_index": row["layer_index"],
                                              "balanced_accuracy": row["balanced_accuracy"]}
                                             for row in scored],
                            "top12": [row["layer_index"] for row in scored[:12]],
                            "top4": [row["layer_index"] for row in scored[:4]],
                            "fits": [{"layer_index": row["layer_index"], "pid": row["pid"],
                                      **fit} for row in scored for fit in row["fits"]],
                            "fit_seconds": round(sum(fit["seconds"] for row in scored
                                                     for fit in row["fits"]), 3),
                            "fit_count": sum(len(row["fits"]) for row in scored),
                            "convergence_warnings": sum(fit["convergence_warning"]
                                                         for row in scored for fit in row["fits"])}
            worker_result["stages"][stage_name] = stage_record
            results["stages"][str(worker_count)] = worker_result
            baseline = (results["stages"].get("1", {}).get("stages", {}).get(stage_name)
                        if worker_count != 1 else stage_record)
            if baseline:
                baseline_scores = {row["layer_index"]: row["balanced_accuracy"]
                                   for row in baseline["layer_scores"]}
                current_scores = {row["layer_index"]: row["balanced_accuracy"]
                                  for row in stage_record["layer_scores"]}
                baseline_top12, baseline_top4 = baseline["top12"], baseline["top4"]
                current_top12, current_top4 = stage_record["top12"], stage_record["top4"]
                worker_result.setdefault("comparison_to_serial", {})[stage_name] = {
                    "exact_layer_scores_match": current_scores == baseline_scores,
                    "max_absolute_score_delta": max(
                        (abs(current_scores[layer] - baseline_scores[layer])
                         for layer in current_scores.keys() & baseline_scores.keys()), default=0.0),
                    "top12_exact_match": current_top12 == baseline_top12,
                    "top12_overlap": len(set(current_top12) & set(baseline_top12)),
                    "top4_exact_match": current_top4 == baseline_top4,
                    "top4_overlap": len(set(current_top4) & set(baseline_top4)),
                }
            output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            progress("worker_stage_complete", stage=stage_name, n_jobs=worker_count,
                     output=str(output_path), wall_seconds=stage_record["wall_seconds"],
                     fit_count=stage_record["fit_count"],
                     convergence_warnings=stage_record["convergence_warnings"])
        results["status"] = "partial" if worker_count != results["workers"][-1] else "complete"
        baseline_worker = results["stages"].get("1", {})
        if baseline_worker:
            baseline12 = baseline_worker["stages"]["screen_1250"]["top12"]
            baseline4 = baseline_worker["stages"]["screen_2000"]["top4"]
            selected12 = worker_result["stages"]["screen_1250"]["top12"]
            selected4 = worker_result["stages"]["screen_2000"]["top4"]
            worker_result["selection_comparison_to_serial"] = {
                "screen_1250_top12_exact_match": selected12 == baseline12,
                "screen_1250_top12_overlap": len(set(selected12) & set(baseline12)),
                "screen_2000_top4_exact_match": selected4 == baseline4,
                "screen_2000_top4_overlap": len(set(selected4) & set(baseline4)),
            }
        output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    progress("worker_pilot_complete", stage="worker_pilot", output=str(output_path))


def run_rile_nested(args, development, outer_folds, languages, progress):
    """Nested manifesto-grouped layer selection and aligned-language transfer."""
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from tempfile import TemporaryDirectory
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.batch_size < 1 or args.max_iter < 1:
        raise ValueError("--batch-size and --max-iter must be positive")
    if args.fit_workers < 1:
        raise ValueError("--fit-workers must be positive")
    if args.fit_workers > 1 and (args.dataset != "rile_v2"
                                 or args.protocol != "budgeted_source_peak"
                                 or args.pilot_convergence or args.pilot_workers):
        raise ValueError("--fit-workers above 1 requires a full RILE-v2 budgeted_source_peak run")
    if args.pilot_convergence:
        progress("pilot_config", stage="pilot", language=args.pilot_language,
                 layer_indices=args.pilot_layers, outer_fold=0, inner_folds=4,
                 max_iter_values=[1000, 3000], C=1.0, feature_preprocessing="raw float32")
    if args.pilot_workers:
        progress("pilot_config", stage="worker_pilot", language="en", outer_fold=0,
                 worker_counts=[1, 2, 4, 6], max_iter=args.pilot_workers_max_iter,
                 C=1.0, feature_preprocessing="raw float32")
    device = choose_device(args.device)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type in {"cuda", "mps"} else torch.float32)
    options = {"fix_mistral_regex": True} if args.model == "mistralai/Ministral-8B-Instruct-2410" else {}
    started = time.monotonic()
    progress("model_load_start", stage="model_load")
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    layer_count, width, blocks = model_dimensions(model.config)
    load_seconds = time.monotonic() - started
    progress("model_loaded", stage="model_load", seconds=round(load_seconds, 2),
             layers=layer_count, hidden_size=width)
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    result_dir = Path(args.output_dir) / "results"
    protocol = args.protocol
    budgeted = protocol == "budgeted_source_peak"
    summary_path, prediction_path = rile_result_paths(result_dir, model_slug, "development",
                                                       "rile_v2", protocol)
    pilot_path = result_dir / f"rile_v2_nested_source_peak_{model_slug}_pilot.json"
    partial_summary_path = summary_path.with_name(summary_path.stem + "_partial.json")
    partial_prediction_path = prediction_path.with_name(prediction_path.stem + "_partial.jsonl.gz")
    labels = np.asarray([item["label"] for item in development["en"]])
    fold_rows = nested_group_folds(development["en"], 5)
    fit_stats = []
    extraction_seconds = {}
    predictions = []
    selections = []
    screening = []
    rq2_scores = []
    with TemporaryDirectory(prefix="rile_v2_nested_features_") as scratch:
        vectors = {}
        extract_languages = (["en"] if args.pilot_workers else
                             [args.pilot_language] if args.pilot_convergence else languages)
        for language in extract_languages:
            began = time.monotonic()
            progress("extraction_start", stage="extraction", language=language,
                     items=len(development[language]))
            vectors[language] = extract_rile_memmap(
                model, tokenizer, development[language], device,
                Path(scratch) / f"{language}.f16", args.batch_size,
                lambda done, total: progress("extraction_progress", stage="extraction",
                    language=language, completed_items=done, total_items=total,
                    items_per_second=round(done / max(time.monotonic() - began, 0.001), 2)))
            extraction_seconds[language] = round(time.monotonic() - began, 2)
            progress("extraction_complete", stage="extraction", language=language,
                     seconds=extraction_seconds[language])
        if args.pilot_workers:
            pilot_path = result_dir / f"rile_v2_budgeted_source_peak_{model_slug}_worker_pilot.json"
            run_worker_pilot(args, vectors["en"], development["en"], fold_rows,
                             pilot_path, progress)
            return
        if args.pilot_convergence:
            layers = [int(x) for x in args.pilot_layers.split(",")]
            if not layers or any(layer < 0 or layer >= layer_count for layer in layers):
                raise ValueError(f"Pilot layers must be indices from 0 to {layer_count - 1}")
            train_outer, _ = fold_rows[0]
            train_items = [development[args.pilot_language][i] for i in train_outer]
            inner = nested_group_folds(train_items, 4)
            comparisons = []
            for layer in layers:
                prior = {}
                for effort in (1000, 3000):
                    predicted = np.empty(len(train_outer), dtype=np.int8)
                    for inner_fold, (train, valid) in enumerate(inner):
                        train_idx = np.asarray(train_outer)[train]
                        valid_idx = np.asarray(train_outer)[valid]
                        probe = probe_fit(vectors[args.pilot_language][train_idx, layer, :],
                                          labels[train_idx], effort, fit_stats,
                                          {"stage": "pilot_inner", "layer_index": layer,
                                           "inner_fold": inner_fold, "max_iter": effort}, progress)
                        predicted[valid] = probe.predict(np.asarray(
                            vectors[args.pilot_language][valid_idx, layer, :], dtype=np.float32))
                    prior[effort] = predicted
                    comparisons.append({"layer_index": layer, "max_iter": effort,
                                        "inner_balanced_accuracy": float(balanced_accuracy_score(
                                            labels[train_outer], predicted)),
                                        "inner_predictions": predicted.tolist()})
                comparisons[-1]["disagreements_vs_1000"] = int(np.count_nonzero(
                    prior[1000] != prior[3000]))
                pilot = {"status": "partial" if layer != layers[-1] else "complete",
                         "dataset": "rile_v2", "protocol": "nested_source_peak_pilot_convergence",
                         "model": args.model, "input_path": str(args.data_path.resolve()),
                         "outer_fold": 0, "language": args.pilot_language,
                         "outer_training_items": len(train_outer), "inner_folds": 4,
                         "layer_indices": layers, "load_seconds": round(load_seconds, 2),
                         "extraction_seconds": extraction_seconds,
                         "feature_bytes": {args.pilot_language: Path(scratch, f"{args.pilot_language}.f16").stat().st_size},
                         "comparisons": [{key: value for key, value in row.items()
                                          if key != "inner_predictions"} for row in comparisons],
                         "fits": fit_stats}
                pilot_path.write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
                progress("pilot_layer_complete", stage="pilot", layer_index=layer,
                         output=str(pilot_path), fit_count=len(fit_stats),
                         disagreement_count=comparisons[-1]["disagreements_vs_1000"])
            progress("pilot_complete", stage="pilot", output=str(pilot_path),
                     fit_count=len(fit_stats), fit_seconds=round(sum(x["seconds"] for x in fit_stats), 2))
            return
        for outer_fold, (train_outer, test_outer) in enumerate(fold_rows):
            train_items = [development["en"][i] for i in train_outer]
            if budgeted:
                stages = []
                for name, limit, folds, first_only, keep in (
                    ("screen_1250", 1250, 5, True, 12),
                    ("screen_2000", 2000, 2, False, 4),
                    ("full_outer_train", len(train_items), 4, False, 1)):
                    if name == "full_outer_train":
                        sampled = list(range(len(train_items)))
                        inner = budgeted_folds(train_items, folds, False)
                        used_seed = None
                    else:
                        sampled, inner, used_seed = budgeted_screen(
                            train_items, limit, folds, first_only, 20260928 + outer_fold * 1000)
                    stages.append((name, sampled, inner, used_seed, keep))
            else:
                inner = nested_group_folds(train_items, 4)
            chosen = {}
            source_probes = {}
            for language in languages:
                candidates = list(range(layer_count))
                stage_plan = stages if budgeted else [("inner", list(range(len(train_items))), inner, None, 1)]
                for stage_name, sampled, stage_folds, used_seed, keep in stage_plan:
                    scores = []
                    sampled_outer = np.asarray(train_outer)[sampled]
                    if args.fit_workers > 1:
                        from joblib import Parallel, delayed

                        fixed_folds = [(sampled_outer[train], sampled_outer[valid])
                                       for train, valid in stage_folds]
                        scored = Parallel(n_jobs=args.fit_workers, backend="loky")(
                            delayed(_worker_screen_layer)(layer, vectors[language], labels,
                                                          fixed_folds, args.max_iter)
                            for layer in candidates)
                        for row in scored:
                            layer = row["layer_index"]
                            scores.append((layer, row["balanced_accuracy"]))
                            for inner_fold, fit in enumerate(row["fits"]):
                                record = {"stage": stage_name if budgeted else "inner",
                                          "outer_fold": outer_fold, "inner_fold": inner_fold,
                                          "language": language, "layer_index": layer,
                                          "seconds": fit["seconds"], "n_iter": fit["n_iter"],
                                          "convergence_warning": fit["convergence_warning"]}
                                fit_stats.append(record)
                                progress("fit_complete", **record)
                    else:
                        for layer in candidates:
                            truth, guesses = [], []
                            for inner_fold, (train, valid) in enumerate(stage_folds):
                                train_idx = sampled_outer[train]
                                valid_idx = sampled_outer[valid]
                                probe = probe_fit(vectors[language][train_idx, layer, :], labels[train_idx],
                                                  args.max_iter, fit_stats,
                                                  {"stage": stage_name if budgeted else "inner",
                                                   "outer_fold": outer_fold, "inner_fold": inner_fold,
                                                   "language": language, "layer_index": layer}, progress)
                                guesses.extend(probe.predict(np.asarray(
                                    vectors[language][valid_idx, layer, :], dtype=np.float32)).tolist())
                                truth.extend(labels[valid_idx].tolist())
                            score = float(balanced_accuracy_score(truth, guesses))
                            scores.append((layer, score))
                            if not budgeted or stage_name == "full_outer_train":
                                selections.append({"outer_fold": outer_fold, "language": language,
                                                   "layer_index": layer,
                                                   "inner_balanced_accuracy": score})
                            if not budgeted:
                                progress("inner_layer_complete", stage="inner_selection",
                                         outer_fold=outer_fold, language=language, layer_index=layer,
                                         inner_balanced_accuracy=score, fit_count=len(fit_stats),
                                         fit_seconds=round(sum(x["seconds"] for x in fit_stats), 2),
                                         convergence_warnings=sum(x["convergence_warning"]
                                                                  for x in fit_stats))
                    if args.fit_workers > 1:
                        if not budgeted or stage_name == "full_outer_train":
                            selections.extend({"outer_fold": outer_fold, "language": language,
                                               "layer_index": layer,
                                               "inner_balanced_accuracy": score}
                                              for layer, score in scores)
                        if not budgeted:
                            for layer, score in scores:
                                progress("inner_layer_complete", stage="inner_selection",
                                         outer_fold=outer_fold, language=language, layer_index=layer,
                                         inner_balanced_accuracy=score, fit_count=len(fit_stats),
                                         fit_seconds=round(sum(x["seconds"] for x in fit_stats), 2),
                                         convergence_warnings=sum(x["convergence_warning"]
                                                                  for x in fit_stats))
                    ranked = sorted(scores, key=lambda pair: (-pair[1], pair[0]))
                    candidates = [layer for layer, _ in ranked[:keep]]
                    if budgeted:
                        record = {"outer_fold": outer_fold, "language": language,
                                  "stage": stage_name, "sample_items": len(sampled),
                                  "sample_manifestos": len({train_items[i]["manifesto_id"] for i in sampled}),
                                  "sampling_seed": used_seed, "grouped_validation_folds": len(stage_folds),
                                  "fit_workers": args.fit_workers,
                                  "layer_scores": [{"layer_index": layer, "balanced_accuracy": score}
                                                   for layer, score in scores],
                                  "selected_layers": candidates}
                        screening.append(record)
                        progress("budgeted_stage_complete", **record, fit_count=len(fit_stats))
                chosen[language] = ranked[0]
                layer = chosen[language][0]
                source_probes[language] = probe_fit(
                    vectors[language][train_outer, layer, :], labels[train_outer],
                    args.max_iter, fit_stats,
                    {"stage": "selected_refit", "outer_fold": outer_fold,
                     "language": language, "layer_index": layer}, progress)
                selected_guess = source_probes[language].predict(np.asarray(
                    vectors[language][test_outer, layer, :], dtype=np.float32))
                for row, value in zip(test_outer, selected_guess):
                    item = development[language][row]
                    predictions.append({"dataset": "rile_v2", "protocol": protocol,
                                        "split": "development_outer_test", "model": args.model,
                                        "fold": outer_fold, "kind": "selected_rq1",
                                        "source_language": language, "target_language": language,
                                        "layer_index": layer, "item_id": item["item_id"],
                                        "manifesto_id": item["manifesto_id"],
                                        "code": item.get("code"), "cmp_code": item.get("cmp_code"),
                                        "source_lang": item.get("source_lang"),
                                        "true_label": int(labels[row]), "predicted_label": int(value)})
                progress("language_selected", stage="inner_selection", outer_fold=outer_fold,
                         language=language, layer_index=layer,
                         inner_balanced_accuracy=chosen[language][1], fit_count=len(fit_stats))
            en_layer = chosen["en"][0]
            for target in languages:
                test_x = np.asarray(vectors[target][test_outer, en_layer, :], dtype=np.float32)
                transfer_guess = source_probes["en"].predict(test_x)
                for kind, guess in (("english_transfer", transfer_guess),):
                    rq2_scores.append({"outer_fold": outer_fold, "target_language": target,
                                       "kind": kind, "layer_index": en_layer,
                                       "balanced_accuracy": float(balanced_accuracy_score(
                                           labels[test_outer], guess))})
                    for row, value in zip(test_outer, guess):
                        item = development[target][row]
                        predictions.append({"dataset": "rile_v2", "protocol": protocol,
                                            "split": "development_outer_test", "model": args.model,
                                            "fold": outer_fold, "kind": kind,
                                            "source_language": "en", "target_language": target,
                                            "layer_index": en_layer, "item_id": item["item_id"],
                                            "manifesto_id": item["manifesto_id"],
                                            "code": item.get("code"), "cmp_code": item.get("cmp_code"),
                                            "source_lang": item.get("source_lang"),
                                            "true_label": int(labels[row]), "predicted_label": int(value)})
                if target != "en":
                    target_probe = probe_fit(vectors[target][train_outer, en_layer, :],
                                             labels[train_outer], args.max_iter, fit_stats,
                                             {"stage": "target_comparator", "outer_fold": outer_fold,
                                              "language": target, "layer_index": en_layer}, progress)
                    target_guess = target_probe.predict(test_x)
                    rq2_scores.append({"outer_fold": outer_fold, "target_language": target,
                                       "kind": "target_trained", "layer_index": en_layer,
                                       "balanced_accuracy": float(balanced_accuracy_score(
                                           labels[test_outer], target_guess))})
                    for row, value in zip(test_outer, target_guess):
                        item = development[target][row]
                        predictions.append({"dataset": "rile_v2", "protocol": protocol,
                                            "split": "development_outer_test", "model": args.model,
                                            "fold": outer_fold, "kind": "target_trained",
                                            "source_language": target, "target_language": target,
                                            "layer_index": en_layer, "item_id": item["item_id"],
                                            "manifesto_id": item["manifesto_id"],
                                            "code": item.get("code"), "cmp_code": item.get("cmp_code"),
                                            "source_lang": item.get("source_lang"),
                                            "true_label": int(labels[row]), "predicted_label": int(value)})
            with gzip.open(partial_prediction_path, "wt", encoding="utf-8") as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            partial_summary_path.write_text(json.dumps({
                "status": "partial", "completed_outer_folds": outer_fold + 1,
                "dataset": "rile_v2", "protocol": protocol, "model": args.model,
                "fit_workers": args.fit_workers,
                "inner_validation_layer_scores": selections,
                "rq2_fold_scores": rq2_scores, "fit_count": len(fit_stats),
                "fit_convergence_warnings": sum(x["convergence_warning"] for x in fit_stats),
                "partial_predictions": str(partial_prediction_path)}, indent=2) + "\n", encoding="utf-8")
            if budgeted:
                partial_summary = json.loads(partial_summary_path.read_text(encoding="utf-8"))
                partial_summary["budgeted_screening"] = screening
                partial_summary_path.write_text(json.dumps(partial_summary, indent=2) + "\n", encoding="utf-8")
            progress("outer_fold_complete", stage="outer", outer_fold=outer_fold,
                     fit_workers=args.fit_workers,
                     selected_layers={language: chosen[language][0] for language in languages},
                     fit_count=len(fit_stats), partial_summary=str(partial_summary_path),
                     partial_predictions=str(partial_prediction_path))
    def metrics(kind, target):
        rows = [p for p in predictions if p["kind"] == kind and p["target_language"] == target]
        truth = [p["true_label"] for p in rows]
        guess = [p["predicted_label"] for p in rows]
        return {"items": len(rows), "accuracy": float(accuracy_score(truth, guess)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, guess)),
                "f1": float(f1_score(truth, guess, pos_label=1)),
                "label_counts": {str(label): sum(value == label for value in truth) for label in (0, 1)}}
    transfer = {target: metrics("english_transfer", target) for target in languages if target != "en"}
    summary = {"dataset": "rile_v2", "protocol": protocol, "model": args.model,
               "fit_workers": args.fit_workers,
               "input_path": str(args.data_path.resolve()), "development_items": len(labels),
               "development_manifestos": len({item["manifesto_id"] for item in development["en"]}),
               "languages": languages, "outer_folds": 5, "inner_folds": 4,
               "feature": "final-token residual hidden states, float16 cache, float32 raw logistic input",
               "logistic_C": 1.0, "max_iter": args.max_iter, "batch_size": args.batch_size,
               "model_load_seconds": round(load_seconds, 2),
               "extraction_seconds": extraction_seconds, "fit_count": len(fit_stats),
               "fit_seconds": round(sum(x["seconds"] for x in fit_stats), 2),
               "fit_convergence_warnings": sum(x["convergence_warning"] for x in fit_stats),
               "fits": fit_stats, "inner_validation_layer_scores": selections,
               "rq1_inner_validation_layer_curves_descriptive": [
                   {"language": language, "layer_index": layer,
                    "mean_inner_balanced_accuracy": float(np.mean([
                        item["inner_balanced_accuracy"] for item in selections
                        if item["language"] == language and item["layer_index"] == layer])),
                    "outer_fold_scores": [
                        {"outer_fold": item["outer_fold"],
                         "inner_balanced_accuracy": item["inner_balanced_accuracy"]}
                        for item in selections
                        if item["language"] == language and item["layer_index"] == layer]}
                   for language in languages for layer in range(layer_count)
                   if not budgeted or any(item["language"] == language and item["layer_index"] == layer
                                          for item in selections)],
               "rq1_selected": {language: metrics("selected_rq1", language) for language in languages},
               "rq2_fold_scores": rq2_scores,
               "rq2_english_transfer": transfer,
               "rq2_primary_equal_target_mean_balanced_accuracy": float(np.mean(
                   [transfer[target]["balanced_accuracy"] for target in transfer])),
               "rq2_target_trained": {target: metrics("target_trained", target)
                                       for target in languages if target != "en"}}
    if budgeted:
        summary["budgeted_screening"] = screening
        summary["planned_fit_count"] = 5 * (len(languages) * (layer_count + 41) + len(languages) - 1)
        summary["screening_rule"] = {"seed": 20260928, "strata": "manifesto_id × label × CMP category",
                                     "allocation": "proportional largest remainder; earliest layer breaks score ties",
                                     "stages": ["1250 items, one of five grouped folds, keep 12",
                                                "2000 items, two grouped folds, keep 4",
                                                "all outer-training items, four grouped folds, keep 1"]}
    with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress("outputs_saved", stage="save", summary=str(summary_path),
             predictions=str(prediction_path), fit_count=len(fit_stats))


def replay_source_layers(original, model: str, languages: list[str], item_count: int,
                         batch_size: int) -> dict[tuple[int, str], int]:
    """Validate and index the thirty recorded training-only layer choices."""
    if (original.get("dataset") != "rile_v2" or original.get("protocol") != "budgeted_source_peak"
            or original.get("model") != model or original.get("languages") != languages
            or original.get("development_items") != item_count
            or original.get("outer_folds") != 5 or original.get("inner_folds") != 4
            or original.get("max_iter") != 3000 or original.get("logistic_C") != 1.0
            or original.get("batch_size") != batch_size):
        raise ValueError("Selection summary does not match this RILE-v2 replay")
    selected = {}
    for fit in original["fits"]:
        if fit["stage"] != "selected_refit":
            continue
        key = (fit["outer_fold"], fit["language"])
        if key in selected:
            raise ValueError(f"Duplicate selected refit: {key}")
        selected[key] = fit["layer_index"]
    expected = {(fold, language) for fold in range(5) for language in languages}
    if set(selected) != expected:
        raise ValueError("Selection summary lacks exactly one source layer per fold and language")
    return selected


def replay_snapshot_revision(snapshot: Path, requested: str, config_revision: str | None) -> str:
    """Require an exact cached Hub snapshot; config metadata may omit its hash."""
    if snapshot.name != requested or snapshot.parent.name != "snapshots":
        raise ValueError(f"Cached model snapshot does not match requested revision {requested}")
    if config_revision is not None and config_revision != requested:
        raise ValueError(f"Model config revision {config_revision} differs from requested {requested}")
    return snapshot.name


def run_rile_source_matrix(args, development, languages, progress):
    """Replay saved training-side layer choices for every source-to-target pair."""
    import numpy as np
    import torch
    from hashlib import sha256
    from huggingface_hub import try_to_load_from_cache
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from tempfile import TemporaryDirectory
    from transformers import AutoModelForCausalLM, AutoTokenizer

    original = json.loads(Path(args.selection_summary).read_text(encoding="utf-8"))
    selected = replay_source_layers(original, args.model, languages,
                                    len(development["en"]), args.batch_size)

    fold_rows = nested_group_folds(development["en"], 5)
    labels = np.asarray([item["label"] for item in development["en"]])
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    result_dir = Path(args.output_dir) / "results"
    summary_path, prediction_path = rile_result_paths(
        result_dir, model_slug, "development", "rile_v2", "source_matrix_replay")
    probe_path = result_dir / f"rile_v2_source_matrix_replay_{model_slug}_probes.npz"
    device = choose_device(args.device)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type in {"cuda", "mps"} else torch.float32)
    options = {"fix_mistral_regex": True} if args.model == "mistralai/Ministral-8B-Instruct-2410" else {}
    cached_config = try_to_load_from_cache(args.model, "config.json", revision=args.model_revision)
    if not isinstance(cached_config, str) or not Path(cached_config).is_file():
        raise ValueError("Pinned model config.json is absent from the local cache")
    snapshot = Path(cached_config).parent
    replay_snapshot_revision(snapshot, args.model_revision, None)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, **options)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=dtype).to(device).eval()
    config_revision = getattr(model.config, "_commit_hash", None)
    resolved_revision = replay_snapshot_revision(snapshot, args.model_revision, config_revision)
    layer_count, _, _ = model_dimensions(model.config)
    if any(not isinstance(layer, int) or not 0 <= layer < layer_count for layer in selected.values()):
        raise ValueError("Saved source layer is outside this model's hidden states")
    progress("replay_model_loaded", revision=resolved_revision, layers=layer_count)

    predictions, fits, repairs, coefficients = [], [], [], {}
    with TemporaryDirectory(prefix="rile_v2_source_matrix_features_") as scratch:
        vectors = {language: extract_rile_memmap(
            model, tokenizer, development[language], device,
            Path(scratch) / f"{language}.f16", args.batch_size, progress=None)
            for language in languages}
        for fold, (train, test) in enumerate(fold_rows):
            train_items = [development["en"][i] for i in train]
            for source in languages:
                layer = selected[fold, source]
                if args.model == "Qwen/Qwen3.5-9B" and fold == 0 and source == "mr":
                    warned_candidate = [fit for fit in original["fits"]
                                        if fit["stage"] == "full_outer_train"
                                        and fit["outer_fold"] == 0 and fit["inner_fold"] == 3
                                        and fit["language"] == "mr" and fit["layer_index"] == 27
                                        and fit["convergence_warning"]]
                    if len(warned_candidate) != 1:
                        raise ValueError("Expected Qwen Marathi inner-fit warning is absent")
                    stage = next(row for row in original["budgeted_screening"]
                                 if row["outer_fold"] == fold and row["language"] == source
                                 and row["stage"] == "full_outer_train")
                    scores = {row["layer_index"]: row["balanced_accuracy"]
                              for row in stage["layer_scores"]}
                    if 27 not in scores:
                        raise ValueError("Qwen Marathi warning candidate is absent")
                    truth, guesses = [], []
                    for inner_fold, (inner_train, inner_valid) in enumerate(nested_group_folds(train_items, 4)):
                        fit = probe_fit(vectors[source][np.asarray(train)[inner_train], 27, :],
                                        labels[np.asarray(train)[inner_train]], 6000, fits,
                                        {"stage": "qwen_mr_candidate_repair", "outer_fold": fold,
                                         "inner_fold": inner_fold, "language": source,
                                         "layer_index": 27}, progress)
                        guesses.extend(fit.predict(np.asarray(
                            vectors[source][np.asarray(train)[inner_valid], 27, :], dtype=np.float32)))
                        truth.extend(labels[np.asarray(train)[inner_valid]])
                    repaired_score = float(balanced_accuracy_score(truth, guesses))
                    scores[27] = repaired_score
                    layer = sorted(scores, key=lambda index: (-scores[index], index))[0]
                    repairs.append({"fold": fold, "language": source, "candidate_layer": 27,
                                    "old_score": next(row["balanced_accuracy"] for row in stage["layer_scores"]
                                                      if row["layer_index"] == 27),
                                    "new_score": repaired_score, "old_selected_layer": selected[fold, source],
                                    "new_selected_layer": layer})
                max_iter = (6000 if args.model == "Qwen/Qwen3.5-9B" and
                            ((fold == 2 and source == "hi") or (fold == 0 and source == "mr" and layer == 27))
                            else 3000)
                if args.model == "Qwen/Qwen3.5-9B" and fold == 2 and source == "hi":
                    prior_fit = [fit for fit in original["fits"]
                                 if fit["stage"] == "selected_refit" and fit["outer_fold"] == 2
                                 and fit["language"] == "hi" and fit["layer_index"] == 27
                                 and fit["convergence_warning"]]
                    if len(prior_fit) != 1 or layer != 27:
                        raise ValueError("Expected Qwen Hindi selected-refit warning is absent")
                    repairs.append({"fold": fold, "language": source, "layer_index": layer,
                                    "old_max_iter": 3000, "new_max_iter": max_iter})
                probe = probe_fit(vectors[source][train, layer, :], labels[train], max_iter,
                                  fits, {"stage": "source_refit", "outer_fold": fold,
                                         "language": source, "layer_index": layer}, progress)
                key = f"f{fold}_{source}"
                coefficients[f"{key}_coef"] = probe.coef_
                coefficients[f"{key}_intercept"] = probe.intercept_
                coefficients[f"{key}_classes"] = probe.classes_
                for target in languages:
                    x = np.asarray(vectors[target][test, layer, :], dtype=np.float32)
                    guesses = probe.predict(x)
                    margins = probe.decision_function(x)
                    for row, guess, margin in zip(test, guesses, margins):
                        item = development[target][row]
                        predictions.append({"dataset": "rile_v2", "protocol": "source_matrix_replay",
                                            "model": args.model, "fold": fold,
                                            "source_language": source, "target_language": target,
                                            "layer_index": layer, "item_id": item["item_id"],
                                            "manifesto_id": item["manifesto_id"],
                                            "code": item.get("code"), "cmp_code": item.get("cmp_code"),
                                            "source_lang": item.get("source_lang"),
                                            "true_label": int(labels[row]), "predicted_label": int(guess),
                                            "decision_score": float(margin)})
            progress("replay_outer_fold_complete", outer_fold=fold, fit_count=len(fits))

    matrix = {}
    for source in languages:
        matrix[source] = {}
        for target in languages:
            rows = [row for row in predictions if row["source_language"] == source
                    and row["target_language"] == target]
            truth = [row["true_label"] for row in rows]
            guess = [row["predicted_label"] for row in rows]
            if len(rows) != len(labels):
                raise ValueError(f"Missing out-of-fold predictions for {source}->{target}")
            matrix[source][target] = {
                "items": len(rows), "accuracy": float(accuracy_score(truth, guess)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, guess)),
                "f1": float(f1_score(truth, guess, pos_label=1))}
    parity = []
    for language in languages:
        prior = original["rq1_selected"][language]
        if any(matrix[language][language][metric] != prior[metric]
               for metric in ("items", "accuracy", "balanced_accuracy", "f1")):
            parity.append({"source": language, "target": language,
                           "original": {metric: prior[metric] for metric in matrix[language][language]},
                           "replay": matrix[language][language]})
    for target in languages:
        if target != "en":
            prior = original["rq2_english_transfer"][target]
            if any(matrix["en"][target][metric] != prior[metric]
                   for metric in ("items", "accuracy", "balanced_accuracy", "f1")):
                parity.append({"source": "en", "target": target,
                               "original": {metric: prior[metric] for metric in matrix["en"][target]},
                               "replay": matrix["en"][target]})
    warned = [fit for fit in fits if fit["convergence_warning"]]
    summary = {"status": "complete" if not parity and not warned else "needs_review",
               "dataset": "rile_v2", "protocol": "source_matrix_replay", "model": args.model,
               "model_revision": resolved_revision, "selection_summary": str(Path(args.selection_summary).resolve()),
               "selection_summary_sha256": sha256(Path(args.selection_summary).read_bytes()).hexdigest(),
               "input_sha256": sha256(args.data_path.read_bytes()).hexdigest(),
               "development_items": len(labels), "languages": languages, "outer_folds": 5,
               "feature": original["feature"], "logistic_C": 1.0,
               "selected_layers": [{"fold": fold, "source_language": source,
                                    "layer_index": next(fit["layer_index"] for fit in fits
                                                        if fit["stage"] == "source_refit"
                                                        and fit["outer_fold"] == fold and fit["language"] == source)}
                                   for fold in range(5) for source in languages],
               "qwen_repairs": repairs, "fits": fits, "convergence_warnings": warned,
               "parity_mismatches": parity, "matrix": matrix,
               "source_offdiagonal_mean": {source: float(np.mean([
                   matrix[source][target]["balanced_accuracy"] for target in languages if target != source]))
                   for source in languages},
               "target_offdiagonal_mean": {target: float(np.mean([
                   matrix[source][target]["balanced_accuracy"] for source in languages if source != target]))
                   for target in languages},
               "global_offdiagonal_mean": float(np.mean([
                   matrix[source][target]["balanced_accuracy"]
                   for source in languages for target in languages if source != target]))}
    with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(probe_path, **coefficients)
    summary["probe_parameters"] = str(probe_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress("replay_outputs_saved", status=summary["status"], summary=str(summary_path),
             predictions=str(prediction_path), probes=str(probe_path))
    if parity or warned:
        raise ValueError(f"Replay needs review: {len(parity)} parity differences, {len(warned)} warnings")


def run_rile_probe(args, development, held_out, folds, languages: list[str], progress) -> None:
    """Run development CV or the separate held-out-category evaluation."""
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tempfile import TemporaryDirectory

    fixed_blocks = {
        "allenai/Olmo-3-7B-Instruct": 17,
        "mistralai/Ministral-8B-Instruct-2410": 31,
        "google/gemma-2-9b-it": 15,
        "Qwen/Qwen3.5-9B": 14,
    }
    layer_provenance = "Prior RQ2 English in-language peak from the earlier survey-statement study."
    if args.model not in fixed_blocks:
        raise ValueError(f"No preregistered RQ2 block for {args.model}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    split = args.rile_split
    splits = ("development", "heldout") if split == "both" else (split,)
    if any([row["item_id"] for row in rows[language]] !=
           [row["item_id"] for row in rows["en"]]
           for rows in (development, held_out) for language in languages):
        raise ValueError("RILE development or held-out language records are not aligned")

    device = choose_device(args.device)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type in {"cuda", "mps"} else torch.float32)
    tokenizer_options = {"fix_mistral_regex": True} if args.model == "mistralai/Ministral-8B-Instruct-2410" else {}
    started = time.monotonic()
    progress("model_load_start", stage="model_load")
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_options)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    fixed_block = fixed_blocks[args.model]
    layer_count, hidden_size, block_count = model_dimensions(model.config)
    if fixed_block >= block_count:
        raise ValueError(f"Fixed RQ2 Block {fixed_block} is outside this model's depth")
    progress("model_loaded", stage="model_load", elapsed_seconds=round(time.monotonic() - started, 2),
             layers=layer_count, hidden_size=hidden_size, fixed_rq2_block=fixed_block)

    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    output_dir = Path(args.output_dir) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for mode in splits:
        evaluation = development if mode == "development" else held_out
        summaries[mode] = {"dataset": args.dataset, "split": mode, "model": args.model,
                           "training_items": len(development["en"]),
                           "evaluation_items": len(evaluation["en"]),
                           "fixed_rq2_block": fixed_block, "fixed_rq2_layer_provenance": layer_provenance,
                           "batch_size": args.batch_size,
                           "languages": languages, "rq1": [], "rq2": []}
        if args.dataset == "rile_v2":
            summaries[mode].update({"protocol": args.protocol,
                                    "input_path": str(args.data_path.resolve()),
                                    "protocol_status": "secondary historical-layer diagnostic; nested primary not implemented"})
        if mode == "heldout":
            summaries[mode]["heldout_manifestos_also_in_development"] = len(
                {item["manifesto_id"] for item in development["en"]} &
                {item["manifesto_id"] for item in held_out["en"]})
            summaries[mode]["rq1_layer_use"] = "Descriptive scores for every layer; do not select a layer from held-out results."

    temp_prefix = "rile_v1_features_" if args.dataset == "rile_v1" else "rile_v2_features_"
    with TemporaryDirectory(prefix=temp_prefix) as temp_name:
        temp_dir = Path(temp_name)
        dev_fixed, heldout_fixed = {}, {}
        for language in languages:
            language_started = time.monotonic()
            train_items = development[language]
            train_labels = np.asarray([item["label"] for item in train_items])
            dev_path = temp_dir / f"{language}_development.f16"
            extract_started = time.monotonic()
            progress("extraction_start", stage="extraction", language=language,
                     split="development", items=len(train_items))
            dev_vectors = extract_rile_memmap(
                model, tokenizer, train_items, device, dev_path, args.batch_size,
                lambda done, total: progress("extraction_progress", stage="extraction",
                    language=language, split="development", completed_items=done, total_items=total,
                    items_per_second=round(done / max(time.monotonic() - extract_started, 0.001), 2),
                    eta_seconds=round((total - done) * (time.monotonic() - extract_started) / done, 1)))
            dev_fixed[language] = np.array(dev_vectors[:, fixed_block + 1, :], copy=True)
            heldout_path = temp_dir / f"{language}_heldout.f16"
            heldout_vectors = None
            if "heldout" in splits:
                extract_started = time.monotonic()
                progress("extraction_start", stage="extraction", language=language,
                         split="heldout", items=len(held_out[language]))
                heldout_vectors = extract_rile_memmap(
                    model, tokenizer, held_out[language], device, heldout_path, args.batch_size,
                    lambda done, total: progress("extraction_progress", stage="extraction",
                        language=language, split="heldout", completed_items=done, total_items=total,
                        items_per_second=round(done / max(time.monotonic() - extract_started, 0.001), 2),
                        eta_seconds=round((total - done) * (time.monotonic() - extract_started) / done, 1)))
                heldout_fixed[language] = np.array(
                    heldout_vectors[:, fixed_block + 1, :], copy=True)

            for mode in splits:
                test_items = development[language] if mode == "development" else held_out[language]
                test_vectors = dev_vectors if mode == "development" else heldout_vectors
                test_labels = np.asarray([item["label"] for item in test_items])
                manifesto_rows = {}
                for i, item in enumerate(test_items):
                    manifesto_rows.setdefault(item["manifesto_id"], []).append(i)
                for layer_index in range(layer_count):
                    layer_started = time.monotonic()
                    if mode == "development":
                        predicted = np.empty(len(test_items), dtype=np.int8)
                        for train_groups, test_groups in folds:
                            train = indices(train_items, train_groups, "manifesto_id")
                            test = indices(test_items, test_groups, "manifesto_id")
                            probe = LogisticRegression(C=1.0, max_iter=1000).fit(
                                np.asarray(dev_vectors[train, layer_index, :], dtype=np.float32),
                                train_labels[train])
                            predicted[test] = probe.predict(
                                np.asarray(test_vectors[test, layer_index, :], dtype=np.float32))
                    else:
                        probe = LogisticRegression(C=1.0, max_iter=1000).fit(
                            np.asarray(dev_vectors[:, layer_index, :], dtype=np.float32), train_labels)
                        predicted = probe.predict(
                            np.asarray(test_vectors[:, layer_index, :], dtype=np.float32))
                    per_manifesto = {
                        group: [int(np.count_nonzero(predicted[rows] == test_labels[rows])), len(rows)]
                        for group, rows in manifesto_rows.items()
                    }
                    per_label = {
                        str(label): [int(np.count_nonzero((predicted == label) & (test_labels == label))),
                                     int(np.count_nonzero(test_labels == label))]
                        for label in (0, 1)
                    }
                    layer_name = "Embedding" if layer_index == 0 else f"Block {layer_index - 1}"
                    layer_result = {
                        "language": language, "layer": layer_name,
                        "accuracy": float(accuracy_score(test_labels, predicted)),
                        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predicted)),
                        "f1": float(f1_score(test_labels, predicted, pos_label=1)),
                        "manifesto_correct_total": per_manifesto, "label_correct_total": per_label,
                    }
                    summaries[mode]["rq1"].append(layer_result)
                    progress("rq1_layer_complete", stage="rq1", split=mode, language=language,
                             layer=layer_name, elapsed_seconds=round(time.monotonic() - layer_started, 2),
                             accuracy=layer_result["accuracy"],
                             balanced_accuracy=layer_result["balanced_accuracy"], f1=layer_result["f1"])
            # Preserve completed-language RQ1 metrics if a later stage is interrupted.
            for mode in splits:
                partial_path, _ = rile_result_paths(output_dir, model_slug, mode,
                                                    args.dataset, args.protocol)
                partial_path = partial_path.with_name(partial_path.stem + "_partial.json")
                partial_path.write_text(json.dumps(summaries[mode], indent=2) + "\n", encoding="utf-8")
            progress("language_complete", stage="extraction_and_rq1", language=language,
                     elapsed_seconds=round(time.monotonic() - language_started, 2))
            del dev_vectors, heldout_vectors, test_vectors
            dev_path.unlink(missing_ok=True)
            heldout_path.unlink(missing_ok=True)

        for mode in splits:
            evaluation = development if mode == "development" else held_out
            fixed_test = dev_fixed if mode == "development" else heldout_fixed
            summary = summaries[mode]
            summary_path, prediction_path = rile_result_paths(output_dir, model_slug, mode,
                                                              args.dataset, args.protocol)
            temporary_path = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
            try:
                with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
                    for source in languages:
                        for target in languages:
                            cell_started = time.monotonic()
                            target_truth = np.asarray([item["label"] for item in evaluation[target]])
                            predicted = np.empty(len(target_truth), dtype=np.int8)
                            sample_folds = np.empty(len(target_truth), dtype=np.int8) if mode == "development" else None
                            if mode == "development":
                                for fold, (train_groups, test_groups) in enumerate(folds):
                                    train = indices(development[source], train_groups, "manifesto_id")
                                    test = indices(evaluation[target], test_groups, "manifesto_id")
                                    labels = np.asarray([item["label"] for item in development[source]])
                                    probe = LogisticRegression(C=1.0, max_iter=1000).fit(
                                        np.asarray(dev_fixed[source][train], dtype=np.float32), labels[train])
                                    predicted[test] = probe.predict(
                                        np.asarray(fixed_test[target][test], dtype=np.float32))
                                    sample_folds[test] = fold
                            else:
                                labels = np.asarray([item["label"] for item in development[source]])
                                probe = LogisticRegression(C=1.0, max_iter=1000).fit(
                                    np.asarray(dev_fixed[source], dtype=np.float32), labels)
                                predicted[:] = probe.predict(np.asarray(fixed_test[target], dtype=np.float32))
                            cell_result = {
                                "source_language": source, "target_language": target,
                                "layer": f"Block {fixed_block}",
                                "accuracy": float(accuracy_score(target_truth, predicted)),
                                "balanced_accuracy": float(balanced_accuracy_score(target_truth, predicted)),
                                "f1": float(f1_score(target_truth, predicted, pos_label=1)),
                            }
                            summary["rq2"].append(cell_result)
                            progress("rq2_cell_complete", stage="rq2", split=mode,
                                     source_language=source, target_language=target,
                                     elapsed_seconds=round(time.monotonic() - cell_started, 2),
                                     accuracy=cell_result["accuracy"],
                                     balanced_accuracy=cell_result["balanced_accuracy"], f1=cell_result["f1"])
                            for i, (item, truth, guess) in enumerate(zip(evaluation[target], target_truth, predicted)):
                                result = {"item_id": item["item_id"], "manifesto_id": item["manifesto_id"],
                                          "model": args.model, "layer": f"Block {fixed_block}",
                                          "source_language": source, "target_language": target,
                                          "true_label": int(truth), "predicted_label": int(guess),
                                          "split": mode, "dataset": args.dataset}
                                if args.dataset == "rile_v2":
                                    result.update({"protocol": args.protocol,
                                                   "source_lang": item.get("source_lang"),
                                                   "code": item.get("code"),
                                                   "cmp_code": item.get("cmp_code")})
                                if sample_folds is not None:
                                    result["fold"] = int(sample_folds[i])
                                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                temporary_path.replace(prediction_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
            try:
                summary_temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                summary_temporary.replace(summary_path)
            finally:
                summary_temporary.unlink(missing_ok=True)
            print(f"Saved RILE summary: {summary_path}")
            print(f"Saved RILE row predictions: {prediction_path}")
            progress("outputs_saved", stage="save", split=mode, summary=str(summary_path),
                     predictions=str(prediction_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--dataset", choices=("legacy", "rile_v1", "rile_v2"), default="legacy")
    parser.add_argument("--protocol", choices=("fixed_survey_en_layer", "nested_source_peak",
                                                "budgeted_source_peak", "source_matrix_replay"))
    parser.add_argument("--selection-summary", help="Completed budgeted RILE-v2 summary for source-matrix replay")
    parser.add_argument("--model-revision", help="Exact 40-character model commit for source-matrix replay")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--fit-workers", type=int, default=1,
                        help="CPU processes for layer scoring in the full RILE-v2 budgeted_source_peak run")
    parser.add_argument("--pilot-convergence", action="store_true")
    parser.add_argument("--pilot-workers", action="store_true",
                        help="Benchmark budgeted RILE-v2 layer screens at 1, 2, 4, and 6 CPU workers")
    parser.add_argument("--pilot-workers-max-iter", type=int, default=3000,
                        help="Iteration cap for the worker pilot (default: 3000)")
    parser.add_argument("--pilot-language", choices=LANGUAGES, default="en")
    parser.add_argument("--pilot-layers", default="0,18",
                        help="Comma-separated hidden-state indices; embedding is 0, Block N is N+1")
    parser.add_argument("--rile-split", choices=("development", "heldout", "both"), default="development",
                        help="RILE evaluation split; both shares one development feature extraction")
    parser.add_argument("--data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--data-only", action="store_true",
                        help="Print RILE split counts and exit without loading a model")
    args = parser.parse_args()
    replay = args.protocol == "source_matrix_replay"
    if replay:
        if (args.dataset != "rile_v2" or not args.selection_summary or
                not re.fullmatch(r"[0-9a-f]{40}", args.model_revision or "") or
                args.rile_split != "development" or args.pilot_workers or args.pilot_convergence or
                args.fit_workers != 1):
            raise ValueError("Source-matrix replay requires RILE-v2 development, one fit worker, "
                             "a completed --selection-summary, and a 40-character --model-revision")
    elif args.selection_summary or args.model_revision:
        raise ValueError("--selection-summary and --model-revision are only for source-matrix replay")
    if args.fit_workers < 1:
        raise ValueError("--fit-workers must be positive")
    if args.fit_workers > 1 and (args.dataset != "rile_v2"
                                 or args.protocol != "budgeted_source_peak"
                                 or args.pilot_convergence or args.pilot_workers):
        raise ValueError("--fit-workers above 1 requires a full RILE-v2 budgeted_source_peak run")
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    if len(languages) != len(LANGUAGES) or set(languages) != set(LANGUAGES):
        raise ValueError(f"Exactly these six languages are required: {', '.join(LANGUAGES)}")
    default_rile_path = ("data/rile_v1/final_translations.jsonl" if args.dataset == "rile_v1"
                         else "data/rile_v2/final_translations.jsonl")
    data_path = Path(args.data or (default_rile_path if args.dataset.startswith("rile_")
                                   else "data/multilingual_statements.json"))
    args.data_path = data_path
    if args.dataset == "rile_v2":
        canonical_v2_path = Path(__file__).resolve().parents[1] / "data/rile_v2/final_translations.jsonl"
        if data_path.resolve() != canonical_v2_path.resolve():
            raise ValueError(f"RILE-v2 requires its canonical corrected dataset at {canonical_v2_path}; "
                             f"refusing input path {data_path.resolve()}")
    if args.protocol and args.dataset not in {"rile_v1", "rile_v2"}:
        raise ValueError("--protocol applies only to a RILE dataset")
    if args.dataset in {"rile_v1", "rile_v2"}:
        if args.dataset == "rile_v1" and args.protocol:
            raise ValueError("--protocol is for RILE-v2 only; RILE-v1 keeps its historical outputs")
        if args.dataset == "rile_v2" and not args.data_only and args.protocol is None:
            raise ValueError("RILE-v2 scored runs require an explicit --protocol")
        if args.pilot_convergence and (args.dataset != "rile_v2" or args.protocol != "nested_source_peak"):
            raise ValueError("--pilot-convergence requires RILE-v2 nested_source_peak")
        if args.pilot_workers and (args.dataset != "rile_v2" or args.protocol != "budgeted_source_peak"):
            raise ValueError("--pilot-workers requires RILE-v2 budgeted_source_peak")
        if args.pilot_workers and args.pilot_convergence:
            raise ValueError("--pilot-workers and --pilot-convergence are separate pilots")
        if args.protocol in {"nested_source_peak", "budgeted_source_peak", "source_matrix_replay"} and args.rile_split != "development":
            raise ValueError("Nested primary uses development only; category holdout is exploratory")
        data, held_out_data, held_out_count = load_rile_data(data_path, languages)
        item_ids = [item["item_id"] for item in data["en"]]
        manifesto_ids = [item["manifesto_id"] for item in data["en"]]
        folds = rile_folds(manifesto_ids)
        if any([item["item_id"] for item in data[language]] != item_ids
               for language in languages):
            raise ValueError("RILE language records are not aligned")
        heldout_item_ids = [item["item_id"] for item in held_out_data["en"]]
        if any([item["item_id"] for item in held_out_data[language]] != heldout_item_ids
               for language in languages):
            raise ValueError("RILE held-out language records are not aligned")
        if (any(train & test for train, test in folds)
                or set().union(*(test for _, test in folds)) != set(manifesto_ids)):
            raise ValueError("RILE manifesto folds overlap or omit a development group")
        if args.data_only:
            modes = ("development", "heldout") if args.rile_split == "both" else (args.rile_split,)
            prediction_fields = {
                mode: ["dataset", "split", "item_id", "manifesto_id", "model", "layer"] +
                      (["fold"] if mode == "development" else []) +
                      ["source_language", "target_language", "true_label", "predicted_label"]
                for mode in modes
            }
            readiness = {
                "dataset": args.dataset,
                "total_items": len(item_ids) + held_out_count,
                "development_items": len(item_ids),
                "held_out_items": held_out_count,
                "development_manifestos": len(set(manifesto_ids)),
                "heldout_manifestos": len({item["manifesto_id"] for item in held_out_data["en"]}),
                "heldout_manifestos_also_in_development": len(
                    {item["manifesto_id"] for item in data["en"]} &
                    {item["manifesto_id"] for item in held_out_data["en"]}),
                "label_counts": {str(label): sum(item["label"] == label for item in data["en"])
                                 for label in (0, 1)},
                "folds": [{"train_manifestos": len(train), "test_manifestos": len(test),
                           "train_items": sum(group in train for group in manifesto_ids),
                           "test_items": sum(group in test for group in manifesto_ids)}
                          for train, test in folds],
                "languages": languages,
                "prediction_fields": prediction_fields,
            }
            if args.dataset == "rile_v2":
                for fields in prediction_fields.values():
                    fields[1:1] = ["protocol", "source_lang", "code", "cmp_code"]
                if args.protocol in {"nested_source_peak", "budgeted_source_peak", "source_matrix_replay"}:
                    prediction_fields["development"] = [
                        "dataset", "protocol", "split", "model", "fold", "kind",
                        "source_language", "target_language", "layer_index", "item_id",
                        "manifesto_id", "code", "cmp_code", "source_lang",
                        "true_label", "predicted_label"]
                readiness.update({"protocol": "data_only_readiness",
                                  "status": "data_only; no model loaded or evaluation run",
                                  "input_path": str(data_path.resolve())})
            print(json.dumps(readiness, indent=2))
            return
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
        result_dir = Path(args.output_dir) / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        progress_name = (f"rile_v1_{model_slug}_progress.jsonl" if args.dataset == "rile_v1"
                         else f"rile_v2_{args.protocol}_{model_slug}_progress.jsonl")
        progress_path = result_dir / progress_name
        if args.dataset == "rile_v2":
            candidate_paths = [progress_path]
            if args.pilot_convergence:
                candidate_paths.append(result_dir / f"rile_v2_nested_source_peak_{model_slug}_pilot.json")
            if args.pilot_workers:
                candidate_paths.append(result_dir / f"rile_v2_budgeted_source_peak_{model_slug}_worker_pilot.json")
            for mode in (("development", "heldout") if args.rile_split == "both" else (args.rile_split,)):
                summary_path, prediction_path = rile_result_paths(result_dir, model_slug, mode,
                                                                  args.dataset, args.protocol)
                candidate_paths.extend((summary_path, prediction_path,
                                        summary_path.with_name(summary_path.stem + "_partial.json"),
                                        prediction_path.with_name(prediction_path.stem + "_partial.jsonl.gz")))
            if replay:
                candidate_paths.append(result_dir / f"rile_v2_source_matrix_replay_{model_slug}_probes.npz")
            existing = [str(path) for path in candidate_paths if path.exists()]
            if existing:
                raise FileExistsError("RILE-v2 output would overwrite existing files; choose a new --output-dir: "
                                      + ", ".join(existing))
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        progress_context = ({"dataset": "rile_v2", "protocol": args.protocol,
                             "input_path": str(data_path.resolve())}
                            if args.dataset == "rile_v2" else None)
        progress = progress_logger(progress_path, run_id, args.model, progress_context)
        start_fields = {"progress_file": str(progress_path), "dataset": args.dataset,
                        "split": args.rile_split, "fit_workers": args.fit_workers}
        if args.protocol == "fixed_survey_en_layer":
            start_fields["fixed_rq2_layer_provenance"] = (
                "Prior RQ2 English in-language peak from the earlier survey-statement study.")
        if args.dataset == "rile_v2":
            start_fields.update({"protocol": args.protocol, "input_path": str(data_path.resolve())})
        progress("run_started", stage="start", **start_fields)
        try:
            if replay:
                run_rile_source_matrix(args, data, languages, progress)
            elif args.protocol in {"nested_source_peak", "budgeted_source_peak"}:
                run_rile_nested(args, data, folds, languages, progress)
            else:
                run_rile_probe(args, data, held_out_data, folds, languages, progress)
            progress("run_complete", stage="complete")
        except Exception as error:
            progress("run_failed", stage="failed", error_type=type(error).__name__, error=str(error))
            raise
        return
    else:
        if args.rile_split != "development":
            raise ValueError("--rile-split applies only to a RILE dataset")
        if args.data_only:
            raise ValueError("--data-only is available only with --dataset rile_v1")
        data = load_data(data_path, languages)
        item_ids = [item["question_id"] for item in data["en"]]
        folds = canonical_folds(item_ids)
        if any([item["question_id"] for item in data[language]] != item_ids
               for language in languages):
            raise ValueError("Language records are not strictly paired and aligned")
        group_field, label_field = "question_id", "polarity"

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(args.device)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type in {"cuda", "mps"} else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    features = {language: extract(model, tokenizer, data[language], device) for language in languages}
    labels = {language: [item[label_field] for item in data[language]] for language in languages}
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)

    for language in languages:
        output = Path(args.output_dir) / "results" / f"multilingual_probe_{model_slug}_{language}.jsonl"
        temporary = output.with_suffix(output.suffix + ".tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for layer, target_features in enumerate(features[language]):
                    name = "Embedding" if layer == 0 else f"Block {layer - 1}"
                    modes = [("in_language", language)]
                    if language != "en":
                        modes.append(("zero_shot", "en"))
                    for mode, source_language in modes:
                        predictions = [None] * len(target_features)
                        sample_folds = [None] * len(target_features)
                        truth, predicted = [], []
                        for fold, (train_questions, test_questions) in enumerate(folds):
                            test = indices(data[language], test_questions, group_field)
                            train = indices(data[source_language], train_questions, group_field)
                            probe = LogisticRegression(C=1.0, max_iter=1000).fit(
                                [features[source_language][layer][i] for i in train],
                                [labels[source_language][i] for i in train])
                            values = probe.predict([target_features[i] for i in test])
                            for i, value in zip(test, values):
                                predictions[i], sample_folds[i] = int(value), fold
                            truth.extend(labels[language][i] for i in test)
                            predicted.extend(map(int, values))
                        accuracy = accuracy_score(truth, predicted)
                        f1 = f1_score(truth, predicted, pos_label=1)
                        for i, item in enumerate(data[language]):
                            result = {
                                "mode": mode, "source_language": source_language,
                                "target_language": language, "layer": name,
                                "fold": sample_folds[i], "accuracy": accuracy, "f1": f1,
                            }
                            result.update({
                                "true_polarity": labels[language][i],
                                "predicted_polarity": predictions[i],
                                "question_id": item["question_id"],
                                "statement": item["statement"],
                            })
                            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        print(f"{language} {mode} {name}: accuracy={accuracy:.4f} f1={f1:.4f}")
                    handle.flush()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
