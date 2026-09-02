"""Compute final RQ2 summary matrices, parity metrics, 28 primary contrasts, and simultaneous bootstrap intervals."""
import gzip
import json
import collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts" / "results"
LANGS = ["en", "es", "de", "zh", "hi", "mr"]
SLUGS = {
    "olmo": "allenai_Olmo-3-7B-Instruct",
    "ministral": "mistralai_Ministral-8B-Instruct-2410",
    "gemma": "google_gemma-2-9b-it",
    "qwen": "Qwen_Qwen3.5-9B",
}
ARTIFACT_FILES = {
    key: f"extraction_diagnostics_{slug}.json.gz"
    for key, slug in SLUGS.items()
}
ARTIFACT_FILES["qwen"] = "extraction_diagnostics_Qwen_Qwen3.5-9B_layer12.json.gz"
HISTORICAL_FILES = {
    key: f"cross_lingual_matrix_{slug}.json"
    for key, slug in SLUGS.items()
}
HISTORICAL_FILES["qwen"] = "cross_lingual_matrix_Qwen_Qwen3.5-9B_block12.json"
FAIR_CONDS = ["current_raw", "current_l2", "content_raw", "content_l2",
              "mean_raw", "mean_l2", "stripped_raw", "stripped_l2"]
NORM_CONDS = ["current_raw", "content_raw", "mean_raw", "stripped_raw"]
OFF = [(s, t) for s in range(6) for t in range(6) if s != t]
BOOT_N = 5000
BOOT_SEED = 20260825


def load_artifact(key: str):
    path = RES / ARTIFACT_FILES[key]
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    
    results = {}
    by_cell = collections.defaultdict(list)
    for r in data["predictions"]:
        key_tuple = (r["lane"], r["norm_only"], r["condition"], r["source_language"], r["target_language"])
        by_cell[key_tuple].append((r["row_index"], r["question_id"], r["pred"] == r["truth"]))
    
    for k, rows in by_cell.items():
        rows_sorted = sorted(rows, key=lambda x: x[0])
        assert len(rows_sorted) == 1160, f"Expected 1160 rows, got {len(rows_sorted)} for {k}"
        results[k] = np.array([x[2] for x in rows_sorted], dtype=bool)
        
    meta = {
        "model": data["model"],
        "layer": data["layer"],
        "data_hash": data.get("data_hash"),
        "conditions": data.get("conditions"),
        "diagnostics": data.get("diagnostics"),
        "extraction_environment": data.get("extraction_environment"),
        "probe_environment": data.get("probe_environment"),
        "fits": data.get("fits", [])
    }
    return results, meta


def make_A(cell_results: dict, lane: str, cond: str, norm_only: bool = False) -> np.ndarray:
    """A[q, s, t] per-question mean correctness for one condition."""
    A = np.empty((580, 6, 6), dtype=np.float32)
    for s_idx, s in enumerate(LANGS):
        for t_idx, t in enumerate(LANGS):
            corr = cell_results[(lane, norm_only, cond, s, t)]
            A[:, s_idx, t_idx] = corr.reshape(580, 2).mean(axis=1)
    return A


def macro_off(M: np.ndarray) -> float:
    return float(np.mean([M[s, t] for s, t in OFF]))


def run_summary(output_path: Path | None = None) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, 580, size=(BOOT_N, 580))
    all_boot_contrasts = {}

    summary = {
        "schema_version": 2,
        "bootstrap": {
            "n_resamples": BOOT_N,
            "seed": BOOT_SEED,
            "unit": "question ID",
            "description": "Question-set simultaneous 95% max-statistic bootstrap intervals across 28 primary contrasts with centered error distribution"
        },
        "models": {}
    }

    contrast_defs = [
        ("content_raw - current_raw", "content_raw", "current_raw"),
        ("mean_raw - current_raw", "mean_raw", "current_raw"),
        ("stripped_raw - current_raw", "stripped_raw", "current_raw"),
        ("current_l2 - current_raw", "current_l2", "current_raw"),
        ("content_l2 - content_raw", "content_l2", "content_raw"),
        ("mean_l2 - mean_raw", "mean_l2", "mean_raw"),
        ("stripped_l2 - stripped_raw", "stripped_l2", "stripped_raw"),
    ]

    for mkey in SLUGS:
        cell_results, meta = load_artifact(mkey)
        entry = {
            "model": meta["model"],
            "layer": meta["layer"],
            "data_hash": meta.get("data_hash"),
            "extraction_environment": meta.get("extraction_environment"),
            "probe_environment": meta.get("probe_environment"),
            "fits_count": len(meta.get("fits", []))
        }
        
        hist_path = RES / HISTORICAL_FILES[mkey]
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        A_leg = make_A(cell_results, "legacy", "current_raw")
        M_leg = A_leg.mean(axis=0)
        diff = np.abs(M_leg - np.asarray(hist["matrix"]))
        entry["legacy_parity"] = {
            "historical_file": hist_path.name,
            "new_diagonal": [round(float(x), 4) for x in M_leg.diagonal()],
            "hist_diagonal": [round(float(x), 4) for x in np.asarray(hist["matrix"]).diagonal()],
            "max_abs_cell_diff": float(diff.max()),
            "mean_abs_cell_diff": float(diff.mean()),
            "within_0.5pp_gate": bool(diff.max() <= 0.005)
        }
        
        A_fair = {c: make_A(cell_results, "fair", c) for c in FAIR_CONDS}
        entry["conditions"] = {}
        for c in FAIR_CONDS:
            M = A_fair[c].mean(axis=0)
            entry["conditions"][c] = {
                "matrix": M.round(6).tolist(),
                "diagonal_mean": float(np.mean([M[i, i] for i in range(6)])),
                "off_diagonal_macro_mean": macro_off(M)
            }
            
        entry["norm_only_controls"] = {}
        for c in NORM_CONDS:
            A_norm = make_A(cell_results, "fair", c, norm_only=True)
            M_norm = A_norm.mean(axis=0)
            entry["norm_only_controls"][c] = {
                "diagonal_mean": float(np.mean([M_norm[i, i] for i in range(6)])),
                "off_diagonal_macro_mean": macro_off(M_norm)
            }
            
        diag = meta.get("diagnostics") or {}
        ts = {}
        for lang, rows in diag.items():
            if rows and isinstance(rows[0], dict):
                ts[lang] = {
                    "statements": len(rows),
                    "truncated_count": sum(1 for r in rows if r.get("truncated")),
                    "mean_token_count": float(np.mean([r.get("token_count", 0) for r in rows])),
                    "mean_chars_per_token": float(np.mean([r.get("chars_per_token", 0) for r in rows])),
                    "terminal_is_punctuation_rate": float(np.mean([bool(r.get("terminal_is_punctuation")) for r in rows])),
                    "byte_fallback_count": sum(r.get("byte_fallback", 0) for r in rows)
                }
        entry["tokenizer_summary"] = ts
        
        boot = {}
        for c in FAIR_CONDS:
            boot[c] = np.array([np.mean([A_fair[c][draw, s, t].mean() for s, t in OFF]) for draw in draws])
            
        entry["primary_contrasts"] = {}
        model_boot = {}
        for name, a, b in contrast_defs:
            est = float(macro_off(A_fair[a].mean(axis=0)) - macro_off(A_fair[b].mean(axis=0)))
            diff_boot = boot[a] - boot[b]
            model_boot[name] = diff_boot
            entry["primary_contrasts"][name] = {"estimate": est, "boot_samples": diff_boot}
            
        all_boot_contrasts[mkey] = model_boot
        summary["models"][mkey] = entry

    names = []
    stack = []
    for mkey in SLUGS:
        for name, diff_boot in all_boot_contrasts[mkey].items():
            est = summary["models"][mkey]["primary_contrasts"][name]["estimate"]
            names.append((mkey, name))
            stack.append(np.abs(diff_boot - est))

    stack = np.vstack(stack)
    max_err = np.max(stack, axis=0)
    crit_halfwidth = float(np.quantile(max_err, 0.95))
    summary["simultaneous"] = {
        "method": "Max-statistic simultaneous 95% interval over 28 contrasts (7 per model x 4 models) with centered bootstrap errors",
        "critical_halfwidth": crit_halfwidth
    }

    for mkey in SLUGS:
        for name in summary["models"][mkey]["primary_contrasts"]:
            est = summary["models"][mkey]["primary_contrasts"][name]["estimate"]
            summary["models"][mkey]["primary_contrasts"][name]["simultaneous_95pct_ci"] = [est - crit_halfwidth, est + crit_halfwidth]
            del summary["models"][mkey]["primary_contrasts"][name]["boot_samples"]

    if output_path is not None:
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        
    return summary


if __name__ == "__main__":
    out = RES / "extraction_diagnostics_summary_qwen_layer12.json"
    run_summary(out)
    print(f"Summary computed and written to {out}")
