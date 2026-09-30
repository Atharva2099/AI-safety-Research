"""Negation confound re-slice, exactly as pre-registered in docs/negation_preregistration.md (commit 0869f61)."""
import gzip
import json

import numpy as np

from src.compute_diagnostics_summary import ARTIFACT_FILES, LANGS, RES, ROOT, SLUGS
# Re-exported for callers that still import these from here (the moved definitions live in src/metrics.py).
from src.metrics import BOOT_N, BOOT_SEED, NEGATION, NEGATION_BY_LANG, auc, mean_ci  # noqa: F401


def load_data(path):
    """Same row order and checks as run_extraction_diagnostics.load_data (copied to avoid importing torch/transformers):
    one row per +/-1 statement, grouped by question, -1 before +1, label = int(polarity == 1)."""
    pairs = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        if row.get("polarity") in (-1, 1):
            pairs.setdefault(row["id"], []).append(row)
    if len(pairs) != 580 or any(len(v) != 2 for v in pairs.values()):
        raise ValueError("Expected 580 paired question IDs")
    if any({r["polarity"] for r in v} != {-1, 1} for v in pairs.values()):
        raise ValueError("Pairs must have opposite labels")
    return {l: [{"question_id": q, "statement": r[l], "label": int(r["polarity"] == 1)}
                for q, pair in pairs.items() for r in sorted(pair, key=lambda r: r["polarity"])] for l in LANGS}

PUBLISHED_OFF_DIAG = {"olmo": 0.7064, "ministral": 0.6813, "gemma": 0.8467, "qwen": 0.8068}
OFF = [(s, t) for s in range(6) for t in range(6) if s != t]
DIAG = [(s, s) for s in range(6)]


def scores_from_predictions(preds, en):
    """score[s, t, row] and truth[row] for the fair/current_raw lane, with alignment guards.
    Rows are ordered as load_data (-1 then +1 per question); en is load_data(...)["en"]."""
    rows = [r for r in preds if r["lane"] == "fair" and r["condition"] == "current_raw" and not r["norm_only"]]
    assert len(rows) == 36 * 1160, f"expected 41760 fair/current_raw rows, got {len(rows)}"
    assert all(r["feature_mode"] == "full" for r in rows), "unexpected feature_mode"
    keys = {(r["source_language"], r["target_language"], r["row_index"]) for r in rows}
    assert len(keys) == len(rows), "duplicated (source, target, row_index)"
    assert keys == {(s, t, i) for s in LANGS for t in LANGS for i in range(1160)}, "missing (source, target, row_index)"
    labels, qids = np.array([x["label"] for x in en]), [x["question_id"] for x in en]
    score, truth = np.full((6, 6, 1160), np.nan), np.full((6, 6, 1160), -1)
    for r in rows:
        s, t, i = LANGS.index(r["source_language"]), LANGS.index(r["target_language"]), r["row_index"]
        assert r["question_id"] == qids[i], f"question_id mismatch at row {i}"
        assert r["pred"] == int(r["decision_score"] > 0), f"pred != (decision_score > 0) at row {i}"
        score[s, t, i], truth[s, t, i] = r["decision_score"], r["truth"]
    assert (truth == labels).all() and (labels == np.arange(1160) % 2).all(), "label/row mapping"
    return score, labels


def check_published_baseline(model, score, truth):
    """Off-diagonal fair/current_raw accuracy must match the published baseline (docs/rq2_findings.md)."""
    off = float(np.mean([((score[s, t] > 0) == truth).mean() for s, t in OFF]))
    assert abs(off - PUBLISHED_OFF_DIAG[model]) < 5e-5, f"{model}: baseline {off:.4f} != published {PUBLISHED_OFF_DIAG[model]}"


def load_scores(model, en):
    """Guarded (score, truth, layer) for one model; every analysis script goes through this."""
    with gzip.open(RES / ARTIFACT_FILES[model], "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    score, truth = scores_from_predictions(data["predictions"], en)
    check_published_baseline(model, score, truth)
    return score, truth, data["layer"]


def per_question(score, truth):
    """Per-question accuracy [580, 6, 6] (0 / 0.5 / 1) and pairwise [580, 6, 6] (+1 statement scored above -1 statement)."""
    pred = (score > 0).astype(int)
    acc = (pred == truth).reshape(6, 6, 580, 2).mean(axis=3).transpose(2, 0, 1)
    neg_side, pos_side = score[..., 0::2], score[..., 1::2]
    pair = ((pos_side > neg_side) + 0.5 * (pos_side == neg_side)).transpose(2, 0, 1)
    return acc, pair


def mean_auc(score, truth, rows, cells):
    return float(np.mean([auc(truth[rows], score[s, t, rows]) for s, t in cells]))


def english_split():
    """English rows, labels, per-statement negation flag, and per-question 'negation-marked' (either side)."""
    en = load_data(ROOT / "data" / "multilingual_statements.json")["en"]
    has_neg = np.array([bool(NEGATION.search(x["statement"])) for x in en])
    labels = np.array([x["label"] for x in en])
    return en, labels, has_neg, has_neg.reshape(580, 2).any(axis=1)


def main():
    en, labels, has_neg, marked = english_split()
    rule_correct = int(((~has_neg).astype(int) == labels).sum())  # rule: negation -> label 0 (-1), else label 1 (+1)
    table = {f"{'+1' if l else '-1'}_{'neg' if n else 'no_neg'}": int(((labels == l) & (has_neg == n)).sum())
             for l in (1, 0) for n in (True, False)}
    print("label x negation:", table, f"| rule-only baseline {rule_correct}/1160 = {rule_correct / 1160:.4f}")
    print(f"questions: {marked.sum()} negation-marked, {(~marked).sum()} clean\n")

    out = {"preregistration_commit": "0869f61", "label_x_negation": table, "rule_only_correct": rule_correct,
           "n_marked": int(marked.sum()), "n_clean": int((~marked).sum()), "models": {}}
    subsets = {"full": np.ones(580, bool), "negation_marked": marked, "clean": ~marked}

    for model in SLUGS:
        score, truth, layer = load_scores(model, en)  # guarded: alignment + published baseline
        acc, pair = per_question(score, truth)
        entry = {"layer": layer}
        for name, qmask in subsets.items():
            rows = np.repeat(qmask, 2)
            entry[name] = {}
            for part, cells in (("off_diagonal", OFF), ("diagonal", DIAG)):
                idx = tuple(np.array(cells).T)
                entry[name][part] = {"pairwise": mean_ci(pair[qmask][:, idx[0], idx[1]]),
                                     "accuracy": mean_ci(acc[qmask][:, idx[0], idx[1]]),
                                     "auc": mean_auc(score, truth, rows, cells)}
        out["models"][model] = entry
        o = {k: entry[k]["off_diagonal"]["pairwise"] for k in subsets}
        print(f"{model:10s} L{layer:>2} off-diag pairwise  full {o['full']['mean']:.3f}  "
              f"marked {o['negation_marked']['mean']:.3f}  clean {o['clean']['mean']:.3f} "
              f"[{o['clean']['ci95'][0]:.3f}, {o['clean']['ci95'][1]:.3f}]")

    # ---- Pre-registered judgements (pairwise, off-diagonal) ----
    m = out["models"]
    clean = {k: m[k]["clean"]["off_diagonal"]["pairwise"] for k in m}
    drop = {k: m[k]["full"]["off_diagonal"]["pairwise"]["mean"] - clean[k]["mean"] for k in m}
    out["predictions"] = {
        "P1_gemma_qwen_clean_lower_ci_gt_0.60": all(clean[k]["ci95"][0] > 0.60 for k in ("gemma", "qwen")),
        "P2_olmo_ministral_clean_le_0.58": all(clean[k]["mean"] <= 0.58 for k in ("olmo", "ministral")),
        "P3_drop_olmo_ministral_gt_gemma_qwen": min(drop["olmo"], drop["ministral"]) > max(drop["gemma"], drop["qwen"]),
    }
    out["drop_full_minus_clean"] = drop
    print("\n", json.dumps(out["predictions"], indent=1), "\ndrops:", {k: round(v, 3) for k, v in drop.items()})
    (RES / "negation_reslice.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
