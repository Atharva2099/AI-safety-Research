"""Is the tone-matched manifesto set separable by tone models that had no say in the matching?

We match the two sides of each issue on cardiff tone scores. That guarantees *cardiff* cannot separate them.
This script checks independent models (product reviews, movie reviews, mixed English, tweets): if they also
sit at AUC 0.50 on the matched set, the matching generalises. If one still separates, matching on a single
model is not enough and we must match on several jointly.

    uv run --with torch --with transformers --with sentencepiece --with protobuf python -m src.manifesto_tone_multimodel

Reads the gitignored data/manifesto_raw.json (Manifesto Project data: never commit or redistribute).
"""
import json
from pathlib import Path

import numpy as np

from src.negation_reslice import auc
from src.tone import TONE_MODELS, tone_probs, valence

ROOT = Path(__file__).resolve().parents[1]
# RILE-keyed issues kept for the English set. NOTE: RILE membership is recalled, not yet checked
# against the codebook -- verify before any of this reaches a paper.
ISSUES = [("601", "602", "national way of life"), ("603", "604", "traditional morality"),
          ("406", "407", "protectionism"), ("504", "505", "welfare"),
          ("701", "702", "labour"), ("203", "204", "constitutionalism")]
VALIDATORS = ["nlptown", "sst2", "siebert", "twitter_rob"]
MIN_WORDS, CAP, BINS, SEED = 5, 300, np.linspace(-1, 1, 11), 20260915


def usable(r):
    t = str(r.get("text") or "").strip()
    return len(t.split()) >= MIN_WORDS and str(r.get("cmp_code")).split(".")[0].isdigit()


def load_issue_rows():
    rows = [r for r in json.loads((ROOT / "data" / "manifesto_raw.json").read_text()) if usable(r)]
    code = {id(r): str(r["cmp_code"]).split(".")[0] for r in rows}
    return rows, code


def match_on(rows, code, rng, keys=("cardiff",), n_bins=10):
    """Per issue: bin both sides on each model's tone, keep equal numbers per joint bin, cap at CAP pairs.

    One key reproduces the original single-model matching. Several keys match jointly, which closes the
    gap where a model that had no say in the matching can still separate the sides -- at the cost of
    data, since the number of cells grows with each model (hence coarser bins when matching jointly).
    """
    wanted = {c for a, b, _ in ISSUES for c in (a, b)}
    rows = [r for r in rows if code[id(r)] in wanted]
    edges = np.linspace(-1, 1, n_bins + 1)
    cell = {}
    for key in keys:
        probs, names = tone_probs(key, "manifesto_six_all", [r["text"] for r in rows])
        v = np.digitize(valence(key, probs, names), edges)
        for i, r in enumerate(rows):
            cell[id(r)] = cell.get(id(r), ()) + (int(v[i]),)

    matched, unmatched = [], []
    for pos_code, neg_code, name in ISSUES:
        A = [r for r in rows if code[id(r)] == pos_code]
        B = [r for r in rows if code[id(r)] == neg_code]
        ia = [cell[id(r)] for r in A]
        ib = [cell[id(r)] for r in B]
        keep_a, keep_b = [], []
        for k in sorted(set(ia) | set(ib)):
            pa = [A[i] for i in range(len(A)) if ia[i] == k]
            pb = [B[j] for j in range(len(B)) if ib[j] == k]
            m = min(len(pa), len(pb))
            keep_a += pa[:m]
            keep_b += pb[:m]
        m = min(len(keep_a), CAP)
        matched += [(name, 1, r) for r in keep_a[:m]] + [(name, 0, r) for r in keep_b[:m]]
        # size-matched random sample from the same issue, for the "before" comparison
        unmatched += [(name, 1, A[i]) for i in rng.choice(len(A), min(m, len(A)), replace=False)]
        unmatched += [(name, 0, B[j]) for j in rng.choice(len(B), min(m, len(B)), replace=False)]
    return matched, unmatched


def aucs(sample, key, cache_key):
    """Per-issue and pooled AUC of one tone model on (issue, label, row) triples."""
    probs, names = tone_probs(key, cache_key, [r["text"] for _, _, r in sample])
    v = valence(key, probs, names)
    y = np.array([lab for _, lab, _ in sample])
    issues = np.array([iss for iss, _, _ in sample])
    per = {iss: auc(y[issues == iss], v[issues == iss]) for iss in sorted(set(issues))}
    return per, auc(y, v)


SCHEMES = [(("cardiff",), 10, "cardiff only"), (("cardiff", "siebert"), 5, "cardiff + siebert (joint)")]


def evaluate(matched, unmatched, tag, suffix):
    """Validator AUCs for one matching scheme; every validator had no say in schemes it is not part of."""
    entry = {"n_pairs": len(matched) // 2, "models": {}}
    print(f"\n=== {tag}: {len(matched) // 2} pairs ===")
    print(f"{'model':<14}{'AUC unmatched':>15}{'AUC matched':>13}")
    for key in ["cardiff"] + VALIDATORS:
        _, pooled_u = aucs(unmatched, key, f"manifesto_random_{suffix}")
        per_m, pooled_m = aucs(matched, key, f"manifesto_matched_{suffix}")
        entry["models"][key] = {"model": TONE_MODELS[key][0], "pooled_unmatched": pooled_u,
                                "pooled_matched": pooled_m, "per_issue_matched": per_m}
        print(f"{key:<14}{pooled_u:>15.3f}{pooled_m:>13.3f}")

    print(f"{'issue':<24}" + "".join(f"{k:>13}" for k in ["cardiff"] + VALIDATORS))
    for iss in sorted({i for i, _, _ in matched}):
        n = sum(1 for i, lab, _ in matched if i == iss and lab == 1)
        print(f"{iss[:22]:<24}" + "".join(f"{entry['models'][k]['per_issue_matched'][iss]:>13.3f}" for k in ["cardiff"] + VALIDATORS) + f"  (n={n})")

    worst = max(abs(entry["models"][k]["pooled_matched"] - 0.5) for k in VALIDATORS)
    worst_iss = max(((abs(entry["models"][k]["per_issue_matched"][i] - 0.5), i, k)
                     for k in VALIDATORS for i in entry["models"][k]["per_issue_matched"]))
    entry["worst_validator_pooled_gap"] = worst
    entry["worst_validator_issue"] = {"issue": worst_iss[1], "model": worst_iss[2], "gap": worst_iss[0]}
    print(f"pooled worst validator gap {worst:.3f} | worst single issue {worst_iss[1]} / {worst_iss[2]} gap {worst_iss[0]:.3f}")
    return entry


def main():
    rng = np.random.default_rng(SEED)
    rows, code = load_issue_rows()
    out = {"schemes": {}}
    for keys, n_bins, tag in SCHEMES:
        matched, unmatched = match_on(rows, code, rng, keys=keys, n_bins=n_bins)
        suffix = "_".join(keys) + f"_{n_bins}"
        out["schemes"][tag] = {"matched_on": list(keys), "n_bins": n_bins,
                               **evaluate(matched, unmatched, tag, suffix)}
    print("\npooled gap under ~0.05 and no single issue above ~0.10 = matching holds up")
    (ROOT / "artifacts" / "results" / "manifesto_tone_multimodel.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
