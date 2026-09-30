"""Does the Manifesto data have the same tone confound as the survey data?

For each opposing category pair (e.g. 601 national-way-of-life positive vs 602 negative), measure how well
tone alone separates the two sides. Tone = P(positive) - P(negative) from the cached cardiff model.
AUC 0.50 = tone is useless; 1.00 = tone alone separates the sides perfectly.

Military (104/105) is excluded from the main table and reported separately: its "negative" side is mostly
condemnation of atrocities rather than an anti-military stance, which makes it a tone contrast by construction.

    uv run --with torch --with transformers --with sentencepiece --with protobuf python -m src.manifesto_tone_check

Reads the gitignored data/manifesto_raw.json (Manifesto Project data: never commit or redistribute).
"""
import json
from pathlib import Path

import numpy as np

from src.negation_reslice import NEGATION, auc
from src.tone import tone_probs, valence

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("601", "602", "national way of life"), ("603", "604", "traditional morality"),
         ("406", "407", "protectionism"), ("504", "505", "welfare"), ("506", "507", "education"),
         ("108", "110", "EU integration"), ("203", "204", "constitutionalism"),
         ("301", "302", "centralisation"), ("607", "608", "multiculturalism"), ("701", "702", "labour")]
EXCLUDED = [("104", "105", "military (excluded)")]
MIN_WORDS, MIN_SIDE, BOOT_N, BOOT_CAP, SEED = 5, 30, 300, 1500, 20260915


def usable(row):
    text = str(row.get("text") or "").strip()
    return len(text.split()) >= MIN_WORDS and str(row.get("cmp_code")).split(".")[0].isdigit()


def auc_ci(va, vb, rng):
    """Bootstrap 95% interval for the tone AUC, resampling sentences within each side."""
    a, b = va[:BOOT_CAP], vb[:BOOT_CAP]
    draws = []
    for _ in range(BOOT_N):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        draws.append(auc(np.r_[np.ones(len(ra)), np.zeros(len(rb))], np.r_[ra, rb]))
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def main():
    rows = [r for r in json.loads((ROOT / "data" / "manifesto_raw.json").read_text()) if usable(r)]
    code = {id(r): str(r["cmp_code"]).split(".")[0] for r in rows}
    wanted = {c for a, b, _ in PAIRS + EXCLUDED for c in (a, b)}
    rows = [r for r in rows if code[id(r)] in wanted]
    print(f"{len(rows)} sentences in the paired categories (>= {MIN_WORDS} words)\n")

    texts = [r["text"] for r in rows]
    probs, names = tone_probs("cardiff", "manifesto_pairs_en", texts)
    v = valence("cardiff", probs, names)
    tone = {id(r): v[i] for i, r in enumerate(rows)}
    has_neg = {id(r): bool(NEGATION.search(texts[i])) for i, r in enumerate(rows)}

    rng = np.random.default_rng(SEED)
    print(f"{'issue':<24}{'n+':>6}{'n-':>6}{'tone AUC':>10}{'95% CI':>18}{'neg+':>7}{'neg-':>7}")
    out = {}
    for pos_code, neg_code, name in PAIRS + EXCLUDED:
        a = [r for r in rows if code[id(r)] == pos_code]
        b = [r for r in rows if code[id(r)] == neg_code]
        if min(len(a), len(b)) < MIN_SIDE:
            print(f"{name:<24}{len(a):>6}{len(b):>6}   too few on one side")
            continue
        va, vb = np.array([tone[id(r)] for r in a]), np.array([tone[id(r)] for r in b])
        a_uc = auc(np.r_[np.ones(len(va)), np.zeros(len(vb))], np.r_[va, vb])
        lo, hi = auc_ci(va, vb, rng)
        na, nb = np.mean([has_neg[id(r)] for r in a]), np.mean([has_neg[id(r)] for r in b])
        out[name] = {"n_pos": len(a), "n_neg": len(b), "tone_auc": a_uc, "tone_auc_ci95": [lo, hi],
                     "tone_pos": float(va.mean()), "tone_neg": float(vb.mean()),
                     "negation_rate_pos": float(na), "negation_rate_neg": float(nb)}
        print(f"{name:<24}{len(a):>6}{len(b):>6}{a_uc:>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}{na:>7.2f}{nb:>7.2f}")

    usable_pairs = sum(min(out[n]["n_pos"], out[n]["n_neg"]) for _, _, n in PAIRS if n in out)
    out["_balanced_pairs_available"] = usable_pairs
    print(f"\nbalanced pairs available across kept issues: {usable_pairs}")
    (ROOT / "artifacts" / "results" / "manifesto_tone_check.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
