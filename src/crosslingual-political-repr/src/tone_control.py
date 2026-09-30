"""Step 1: do the probes carry information beyond tone? Implements Amendment 3 of docs/negation_preregistration.md.

Unit = question. Everything is computed on per-question differences (+1 side minus -1 side).
Refuses to run unless docs/negation_preregistration.md is committed with no local changes; records that commit.
"""
import json
import platform
import subprocess

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.compute_diagnostics_summary import LANGS, RES, ROOT, SLUGS
from src.negation_reslice import BOOT_N, BOOT_SEED, OFF, english_split, load_data, load_scores, mean_ci, per_question
from src.tone import TONE_MODELS, cache_meta, tone_probs, valence

C, N_FOLDS, NEAR_TIE = 1.0, 5, 0.10
S_IDX, T_IDX = tuple(np.array(OFF).T)  # the 30 off-diagonal source->target cells
COVERAGE = {"cardiff": "trained on ar, en, fr, de, hi, it, es, pt (not zh, mr)",
            "nlptown": "trained on en, nl, de, fr, it, es (used on en only)"}


def pairwise(d):
    """+1 side higher = 1, tie = 0.5, lower = 0 (d = value(+1) - value(-1))."""
    return (d > 0) + 0.5 * (d == 0)


def tone_controlled(tone, neg, probe):
    """Check 2. Per-question difference features; conditional logit with held-out (GroupKFold by question) scoring.
    Model A = tone + negation, model B = tone + negation + probe. Returns per-question held-out log-loss and pairwise."""
    X = {"A": np.column_stack([tone, neg]), "B": np.column_stack([tone, neg, probe])}
    n = len(tone)
    q = np.arange(n)
    res = {}
    for name, x in X.items():
        ll, pw = np.empty(n), np.empty(n)
        xs, ys, gs = np.vstack([x, -x]), np.r_[np.ones(n), np.zeros(n)], np.r_[q, q]  # symmetrized rows
        for tr, _ in GroupKFold(N_FOLDS).split(xs, ys, gs):
            test = np.setdiff1d(q, gs[tr])  # held-out questions (both rows of a question share a fold)
            sd = xs[tr].std(axis=0)  # training-fold scale; symmetrized mean is exactly 0
            sd[sd == 0] = 1
            clf = LogisticRegression(C=C, fit_intercept=False).fit(xs[tr] / sd, ys[tr])
            m = x[test] / sd @ clf.coef_[0]  # held-out logit, y = 1 orientation
            ll[test], pw[test] = np.logaddexp(0, -m), pairwise(m)
        sd = xs.std(axis=0)
        sd[sd == 0] = 1
        coef = LogisticRegression(C=C, fit_intercept=False).fit(xs / sd, ys).coef_[0]
        res[name] = {"ll": ll, "pairwise": pw, "coef_std": coef.tolist()}
    return {"logloss_improvement_A_minus_B": mean_ci(res["A"]["ll"] - res["B"]["ll"]),
            "heldout_logloss": {k: float(res[k]["ll"].mean()) for k in res},
            "heldout_pairwise": {k: mean_ci(res[k]["pairwise"]) for k in res},
            "coef_std_full_data": {"A": dict(zip(["tone", "negation"], res["A"]["coef_std"])),
                                   "B": dict(zip(["tone", "negation", "probe"], res["B"]["coef_std"]))}}


def cell_ci(values, mask):
    """Pooled mean over the selected (question, cell) entries, bootstrap over questions (fresh BOOT_SEED rng)."""
    keep = mask.any(axis=1)
    num, den = (values * mask).sum(axis=1)[keep], mask.sum(axis=1)[keep]
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(num), size=(BOOT_N, len(num)))
    boots = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return {"mean": float(num.sum() / den.sum()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "n_questions": int(keep.sum()), "n_cells": int(den.sum())}


def english_checks(probe_pair, probe_margin, dtone, dneg, clean):
    """Checks 1-3 with one English tone difference per question."""
    head = {k: {"probe_minus_tone": mean_ci(probe_pair[m] - pairwise(dtone[m])), "probe": mean_ci(probe_pair[m]),
                "tone": mean_ci(pairwise(dtone[m]))} for k, m in {"all": np.ones(len(dtone), bool), "clean": clean}.items()}
    return {"check1_head_to_head": head, "check2_tone_controlled": tone_controlled(dtone, dneg, probe_margin),
            "check3_tone_wrong": {"tone_wrong": mean_ci(probe_pair[dtone < 0]),
                                  "near_tie": mean_ci(probe_pair[np.abs(dtone) < NEAR_TIE])}}


def target_checks(pair_cells, probe_margin, dtone_lang, dneg, clean):
    """Robustness 4b: cell s->t uses the tone of target language t (dtone_lang[q, lang])."""
    tone_cells = dtone_lang[:, T_IDX]  # [q, 30]
    head = {k: {"probe_minus_tone": mean_ci(pair_cells[m] - pairwise(tone_cells[m]))}
            for k, m in {"all": np.ones(len(dneg), bool), "clean": clean}.items()}
    non_en = dtone_lang[:, 1:].mean(axis=1)  # check 2 feature: mean tone difference over the 5 non-English languages
    return {"check1_head_to_head": head, "check2_tone_controlled": tone_controlled(non_en, dneg, probe_margin),
            "check3_tone_wrong": {"tone_wrong": cell_ci(pair_cells, tone_cells < 0),
                                  "near_tie": cell_ci(pair_cells, np.abs(tone_cells) < NEAR_TIE)}}


def verdict(main, v4a, v4b):
    lo = lambda r: r["check2_tone_controlled"]["logloss_improvement_A_minus_B"]["ci95"][0]
    rule = {"i_check2_cardiff_en_ci_gt_0": lo(main) > 0,
            "ii_check3_tone_wrong_lower_ci_gt_0.50": main["check3_tone_wrong"]["tone_wrong"]["ci95"][0] > 0.50,
            "iii_check2_ci_gt_0_under_4a_and_4b": lo(v4a) > 0 and lo(v4b) > 0}
    return rule, "probe carries information beyond tone" if all(rule.values()) else "not established"


def tone_diffs(key, data):
    """Per-question tone difference (+1 minus -1) for every language already needed."""
    langs = LANGS if key == "cardiff" else ["en"]
    return {l: (lambda v: v[1::2] - v[0::2])(valence(key, *tone_probs(key, l, [x["statement"] for x in data[l]])))
            for l in langs}


def prereg_commit():
    """Commit hash of the pre-registration; stop if it has uncommitted changes (so results can't precede the rules)."""
    doc = "docs/negation_preregistration.md"
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if git("status", "--porcelain", "--", doc):
        raise SystemExit(f"{doc} has uncommitted changes: commit Amendment 3 before running tone_control.")
    return git("log", "-1", "--format=%H", "--", doc)


def main():
    amendment_commit = prereg_commit()
    en, _, has_neg, marked = english_split()
    data = load_data(ROOT / "data" / "multilingual_statements.json")
    dneg = has_neg[1::2].astype(float) - has_neg[0::2]
    tone = {k: tone_diffs(k, data) for k in TONE_MODELS}
    dtone_lang = np.column_stack([tone["cardiff"][l] for l in LANGS])  # [580, 6]
    out = {"preregistration_amendment": "Amendment 3, docs/negation_preregistration.md", "preregistration_amendment_commit": amendment_commit,
           "boot": {"n": BOOT_N, "seed": BOOT_SEED, "unit": "question"},
           "logit": {"C": C, "fit_intercept": False, "folds": f"GroupKFold({N_FOLDS}) by question", "standardize": "training-fold SD"},
           "near_tie_threshold": NEAR_TIE, "tone_models": {k: cache_meta(k) for k in TONE_MODELS}, "tone_coverage": COVERAGE,
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
           "n_questions": {"all": 580, "clean": int((~marked).sum()), "tone_wrong_cardiff_en": int((tone["cardiff"]["en"] < 0).sum()),
                           "near_tie_cardiff_en": int((np.abs(tone["cardiff"]["en"]) < NEAR_TIE).sum())},
           "models": {}}
    print(f"{'model':10s} {'probe-tone':>18s} {'LL impr (en)':>22s} {'tone-wrong probe':>20s} {'4a LL lo':>9s} {'4b LL lo':>9s}  verdict")
    for model in SLUGS:
        score, truth, layer = load_scores(model, en)  # guarded: alignment + published baseline
        pair_cells = per_question(score, truth)[1][:, S_IDX, T_IDX]  # [580, 30]
        margin = (score[..., 1::2] - score[..., 0::2]).transpose(2, 0, 1)[:, S_IDX, T_IDX].mean(axis=1)
        r = {"layer": layer,
             "cardiff_en": english_checks(pair_cells.mean(axis=1), margin, tone["cardiff"]["en"], dneg, ~marked),
             "robust_4a_nlptown_en": english_checks(pair_cells.mean(axis=1), margin, tone["nlptown"]["en"], dneg, ~marked),
             "robust_4b_cardiff_target": target_checks(pair_cells, margin, dtone_lang, dneg, ~marked)}
        r["decision_rule"], r["verdict"] = verdict(r["cardiff_en"], r["robust_4a_nlptown_en"], r["robust_4b_cardiff_target"])
        out["models"][model] = r
        c = r["cardiff_en"]
        h, i, w = (c["check1_head_to_head"]["all"]["probe_minus_tone"], c["check2_tone_controlled"]["logloss_improvement_A_minus_B"],
                   c["check3_tone_wrong"]["tone_wrong"])
        f = lambda x: f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
        lo = lambda k: r[k]["check2_tone_controlled"]["logloss_improvement_A_minus_B"]["ci95"][0]
        print(f"{model:10s} {f(h):>18s} {f(i):>22s} {f(w):>20s} {lo('robust_4a_nlptown_en'):+9.3f} "
              f"{lo('robust_4b_cardiff_target'):+9.3f}  {r['verdict']}")
    (RES / "tone_control.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
