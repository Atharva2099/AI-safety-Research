"""Amendment 4: is the beyond-tone signal just a preference for sentences without a negation word?

Test 3 of Amendment 3 (tone-wrong pairs) restricted to pairs with no negation word on either side.
Exploratory: written after Step 1 was run.
"""
import json

import numpy as np

from src.compute_diagnostics_summary import RES, ROOT, SLUGS
from src.negation_reslice import OFF, english_split, load_data, load_scores, mean_ci, per_question
from src.tone import tone_probs, valence
from src.tone_control import NEAR_TIE, prereg_commit


def main():
    en, _, _, marked = english_split()
    load_data(ROOT / "data" / "multilingual_statements.json")  # validation only
    probs, names = tone_probs("cardiff", "en", [x["statement"] for x in en])
    v = valence("cardiff", probs, names)
    dtone = v[1::2] - v[0::2]

    subsets = {"tone_wrong": dtone < 0, "tone_wrong_clean": (dtone < 0) & ~marked,
               "near_tie": np.abs(dtone) < NEAR_TIE, "near_tie_clean": (np.abs(dtone) < NEAR_TIE) & ~marked}
    print({k: int(m.sum()) for k, m in subsets.items()}, "questions\n")

    out = {"preregistration_amendment": "Amendment 4, docs/negation_preregistration.md",
           "preregistration_amendment_commit": prereg_commit(),
           "n_questions": {k: int(m.sum()) for k, m in subsets.items()}, "models": {}}
    s_idx, t_idx = tuple(np.array(OFF).T)
    for model in SLUGS:
        score, truth, _ = load_scores(model, en)
        _, pair = per_question(score, truth)
        probe_q = pair[:, s_idx, t_idx].mean(axis=1)  # off-diagonal pairwise per question
        out["models"][model] = {k: mean_ci(probe_q[m]) for k, m in subsets.items()}
        r = out["models"][model]
        print(f"{model:10s} tone-wrong {r['tone_wrong']['mean']:.3f} [{r['tone_wrong']['ci95'][0]:.3f}, {r['tone_wrong']['ci95'][1]:.3f}]"
              f"   tone-wrong & clean {r['tone_wrong_clean']['mean']:.3f} "
              f"[{r['tone_wrong_clean']['ci95'][0]:.3f}, {r['tone_wrong_clean']['ci95'][1]:.3f}]"
              f"  (n={out['n_questions']['tone_wrong_clean']})")

    out["negation_robust"] = {m: bool(out["models"][m]["tone_wrong_clean"]["ci95"][0] > 0.50) for m in SLUGS}
    print("\nbeyond-tone survives removing negation pairs:", out["negation_robust"])
    (RES / "clean_tone_wrong.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
