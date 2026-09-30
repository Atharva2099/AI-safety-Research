"""P4 sentiment check, as pre-registered in docs/negation_preregistration.md (Amendment 1)."""
import json

import numpy as np

from src.compute_diagnostics_summary import SLUGS, RES
from src.negation_reslice import OFF, english_split, load_scores, mean_ci, per_question
from src.tone import TONE_MODELS, cache_meta, tone_probs, valence


def main():
    en, labels, _, marked = english_split()
    probs, names = tone_probs("cardiff", "en", [x["statement"] for x in en])
    print("sentiment labels:", names)
    top = probs.argmax(axis=1)

    # label x sentiment table: how often is +1 'positive' and -1 'negative'?
    table = {f"{'+1' if l else '-1'}_{names[k]}": int(((labels == l) & (top == k)).sum()) for l in (1, 0) for k in range(3)}
    print("label x sentiment:", table)

    # Exploratory (not pre-registered): sentiment-only pairwise baseline -- does the +1 side sound more positive?
    v = valence("cardiff", probs, names)
    sent_pair = (v[1::2] > v[0::2]) + 0.5 * (v[1::2] == v[0::2])

    matched = top[0::2] == top[1::2]  # both sides of the pair get the same sentiment label
    subsets = {"clean": ~marked, "sentiment_matched": matched, "clean_and_matched": ~marked & matched}
    print({k: int(v.sum()) for k, v in subsets.items()}, "questions\n")

    out = {"preregistration_commit": "0869f61", "sentiment_model": TONE_MODELS["cardiff"][0],
           "sentiment_model_meta": cache_meta("cardiff"), "label_x_sentiment": table,
           "n_questions": {k: int(v.sum()) for k, v in subsets.items()},
           "sentiment_only_pairwise_baseline": {k: float(sent_pair[v].mean()) for k, v in {"full": np.ones(580, bool), **subsets}.items()},
           "per_statement_sentiment": probs.round(4).tolist(), "models": {}}
    print("sentiment-only pairwise baseline (exploratory):", {k: round(v, 3) for k, v in out["sentiment_only_pairwise_baseline"].items()})

    idx = tuple(np.array(OFF).T)
    for model in SLUGS:
        score, truth, _ = load_scores(model, en)  # guarded: alignment + published baseline
        acc, pair = per_question(score, truth)
        out["models"][model] = {k: {"pairwise": mean_ci(pair[m][:, idx[0], idx[1]]),
                                    "accuracy": mean_ci(acc[m][:, idx[0], idx[1]])} for k, m in subsets.items()}
        r = out["models"][model]["clean_and_matched"]["pairwise"]
        print(f"{model:10s} clean+matched off-diag pairwise {r['mean']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")

    cm = {k: out["models"][k]["clean_and_matched"]["pairwise"] for k in out["models"]}
    out["predictions"] = {"P4_gemma_qwen_clean_matched_lower_ci_gt_0.58": all(cm[k]["ci95"][0] > 0.58 for k in ("gemma", "qwen"))}
    print("\n", out["predictions"])
    (RES / "sentiment_check.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
