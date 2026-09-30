import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.negation_reslice import LANGS, auc, load_data, mean_ci, per_question, scores_from_predictions  # noqa: E402


def toy():
    """Synthetic English rows (580 questions, -1 then +1) and matching fair/current_raw predictions."""
    en = [{"question_id": 1000 + i // 2, "statement": "x", "label": i % 2} for i in range(1160)]
    rng = np.random.default_rng(0)
    preds = []
    for s in LANGS:
        for t in LANGS:
            for i, d in enumerate(rng.normal(size=1160)):
                preds.append({"lane": "fair", "condition": "current_raw", "norm_only": False, "feature_mode": "full",
                              "source_language": s, "target_language": t, "row_index": i, "question_id": 1000 + i // 2,
                              "truth": i % 2, "decision_score": float(d), "pred": int(d > 0)})
    preds.append({**preds[0], "lane": "legacy"})  # other lanes are ignored
    return preds, en


def test_guards_accept_valid_rows():
    preds, en = toy()
    score, truth = scores_from_predictions(preds, en)
    assert score.shape == (6, 6, 1160) and (truth == np.arange(1160) % 2).all()


@pytest.mark.parametrize("break_it", [
    lambda p: p.__setitem__(1, dict(p[0])),                                         # duplicated row
    lambda p: p.pop(5),                                                             # dropped row
    lambda p: (p[3].__setitem__("row_index", 7), p[7].__setitem__("row_index", 3)),  # shuffled row_index
    lambda p: p[4].__setitem__("question_id", 1999),                                # wrong question_id
    lambda p: p[6].__setitem__("decision_score", -p[6]["decision_score"]),          # pred/score mismatch
    lambda p: p[8].__setitem__("truth", 1 - p[8]["truth"]),                         # truth != label
    lambda p: p[9].__setitem__("feature_mode", "norm_only"),                        # wrong feature mode
])
def test_guards_reject_broken_rows(break_it):
    preds, en = toy()
    break_it(preds)
    with pytest.raises(AssertionError):
        scores_from_predictions(preds, en)


def test_load_data_rejects_same_polarity_twice(tmp_path):
    rows = [{"id": q, "polarity": p, **{l: "x" for l in LANGS}} for q in range(580) for p in (-1, 1)]
    path = tmp_path / "d.json"
    path.write_text(json.dumps(rows))
    assert len(load_data(path)["en"]) == 1160
    rows[1]["polarity"] = -1
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        load_data(path)


def test_pairwise_orientation():
    score = np.zeros((6, 6, 1160))
    score[..., 0], score[..., 1] = -1.0, 2.0   # q0: +1 side higher -> 1
    score[..., 2], score[..., 3] = 3.0, 1.0    # q1: -1 side higher -> 0
    score[..., 4], score[..., 5] = 0.5, 0.5    # q2: tie -> 0.5
    acc, pair = per_question(score, np.arange(1160) % 2)
    assert pair[0].min() == 1 and pair[1].max() == 0 and (pair[2] == 0.5).all()
    assert acc[0, 0, 0] == 1 and acc[1, 0, 0] == 0.5


def test_auc_matches_brute_force():
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 40), rng.integers(0, 5, 40).astype(float)  # integer scores -> ties
    brute = [1.0 if a > b else 0.5 if a == b else 0.0 for a in s[y == 1] for b in s[y == 0]]
    assert auc(y, s) == pytest.approx(np.mean(brute))


def test_mean_ci_deterministic_and_paired():
    rng = np.random.default_rng(2)
    a, b = rng.random((50, 30)), rng.random(50)
    assert mean_ci(a) == mean_ci(a)
    # same-length vectors are resampled with the same question indices
    boots = lambda v: mean_ci(v)["ci95"]
    assert np.allclose(np.array(boots(a.mean(axis=1) + 1.0)), np.array(boots(a.mean(axis=1))) + 1.0)
    assert mean_ci(b)["n_questions"] == 50
