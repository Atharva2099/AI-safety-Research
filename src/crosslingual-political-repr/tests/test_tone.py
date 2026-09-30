import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.tone_control import cell_ci, tone_controlled  # noqa: E402


def synthetic(probe_signal, seed=0, n=580):
    """Per-question differences (+1 minus -1). Tone and negation are informative; the probe is noise or independent signal."""
    rng = np.random.default_rng(seed)
    tone = rng.normal(0.6, 0.8, n)
    neg = rng.choice([-1.0, 0.0, 1.0], n, p=[0.4, 0.5, 0.1])
    probe = rng.normal(probe_signal, 1.0, n)
    return tone, neg, probe


def test_noise_probe_gives_no_improvement():
    lo, hi = tone_controlled(*synthetic(0.0))["logloss_improvement_A_minus_B"]["ci95"]
    assert lo < 0 < hi


def test_independent_probe_signal_gives_improvement():
    lo, _ = tone_controlled(*synthetic(0.8))["logloss_improvement_A_minus_B"]["ci95"]
    assert lo > 0


def test_cell_ci_pools_selected_cells():
    values = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    mask = np.array([[True, True], [True, False], [False, False]])
    r = cell_ci(values, mask)
    assert r["mean"] == pytest.approx(2 / 3) and r["n_questions"] == 2 and r["n_cells"] == 3


@pytest.mark.skipif(os.environ.get("HF_HUB_OFFLINE") == "1", reason="needs the pinned model from the Hub (or HF cache)")
def test_tone_batch_size_invariance():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from src.compute_diagnostics_summary import ROOT
    from src.negation_reslice import load_data
    from src.tone import score_texts
    texts = [x["statement"] for x in load_data(ROOT / "data" / "multilingual_statements.json")["en"][:16]]
    p1, _ = score_texts("cardiff", texts, batch_size=1)
    p8, _ = score_texts("cardiff", texts, batch_size=8)
    assert (p1.argmax(1) == p8.argmax(1)).all() and np.abs(p1 - p8).max() < 1e-4
