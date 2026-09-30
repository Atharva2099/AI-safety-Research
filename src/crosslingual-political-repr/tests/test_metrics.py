import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from src import metrics, negation_reslice  # noqa: E402
from src.compute_diagnostics_summary import RES as RES_SURVEY, ROOT as ROOT_SURVEY  # noqa: E402


def test_res_and_root_match_the_survey_module():
    """metrics.py must resolve to the same artifacts/results directory the live scripts used before the move."""
    assert metrics.ROOT == ROOT_SURVEY
    assert metrics.RES == RES_SURVEY == metrics.ROOT / "artifacts" / "results"


def test_negation_reslice_still_exports_the_moved_names():
    """The survey line keeps importing these from src.negation_reslice; they must be the same objects."""
    for name in ("auc", "mean_ci", "NEGATION", "NEGATION_BY_LANG", "BOOT_N", "BOOT_SEED"):
        assert getattr(negation_reslice, name) is getattr(metrics, name)


def test_auc_is_the_tie_halving_pairwise_rate():
    y = np.array([0, 0, 1, 1])
    assert metrics.auc(y, np.array([0.0, 1.0, 2.0, 3.0])) == 1.0
    assert metrics.auc(y, np.array([3.0, 2.0, 1.0, 0.0])) == 0.0
    assert metrics.auc(y, np.array([1.0, 1.0, 1.0, 1.0])) == 0.5  # all ties


def test_mean_ci_is_deterministic_and_brackets_the_mean():
    v = np.random.default_rng(0).normal(0.3, 1.0, 580)
    a, b = metrics.mean_ci(v), metrics.mean_ci(v)
    assert a == b and a["n_questions"] == 580
    assert a["ci95"][0] < a["mean"] == float(v.mean()) < a["ci95"][1]
    # 2-D input averages cells first, so it matches the row means fed in directly
    cells = np.stack([v, v + 1.0], axis=1)
    assert metrics.mean_ci(cells) == metrics.mean_ci(v + 0.5)


def test_negation_patterns_are_per_language():
    assert metrics.NEGATION_BY_LANG["en"] is metrics.NEGATION
    assert metrics.NEGATION.search("we should n't do that")
    assert metrics.NEGATION_BY_LANG["de"].search("wir werden nicht zahlen")
    assert metrics.NEGATION_BY_LANG["es"].search("no vamos a pagar")
    # the reason the mapping exists: the English pattern finds nothing in this German sentence
    assert not metrics.NEGATION.search("wir werden nicht zahlen")
