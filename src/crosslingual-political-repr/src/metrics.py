"""Shared paths and small metric helpers used by both the survey and the Manifesto/RILE code lines.

Moved here verbatim from src/negation_reslice.py (auc, mean_ci, BOOT_N, BOOT_SEED, NEGATION,
NEGATION_BY_LANG) so the live dataset pipeline does not import a survey analysis script.
ROOT/RES resolve to the same paths as in src/compute_diagnostics_summary.py.
"""
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts" / "results"

NEGATION = re.compile(r"\b(not|no|never|nor|none|nobody|nothing|neither|without|cannot)\b|n['’]t\b", re.I)
# Per-language negation markers. The English list is the pre-registered one above; de/es follow the
# lists declared in docs/negation_preregistration.md. Applying the English pattern to German or Spanish
# text matches almost nothing, so a "balanced" negation rate there would be an artifact, not a finding.
NEGATION_BY_LANG = {
    "en": NEGATION,
    "de": re.compile(r"\b(nicht|kein\w*|nie|niemals|nichts|niemand|ohne|weder)\b", re.I),
    "es": re.compile(r"\b(no|nunca|ni|sin|nada|nadie|jamás|jamas|tampoco)\b", re.I),
}
BOOT_N, BOOT_SEED = 5000, 20260915


def mean_ci(per_q, seed=BOOT_SEED):
    """Mean over questions (cells averaged first if 2-D), with a 95% bootstrap interval that resamples question IDs.
    A fresh RNG per call: the same quantity always gets the same CI, and two equal-length per-question vectors
    get the same resampled question indices (so a CI of a per-question difference is a paired bootstrap)."""
    rng = np.random.default_rng(seed)
    v = per_q.mean(axis=1) if per_q.ndim == 2 else per_q
    boots = v[rng.integers(0, len(v), size=(BOOT_N, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], "n_questions": len(v)}


def auc(y, s):
    """P(random +1 statement scores above random -1 statement); ties count half."""
    pos, neg = s[y == 1][:, None], s[y == 0][None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())
