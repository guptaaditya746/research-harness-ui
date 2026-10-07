"""Confidence intervals and paired tests over questions (SciPy).

With about 30 questions the intervals are wide; that is the honest answer, and
the reason to report them. The unit of resampling is the question: repetitions
of one question are averaged first, because they are not independent samples.

- Intervals are bias-corrected and accelerated (BCa) bootstrap intervals
  (Efron and Tibshirani, 1993, ch. 14): at this sample size the plain percentile
  interval covers less than its nominal 95%. When BCa cannot be computed (a
  degenerate jackknife), the percentile interval is used and ``method`` says so.
- One value has no interval (``low``/``high`` are None); values that never vary
  have a zero-width one.
- A paired difference also gets a two-sided sign-flip permutation p-value: under
  "the two paths score alike" each per-question difference is as likely to have
  either sign. Exact up to 13 questions, Monte Carlo (seeded) above.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from scipy.stats import bootstrap, permutation_test


def mean_ci(values: Sequence[float], *, confidence: float = 0.95, resamples: int = 2000,
            seed: int = 0) -> dict[str, float | int | str | None]:
    """Mean with a BCa bootstrap interval (percentile when BCa is degenerate)."""
    data = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not data:
        return {"mean": None, "low": None, "high": None, "n": 0, "method": None}
    centre = sum(data) / len(data)
    if len(data) < 2:
        return {"mean": centre, "low": None, "high": None, "n": 1, "method": None}
    if len(set(data)) == 1:
        return {"mean": centre, "low": centre, "high": centre, "n": len(data), "method": "constant"}
    for method in ("BCa", "percentile"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = bootstrap((np.asarray(data),), np.mean, confidence_level=confidence, n_resamples=resamples,
                               method=method, rng=np.random.default_rng(seed))
        low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
        if not (math.isnan(low) or math.isnan(high)):
            return {"mean": centre, "low": low, "high": high, "n": len(data), "method": method}
    return {"mean": centre, "low": None, "high": None, "n": len(data), "method": None}


def sign_flip_p(differences: Sequence[float], *, resamples: int = 9999, seed: int = 0) -> float | None:
    """Two-sided p-value of "mean difference = 0" by flipping the sign of each paired difference."""
    data = np.asarray([float(value) for value in differences], dtype=float)
    if data.size == 0:
        return None
    if data.size == 1 or not np.any(data):
        return 1.0                                   # both signs are equally extreme
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_test((data,), np.mean, permutation_type="samples", vectorized=True,
                                  n_resamples=resamples, alternative="two-sided",
                                  rng=np.random.default_rng(seed))
    return float(min(1.0, result.pvalue))


def paired_difference_ci(left: Sequence[float], right: Sequence[float], **options) -> dict[str, float | int | str | None]:
    """Mean of ``left - right`` over the same questions, with its interval and sign-flip p-value."""
    if len(left) != len(right):
        raise ValueError("paired values need the same questions on both sides")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    return {**mean_ci(differences, **options), "p": sign_flip_p(differences)}


__all__ = ["mean_ci", "paired_difference_ci", "sign_flip_p"]
