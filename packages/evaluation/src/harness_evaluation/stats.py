"""Confidence intervals over questions (percentile bootstrap, SciPy).

With about 30 questions the intervals are wide; that is the honest answer, and
the reason to report them. The unit of resampling is the question: repetitions
of one question are averaged first, because they are not independent samples.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from scipy.stats import bootstrap


def mean_ci(values: Sequence[float], *, confidence: float = 0.95, resamples: int = 2000,
            seed: int = 0) -> dict[str, float | int | None]:
    """Mean with a percentile-bootstrap interval; the interval collapses when it cannot vary."""
    data = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not data:
        return {"mean": None, "low": None, "high": None, "n": 0}
    centre = sum(data) / len(data)
    if len(data) < 2 or len(set(data)) == 1:
        return {"mean": centre, "low": centre, "high": centre, "n": len(data)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bootstrap((np.asarray(data),), np.mean, confidence_level=confidence, n_resamples=resamples,
                           method="percentile", rng=np.random.default_rng(seed))
    return {"mean": centre, "low": float(result.confidence_interval.low),
            "high": float(result.confidence_interval.high), "n": len(data)}


def paired_difference_ci(left: Sequence[float], right: Sequence[float], **options) -> dict[str, float | int | None]:
    """Mean of ``left - right`` over the same questions, with its interval."""
    if len(left) != len(right):
        raise ValueError("paired values need the same questions on both sides")
    return mean_ci([a - b for a, b in zip(left, right, strict=True)], **options)


__all__ = ["mean_ci", "paired_difference_ci"]
