"""Descriptive sign summaries for a return series.

These quantities do not rank forecasts. Threshold probabilities are supplied
by the caller. This module does not choose them from a loss comparison.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binomtest, skew


def _finite_series(values: ArrayLike, name: str) -> np.ndarray:
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size < 2:
        raise ValueError(f"{name} requires at least two observations")
    if not np.isfinite(series).all():
        raise ValueError(f"{name} must be finite")
    return series


def bias_corrected_skewness(values: ArrayLike) -> float:
    """Bias-corrected Fisher-Pearson skewness of a finite series."""
    series = _finite_series(values, "values")
    return float(skew(series, bias=False))


def tail_sign_summary(values: ArrayLike, quantile: float) -> dict[str, float | int | None]:
    """Counts and mean absolute sizes beyond a fixed percentile of ``|r|``.

    The exceedance rule is strict. ``r > c`` and ``r < -c`` are the two
    sides, with ``c`` the linear quantile of ``|r|``. The binomial p-value
    tests equal up and down counts. It is descriptive. No multiplicity
    adjustment is applied.
    """
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError(f"quantile must lie in (0, 1); got {quantile}")
    series = _finite_series(values, "values")
    # Freeze the threshold from |r| before splitting the sign.
    threshold = float(np.quantile(np.abs(series), float(quantile), method="linear"))
    up = series[series > threshold]
    down = series[series < -threshold]
    n_up = int(up.size)
    n_down = int(down.size)
    n_total = n_up + n_down
    binomial_p: float | None = None
    if n_total > 0:
        binomial_p = float(binomtest(n_up, n_total, 0.5, alternative="two-sided").pvalue)
    return {
        "quantile": float(quantile),
        "threshold": threshold,
        "n_up": n_up,
        "n_down": n_down,
        "mean_abs_up": None if n_up == 0 else float(np.mean(np.abs(up))),
        "mean_abs_down": None if n_down == 0 else float(np.mean(np.abs(down))),
        "binomial_p": binomial_p,
    }
