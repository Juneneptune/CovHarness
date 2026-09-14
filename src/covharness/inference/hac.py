"""Bartlett / Newey-West HAC long-run variance of a scalar series.

The estimator is the intercept-only sandwich used by Diebold-Mariano tests
of a mean loss differential. It is not an IID standard error.

Let ``d_t`` be demeaned as ``u_t = d_t - mean(d)``. Sample autocovariances
use the ``1/T`` divisor

    gamma_j = T^{-1} sum_{t=j+1}^{T} u_t u_{t-j},  j = 0, ..., L.

Bartlett (Newey-West 1987) weights are

    k(j) = 1 - j / (L + 1),  j = 1, ..., L.

The long-run variance is

    omega = gamma_0 + 2 sum_{j=1}^{L} k(j) gamma_j.

The default lag is the Newey-West (1994) Bartlett rule

    L = floor(4 (T/100)^{2/9}).

A user-supplied lag replaces that rule. ``L = 0`` returns ``gamma_0``, the
heteroskedasticity-only variance. If every supplied observation is exactly
equal, HAC is undefined and is rejected before demeaning. A non-finite
long-run variance is also rejected. No ridge or jitter is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.inference.differentials import as_1d_finite
from covharness.inference.exceptions import DegenerateLossDifferentialError

KERNEL_BARTLETT = "bartlett"
LAG_RULE_NEWEY_WEST_1994 = "newey_west_1994"
LAG_RULE_USER = "user"


@dataclass(frozen=True)
class HACResult:
    """Auditable HAC long-run variance of a scalar mean."""

    n_observations: int
    mean: float
    sample_variance: float
    long_run_variance: float
    standard_error_of_mean: float
    kernel: str
    maxlags: int
    lag_rule: str


def newey_west_1994_lags(n_observations: int) -> int:
    """Return ``floor(4 (T/100)^{2/9})``, truncated to a feasible lag."""
    if n_observations < 2:
        raise ValueError("HAC requires at least two observations")
    lag = int(np.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0)))
    return max(0, min(lag, n_observations - 1))


def resolve_hac_lag(n_observations: int, maxlags: int | None) -> tuple[int, str]:
    """Return the applied lag and the rule name used to obtain it."""
    if maxlags is None:
        return newey_west_1994_lags(n_observations), LAG_RULE_NEWEY_WEST_1994
    if maxlags < 0:
        raise ValueError("maxlags cannot be negative")
    return min(int(maxlags), n_observations - 1), LAG_RULE_USER


def bartlett_weights(maxlags: int) -> NDArray[np.floating]:
    """Return Newey-West Bartlett weights ``k(j)`` for ``j = 1, ..., L``."""
    if maxlags < 0:
        raise ValueError("maxlags cannot be negative")
    if maxlags == 0:
        return np.array([], dtype=float)
    lags = np.arange(1, maxlags + 1, dtype=float)
    return 1.0 - lags / (maxlags + 1.0)


def sample_autocovariances(
    centered: NDArray[np.floating], maxlags: int
) -> NDArray[np.floating]:
    """Return ``gamma_0, ..., gamma_L`` with the ``1/T`` divisor."""
    n_obs = centered.shape[0]
    gammas = np.empty(maxlags + 1, dtype=float)
    # gamma_0 is the 1/T second moment of the centered series.
    gammas[0] = float(np.dot(centered, centered) / n_obs)
    for lag in range(1, maxlags + 1):
        gammas[lag] = float(np.dot(centered[lag:], centered[:-lag]) / n_obs)
    return gammas


def hac_long_run_variance(
    values: ArrayLike,
    *,
    maxlags: int | None = None,
) -> HACResult:
    """Estimate the Bartlett HAC long-run variance of ``values``.

    The standard error of the sample mean is ``sqrt(omega / T)``.
    """
    series = as_1d_finite(values, "values")
    n_obs = int(series.shape[0])
    if n_obs < 2:
        raise ValueError("HAC requires at least two observations")

    # Exact equality of the supplied observations, before any demeaning.
    if np.all(series == series[0]):
        raise DegenerateLossDifferentialError(
            "all loss-differential observations are equal. HAC is undefined. "
            "No jitter was added."
        )

    # Resolve the lag from the centralized rule or an explicit override.
    lag, lag_rule = resolve_hac_lag(n_obs, maxlags)

    mean = float(series.mean())
    centered = series - mean
    gammas = sample_autocovariances(centered, lag)
    weights = bartlett_weights(lag)
    long_run = float(gammas[0] + 2.0 * np.dot(weights, gammas[1:]))
    # Treat a numerically tiny negative omega as zero; do not invent a finite SE.
    if long_run < 0.0 and np.isclose(long_run, 0.0, atol=1e-15, rtol=0.0):
        long_run = 0.0
    if long_run < 0.0:
        raise DegenerateLossDifferentialError(
            f"HAC long-run variance is negative ({long_run}). "
            "This is not repaired."
        )
    if long_run == 0.0:
        raise DegenerateLossDifferentialError(
            "HAC long-run variance is zero. The mean standard error is "
            "undefined. No jitter was added."
        )
    if not np.isfinite(long_run):
        raise DegenerateLossDifferentialError(
            "HAC long-run variance is not finite. The mean standard error is "
            "undefined. No repair was applied."
        )
    standard_error = float(np.sqrt(long_run / n_obs))
    return HACResult(
        n_observations=n_obs,
        mean=mean,
        sample_variance=float(gammas[0]),
        long_run_variance=long_run,
        standard_error_of_mean=standard_error,
        kernel=KERNEL_BARTLETT,
        maxlags=lag,
        lag_rule=lag_rule,
    )
