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
    lag_rule_name: str | None = None,
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
    if lag_rule_name is not None:
        lag_rule = lag_rule_name
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


# Kiefer and Vogelsang (2005), Table I, Bartlett kernel, right-tail critical
# values of t_b. Columns are b = 0.02, 0.04, ..., 1.0. The b = 1 column is
# the analytical Abadir and Paruolo value. Linear interpolation in b is the
# paper's stated rule for intermediate bandwidth fractions.
_FIXED_B_GRID = np.linspace(0.02, 1.0, 50)
_FIXED_B_TAILS = (0.90, 0.95, 0.975, 0.99)
_FIXED_B_TABLE = np.array(
    [
        [
            1.323, 1.343, 1.368, 1.390, 1.414, 1.442, 1.469, 1.498, 1.529, 1.563,
            1.587, 1.615, 1.644, 1.674, 1.709, 1.737, 1.767, 1.796, 1.830, 1.865,
            1.901, 1.931, 1.963, 1.994, 2.022, 2.053, 2.086, 2.117, 2.149, 2.183,
            2.211, 2.242, 2.272, 2.297, 2.325, 2.359, 2.389, 2.418, 2.450, 2.477,
            2.506, 2.533, 2.560, 2.591, 2.618, 2.649, 2.678, 2.706, 2.733, 2.740,
        ],
        [
            1.690, 1.731, 1.772, 1.813, 1.861, 1.902, 1.944, 1.988, 2.030, 2.081,
            2.124, 2.179, 2.222, 2.274, 2.324, 2.367, 2.412, 2.459, 2.505, 2.556,
            2.601, 2.651, 2.696, 2.739, 2.781, 2.828, 2.872, 2.913, 2.956, 3.007,
            3.048, 3.082, 3.124, 3.162, 3.198, 3.245, 3.291, 3.330, 3.367, 3.408,
            3.444, 3.494, 3.537, 3.579, 3.616, 3.654, 3.692, 3.727, 3.764, 3.764,
        ],
        [
            2.018, 2.072, 2.125, 2.179, 2.235, 2.296, 2.355, 2.417, 2.481, 2.553,
            2.617, 2.678, 2.736, 2.805, 2.878, 2.930, 2.999, 3.067, 3.129, 3.192,
            3.253, 3.318, 3.382, 3.447, 3.514, 3.567, 3.636, 3.684, 3.740, 3.783,
            3.834, 3.880, 3.934, 3.995, 4.054, 4.101, 4.153, 4.208, 4.260, 4.312,
            4.367, 4.417, 4.470, 4.524, 4.568, 4.617, 4.664, 4.713, 4.764, 4.771,
        ],
        [
            2.377, 2.459, 2.537, 2.627, 2.709, 2.792, 2.882, 2.961, 3.051, 3.140,
            3.229, 3.315, 3.385, 3.476, 3.580, 3.707, 3.791, 3.858, 3.942, 4.038,
            4.111, 4.212, 4.306, 4.399, 4.480, 4.567, 4.645, 4.711, 4.762, 4.831,
            4.912, 4.981, 5.041, 5.124, 5.190, 5.279, 5.333, 5.376, 5.445, 5.493,
            5.554, 5.609, 5.672, 5.724, 5.782, 5.868, 5.933, 5.998, 6.058, 6.090,
        ],
    ],
    dtype=float,
)


def llsw_sqrt_lag(n_observations: int) -> int:
    """Return ``floor(1.3 sqrt(T))``, the LLSW 2018 Bartlett lag."""
    if n_observations < 2:
        raise ValueError("HAC requires at least two observations")
    lag = int(np.floor(1.3 * np.sqrt(n_observations)))
    return max(0, min(lag, n_observations - 1))


def andrews_1991_bartlett_lag(values: ArrayLike) -> int:
    """Return the Andrews (1991) AR(1) plug-in lag for the Bartlett kernel.

    The plug-in uses the Yule-Walker correlation and
    ``1.1447 (alpha T)^{1/3}`` with ``alpha = 4 rho^2 / (1-rho^2)^2``.
    A sample correlation outside (-1, 1) is undefined and is not clipped.
    """
    series = as_1d_finite(values, "values")
    n_obs = int(series.shape[0])
    if n_obs < 3:
        raise ValueError("the Andrews lag requires at least three observations")
    centered = series - float(series.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise DegenerateLossDifferentialError("Andrews lag is undefined for a constant series")
    rho = float(np.dot(centered[1:], centered[:-1]) / denominator)
    if abs(rho) >= 1.0:
        raise DegenerateLossDifferentialError(
            f"Andrews AR(1) correlation is {rho}, outside (-1, 1). The lag is not clipped."
        )
    alpha = (4.0 * rho * rho) / (1.0 - rho * rho) ** 2
    raw = 1.1447 * (alpha * n_obs) ** (1.0 / 3.0)
    lag = int(np.floor(raw))
    return max(0, min(lag, n_obs - 1))


def bartlett_fixed_b_critical(bandwidth_fraction: float, tail: float = 0.975) -> float:
    """Interpolate Kiefer and Vogelsang (2005) Table I at ``b = (L+1)/T``.

    ``tail`` is the right-tail probability mass retained, so 0.975 is the
    two-sided 5 percent critical value. Values of ``b`` outside the
    published grid are held at the nearest tabulated endpoint.
    """
    if tail not in _FIXED_B_TAILS:
        raise ValueError(f"tail must be one of {_FIXED_B_TAILS}")
    row = _FIXED_B_TABLE[_FIXED_B_TAILS.index(tail)]
    fraction = float(bandwidth_fraction)
    if not np.isfinite(fraction):
        raise ValueError("bandwidth fraction must be finite")
    return float(np.interp(fraction, _FIXED_B_GRID, row))


def bartlett_fixed_b_pvalue(statistic: float, bandwidth_fraction: float) -> float:
    """Two-sided p-value by interpolating Table I at the supplied ``b``.

    The probability axis uses the published 90, 95, 97.5, and 99 percent
    points. Outside that range the p-value is censored at 0.20 or 0.02.
    The 5 percent decision uses ``bartlett_fixed_b_critical`` directly.
    """
    quantiles = np.array(
        [bartlett_fixed_b_critical(bandwidth_fraction, tail) for tail in _FIXED_B_TAILS],
        dtype=float,
    )
    upper_tails = np.array([0.10, 0.05, 0.025, 0.01], dtype=float)
    magnitude = abs(float(statistic))
    if magnitude <= quantiles[0]:
        one_tail = 0.10
    elif magnitude >= quantiles[-1]:
        one_tail = 0.01
    else:
        one_tail = float(np.interp(magnitude, quantiles, upper_tails))
    return float(min(1.0, 2.0 * one_tail))


def ewc_nu(n_observations: int) -> int:
    """Return ``floor(0.4 T^{2/3})``, at least 1."""
    if n_observations < 2:
        raise ValueError("EWC requires at least two observations")
    return max(1, int(np.floor(0.4 * n_observations ** (2.0 / 3.0))))


def ewc_long_run_variance(values: ArrayLike, nu: int | None = None) -> tuple[float, int]:
    """Equal-weighted cosine long-run variance, LLSW 2018 equation (10).

    ``phi_j = sqrt(2/T) sum_t z_t cos(pi j (t-1/2)/T)`` on the demeaned
    series, and the estimator averages ``phi_j^2`` over ``j = 1, ..., nu``.
    """
    series = as_1d_finite(values, "values")
    n_obs = int(series.shape[0])
    if np.all(series == series[0]):
        raise DegenerateLossDifferentialError("EWC is undefined for a constant series")
    terms = ewc_nu(n_obs) if nu is None else int(nu)
    if terms < 1:
        raise ValueError("EWC nu must be at least 1")
    centered = series - float(series.mean())
    time = np.arange(1, n_obs + 1, dtype=float)
    scale = np.sqrt(2.0 / n_obs)
    total = 0.0
    for index in range(1, terms + 1):
        projection = scale * float(np.dot(centered, np.cos(np.pi * index * (time - 0.5) / n_obs)))
        total += projection * projection
    omega = total / terms
    if not np.isfinite(omega) or omega <= 0.0:
        raise DegenerateLossDifferentialError("EWC long-run variance is not strictly positive")
    return float(omega), terms
