"""Giacomini-Rossi (2010) out-of-sample Fluctuation test, Proposition 1.

The project differential is ``d_t = L_{A,t} - L_{B,t}``. A negative local
statistic means A is locally better. A positive local statistic means B is
locally better.

The centered rolling path is

    F_t = omega_hat^{-1} * m^{-1/2} * sum_{j=t-m/2}^{t+m/2-1} d_j

with even ``m = 2 * floor(0.30 * P / 2)``. The global long-run variance uses
uncentered ``1/P`` autocovariances of the full series and the Block 3A
Bartlett / Newey-West lag convention. It is not re-estimated inside windows
and it does not demean. The one-time reversal test is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.inference.differentials import loss_differential
from covharness.inference.exceptions import DegenerateFluctuationVarianceError
from covharness.inference.hac import bartlett_weights, resolve_hac_lag, sample_autocovariances

FLUCTUATION_MU = 0.30
FLUCTUATION_ALPHA = 0.05
FLUCTUATION_CRITICAL_VALUE = 3.012
FLUCTUATION_SIGN = "d_t = L_A,t - L_B,t; negative F_t means A is locally better"


@dataclass(frozen=True)
class FluctuationResult:
    """Two-sided Giacomini-Rossi fluctuation test of a loss differential."""

    n_observations: int
    window_length: int
    target_mu: float
    realized_mu: float
    center_indices: NDArray[np.intp]
    path: NDArray[np.floating]
    long_run_variance: float
    omega_hat: float
    max_abs_statistic: float
    critical_value: float
    alpha: float
    reject: bool
    sign_convention: str
    hac_lags: int
    hac_kernel: str


def even_centered_window_length(n_observations: int, mu: float = FLUCTUATION_MU) -> int:
    """Return ``m = 2 * floor(mu * P / 2)``, which is always even."""
    if n_observations < 3:
        raise ValueError("the fluctuation test requires at least three observations")
    if mu <= 0.0 or mu >= 1.0:
        raise ValueError("mu must lie in (0, 1)")
    window = 2 * int(np.floor(mu * n_observations / 2.0))
    if window < 2:
        raise ValueError(
            "the fluctuation window length must be at least 2. "
            f"P={n_observations} and mu={mu} produced m={window}."
        )
    if n_observations <= window:
        raise ValueError(
            "the fluctuation test requires P > m; "
            f"got P={n_observations} and m={window}"
        )
    return window


def uncentered_bartlett_long_run_variance(
    differential: NDArray[np.floating],
    *,
    maxlags: int | None = None,
) -> tuple[float, int]:
    """Giacomini-Rossi global LRV of ``d_t`` without subtracting ``dbar``.

    ``gamma_j^0 = P^{-1} sum_{t=j+1}^P d_t d_{t-j}``. Bandwidth and Bartlett
    weights reuse the Block 3A conventions. The demeaned Block 3A HAC
    estimator is not called.
    """
    n_obs = int(differential.shape[0])
    lag, _lag_rule = resolve_hac_lag(n_obs, maxlags)
    # Uncentered 1/T products. Do not demean.
    gammas = sample_autocovariances(differential, lag)
    weights = bartlett_weights(lag)
    long_run = float(gammas[0] + 2.0 * np.dot(weights, gammas[1:]))
    if long_run < 0.0 and np.isclose(long_run, 0.0, atol=1e-15, rtol=0.0):
        long_run = 0.0
    if long_run < 0.0:
        raise DegenerateFluctuationVarianceError(
            f"Giacomini-Rossi long-run variance is negative ({long_run}). "
            "This is not repaired."
        )
    if long_run == 0.0:
        raise DegenerateFluctuationVarianceError(
            "Giacomini-Rossi long-run variance is zero. The fluctuation "
            "statistic is undefined. No jitter was added."
        )
    if not np.isfinite(long_run):
        raise DegenerateFluctuationVarianceError(
            "Giacomini-Rossi long-run variance is not finite. No repair "
            "was applied."
        )
    return long_run, lag


def giacomini_rossi_fluctuation(
    loss_a: ArrayLike,
    loss_b: ArrayLike,
    *,
    mu: float = FLUCTUATION_MU,
    alpha: float = FLUCTUATION_ALPHA,
    maxlags: int | None = None,
) -> FluctuationResult:
    """Two-sided Giacomini-Rossi fluctuation test on paired loss series."""
    if alpha != FLUCTUATION_ALPHA:
        raise ValueError(
            "the frozen two-sided critical value is for alpha="
            f"{FLUCTUATION_ALPHA}; got {alpha}"
        )
    if mu != FLUCTUATION_MU:
        raise ValueError(
            f"the frozen fluctuation window fraction is mu={FLUCTUATION_MU}; "
            f"got {mu}"
        )
    differential = loss_differential(loss_a, loss_b)
    n_obs = int(differential.shape[0])
    window = even_centered_window_length(n_obs, mu=mu)
    half = window // 2
    # Drop the first and last m/2 centers.
    center_indices = np.arange(half, n_obs - half + 1, dtype=np.intp)
    long_run, lag = uncentered_bartlett_long_run_variance(
        differential, maxlags=maxlags
    )
    omega_hat = float(np.sqrt(long_run))
    path = np.empty(center_indices.shape[0], dtype=float)
    scale = 1.0 / (omega_hat * np.sqrt(window))
    for index, center in enumerate(center_indices):
        # Inclusive window [center - m/2, center + m/2 - 1].
        local_sum = float(np.sum(differential[center - half : center + half]))
        path[index] = scale * local_sum
    max_abs = float(np.max(np.abs(path)))
    reject = bool(max_abs > FLUCTUATION_CRITICAL_VALUE)
    return FluctuationResult(
        n_observations=n_obs,
        window_length=window,
        target_mu=float(mu),
        realized_mu=float(window / n_obs),
        center_indices=center_indices,
        path=path,
        long_run_variance=long_run,
        omega_hat=omega_hat,
        max_abs_statistic=max_abs,
        critical_value=FLUCTUATION_CRITICAL_VALUE,
        alpha=float(alpha),
        reject=reject,
        sign_convention=FLUCTUATION_SIGN,
        hac_lags=lag,
        hac_kernel="bartlett",
    )
