"""Hansen (2005) test for Superior Predictive Ability.

The input is a loss matrix ``L[t, m]`` with lower loss better. Model columns
are frozen forecasting methods. Rows are evaluation dates of one channel.
SPA is not pooled across losses or covariance proxies.

For benchmark 0 and alternative k the Hansen differential is

    d[k, t] = L[t, 0] - L[t, k].

A positive value means alternative k beats the benchmark on date t. The null
is that no alternative beats the benchmark,

    H0: E[d_k] <= 0 for every k.

The studentized statistic is

    T_SPA = max(0, max_k sqrt(T) * mean(d_k) / omega_hat_k).

``omega_hat_k^2`` is the stationary-bootstrap population long-run variance,
not Block 3A Bartlett / Newey-West HAC. The same omega_hat is used for the
observed statistic and every bootstrap replicate.

p-values use Hansen's strict inequality ``mean(T* > T)``. The primary
p-value is the consistent recentering p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.inference.bootstrap import (
    BOOTSTRAP_METHOD,
    BOOTSTRAP_SEED,
    DEFAULT_N_BOOT,
    as_finite_loss_matrix,
    default_block_length,
    restart_probability,
    stationary_bootstrap_indices,
)
from covharness.inference.exceptions import DegenerateLossDifferentialError

SPA_ALPHA = 0.05
MIN_SPA_OBSERVATIONS = 3


@dataclass(frozen=True)
class SPAResult:
    """Hansen (2005) SPA test of a loss matrix against one benchmark."""

    n_observations: int
    n_models: int
    n_alternatives: int
    benchmark_index: int
    benchmark_label: str
    mean_differentials: NDArray[np.floating]
    long_run_variances: NDArray[np.floating]
    studentized_statistics: NDArray[np.floating]
    statistic: float
    p_value_lower: float
    p_value_consistent: float
    p_value_upper: float
    block_length: int
    restart_probability: float
    n_boot: int
    seed: int
    alpha: float
    bootstrap_method: str


def hansen_lil_threshold(n_observations: int) -> float:
    """Return ``-sqrt(2 log log T)``, the Hansen consistent-recentering cutoff."""
    if n_observations < MIN_SPA_OBSERVATIONS:
        raise ValueError(
            "SPA consistent recentering requires at least "
            f"{MIN_SPA_OBSERVATIONS} observations so that log log T is defined"
        )
    log_log = float(np.log(np.log(n_observations)))
    if log_log <= 0.0:
        raise ValueError("SPA consistent recentering requires log(log T) > 0")
    return -float(np.sqrt(2.0 * log_log))


def stationary_bootstrap_long_run_variance(
    differentials: NDArray[np.floating],
    restart_q: float,
) -> NDArray[np.floating]:
    """Return Hansen's geometric-kernel long-run variance for each column.

    ``differentials`` has shape ``(T, K)``. Autocovariances use the ``1/T``
    divisor on the demeaned series. The kernel is

        kappa(T, i) = ((T-i)/T) (1-q)^i + (i/T) (1-q)^{T-i}.
    """
    n_obs, n_alts = differentials.shape
    centered = differentials - differentials.mean(axis=0, keepdims=True)
    # gamma_0 is the 1/T second moment of each demeaned alternative.
    omega = np.sum(centered * centered, axis=0) / n_obs
    lags = np.arange(1, n_obs, dtype=float)
    one_minus_q = 1.0 - restart_q
    kappa = ((n_obs - lags) / n_obs) * (one_minus_q**lags) + (
        lags / n_obs
    ) * (one_minus_q ** (n_obs - lags))
    # Add the two-sided geometric kernel at every feasible lag.
    for lag in range(1, n_obs):
        gamma_lag = np.sum(centered[lag:] * centered[:-lag], axis=0) / n_obs
        omega = omega + 2.0 * kappa[lag - 1] * gamma_lag
    if np.any(omega < 0.0) or (not np.isfinite(omega).all()):
        raise DegenerateLossDifferentialError(
            "SPA long-run variance is not finite and positive. "
            "This is not repaired."
        )
    if np.any(omega == 0.0):
        raise DegenerateLossDifferentialError(
            "SPA long-run variance is zero. The studentized statistic is "
            "undefined. No jitter was added."
        )
    return np.asarray(omega, dtype=float)


def superior_predictive_ability(
    losses: ArrayLike,
    *,
    benchmark_index: int,
    benchmark_label: str | None = None,
    block_length: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = SPA_ALPHA,
) -> SPAResult:
    """Hansen SPA test of whether any column beats the supplied benchmark."""
    loss_matrix = as_finite_loss_matrix(losses, "losses")
    n_obs, n_models = loss_matrix.shape
    if n_obs < MIN_SPA_OBSERVATIONS:
        raise ValueError(
            "SPA requires at least "
            f"{MIN_SPA_OBSERVATIONS} evaluation dates; got {n_obs}"
        )
    if n_models < 2:
        raise ValueError("SPA requires a benchmark and at least one alternative")
    bench = int(benchmark_index)
    if bench < 0 or bench >= n_models:
        raise ValueError(
            f"benchmark_index must be in 0, ..., {n_models - 1}; got {benchmark_index}"
        )
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")

    # Form Hansen differentials d[k,t] = L_benchmark,t - L_k,t.
    alternative_index = [j for j in range(n_models) if j != bench]
    differentials = loss_matrix[:, bench : bench + 1] - loss_matrix[:, alternative_index]
    # Exact equality of an alternative differential is outside Assumption 1.
    for alt_pos in range(differentials.shape[1]):
        series = differentials[:, alt_pos]
        if np.all(series == series[0]):
            raise DegenerateLossDifferentialError(
                "all benchmark-versus-alternative observations are equal. "
                "SPA is undefined. No jitter was added."
            )

    length = default_block_length(n_obs) if block_length is None else int(block_length)
    restart_q = restart_probability(n_obs, length)
    omega = stationary_bootstrap_long_run_variance(differentials, restart_q)
    means = differentials.mean(axis=0)
    studentized = np.sqrt(n_obs) * means / np.sqrt(omega)
    statistic = float(max(0.0, float(np.max(studentized))))

    # Hansen g functions recentering the bootstrap about mu^l, mu^c, and mu^u.
    lil = hansen_lil_threshold(n_obs)
    g_lower = np.maximum(means, 0.0)
    g_consistent = means * (means >= lil * np.sqrt(omega / n_obs))
    g_upper = means.copy()

    indices = stationary_bootstrap_indices(
        n_obs, n_boot, block_length=length, seed=seed
    )
    star_means = differentials[indices].mean(axis=1)
    p_lower = _spa_pvalue(star_means, g_lower, omega, n_obs, statistic)
    p_consistent = _spa_pvalue(star_means, g_consistent, omega, n_obs, statistic)
    p_upper = _spa_pvalue(star_means, g_upper, omega, n_obs, statistic)
    label = str(bench) if benchmark_label is None else str(benchmark_label)
    return SPAResult(
        n_observations=n_obs,
        n_models=n_models,
        n_alternatives=n_models - 1,
        benchmark_index=bench,
        benchmark_label=label,
        mean_differentials=np.asarray(means, dtype=float),
        long_run_variances=np.asarray(omega, dtype=float),
        studentized_statistics=np.asarray(studentized, dtype=float),
        statistic=statistic,
        p_value_lower=p_lower,
        p_value_consistent=p_consistent,
        p_value_upper=p_upper,
        block_length=length,
        restart_probability=restart_q,
        n_boot=int(n_boot),
        seed=int(seed),
        alpha=float(alpha),
        bootstrap_method=BOOTSTRAP_METHOD,
    )


def _spa_pvalue(
    star_means: NDArray[np.floating],
    recentering: NDArray[np.floating],
    omega: NDArray[np.floating],
    n_obs: int,
    statistic: float,
) -> float:
    """Return Hansen's bootstrap p-value ``mean(T* > T)`` for one recentering."""
    z_means = star_means - recentering
    studentized_star = np.sqrt(n_obs) * z_means / np.sqrt(omega)
    t_star = np.maximum(0.0, np.max(studentized_star, axis=1))
    return float(np.mean(t_star > statistic))
