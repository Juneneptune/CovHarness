"""Descriptive HAR-style leverage regression.

The regression asks whether next-period log variance responds differently
to positive and negative origin-day returns. It does not rank models and
does not impose the equity leverage sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from covharness.inference.mz import hac_long_run_covariance

Z_CRITICAL = 1.96
N_COEF = 6
COEF_NAMES = ("intercept", "daily", "weekly", "monthly", "g_pos", "g_neg")
RESPONSE_LAG = 22


class LeverageDesignError(ValueError):
    """The leverage design is invalid and was not repaired."""


@dataclass(frozen=True)
class LeverageFit:
    """One OLS fit and the HAC interval for ``g_pos + g_neg``."""

    coefficients: NDArray[np.floating]
    gap: float
    gap_se: float
    gap_ci_low: float
    gap_ci_high: float
    hac_lag: int
    n_observations: int
    g_neg_negative: bool
    abs_g_neg_exceeds_g_pos: bool
    leverage_label: bool


def har_leverage_design(
    variance: NDArray[np.floating],
    daily_return: NDArray[np.floating],
    response_index: NDArray[np.intp],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build the A4 design at the supplied response indices.

    ``variance`` and ``daily_return`` have shape ``(T,)``. Each response
    index ``k`` uses only observations strictly before ``k``. Weekly is the
    log of the mean of four variances ``[k-5, k-1)``. Monthly is the log of
    the mean of seventeen variances ``[k-22, k-5)``. The signed return is
    the origin observation ``k-1``.
    """
    level = np.asarray(variance, dtype=float)
    returns = np.asarray(daily_return, dtype=float)
    index = np.asarray(response_index, dtype=np.intp)
    if level.ndim != 1 or returns.shape != level.shape:
        raise LeverageDesignError("variance and return must be aligned 1-d series")
    if index.ndim != 1 or index.size < 2:
        raise LeverageDesignError("at least two response dates are required")
    if np.any(index < RESPONSE_LAG) or np.any(index >= level.shape[0]):
        raise LeverageDesignError("a response index falls outside the lagged sample")
    design = np.empty((index.size, N_COEF), dtype=float)
    response = np.empty(index.size, dtype=float)
    # One row per target. Features stop at the origin.
    for row, k in enumerate(index.tolist()):
        window = level[k - RESPONSE_LAG : k]
        origin_return = float(returns[k - 1])
        if not np.isfinite(window).all() or np.any(window <= 0.0):
            raise LeverageDesignError("lagged variance must be finite and strictly positive")
        if not np.isfinite(level[k]) or level[k] <= 0.0 or not np.isfinite(origin_return):
            raise LeverageDesignError("response variance and origin return must be finite")
        daily = float(level[k - 1])
        weekly = float(level[k - 5 : k - 1].mean())
        monthly = float(level[k - 22 : k - 5].mean())
        response[row] = float(np.log(level[k]))
        design[row, 0] = 1.0
        design[row, 1] = float(np.log(daily))
        design[row, 2] = float(np.log(weekly))
        design[row, 3] = float(np.log(monthly))
        design[row, 4] = max(origin_return, 0.0)
        design[row, 5] = min(origin_return, 0.0)
    return design, response


def fit_leverage_ols(
    design: NDArray[np.floating],
    response: NDArray[np.floating],
) -> LeverageFit:
    """OLS with a Bartlett interval for ``g_pos + g_neg``.

    The daily score is ``x_t e_t``. The sandwich matches the pooled
    Mincer-Zarnowitz scaling ``(X'X)^{-1} (T Omega) (X'X)^{-1}``.
    """
    x = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.ndim != 2 or x.shape[1] != N_COEF or y.shape != (x.shape[0],):
        raise LeverageDesignError("design must be (T, 6) and response (T,)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise LeverageDesignError("design and response must be finite")
    gram = x.T @ x
    if int(np.linalg.matrix_rank(gram)) < N_COEF:
        raise LeverageDesignError("leverage design is rank deficient. No ridge was applied.")
    coefficients = np.linalg.solve(gram, x.T @ y)
    residual = y - x @ coefficients
    scores = x * residual[:, None]
    omega, lag = hac_long_run_covariance(scores)
    n_obs = int(y.shape[0])
    # Same sandwich scaling as the pooled MZ estimator.
    covariance = np.linalg.solve(gram, np.linalg.solve(gram, n_obs * omega).T).T
    contrast = np.zeros(N_COEF, dtype=float)
    contrast[4] = 1.0
    contrast[5] = 1.0
    gap_variance = float(contrast @ covariance @ contrast)
    if not np.isfinite(gap_variance) or gap_variance < 0.0:
        raise LeverageDesignError("HAC variance of the leverage gap is not finite and nonnegative")
    gap = float(coefficients[4] + coefficients[5])
    gap_se = float(np.sqrt(gap_variance))
    g_pos = float(coefficients[4])
    g_neg = float(coefficients[5])
    g_neg_negative = bool(g_neg < 0.0)
    larger = bool(abs(g_neg) > g_pos)
    return LeverageFit(
        coefficients=np.asarray(coefficients, dtype=float),
        gap=gap,
        gap_se=gap_se,
        gap_ci_low=gap - Z_CRITICAL * gap_se,
        gap_ci_high=gap + Z_CRITICAL * gap_se,
        hac_lag=int(lag),
        n_observations=n_obs,
        g_neg_negative=g_neg_negative,
        abs_g_neg_exceeds_g_pos=larger,
        leverage_label=bool(g_neg_negative and larger),
    )


def fit_pooled_leverage_ols(
    designs: NDArray[np.floating],
    responses: NDArray[np.floating],
) -> LeverageFit:
    """Common-coefficient OLS. Asset scores are summed within each day.

    ``designs`` has shape ``(N, T, 6)`` and ``responses`` has shape ``(N, T)``.
    """
    x = np.asarray(designs, dtype=float)
    y = np.asarray(responses, dtype=float)
    if x.ndim != 3 or x.shape[2] != N_COEF or y.shape != x.shape[:2]:
        raise LeverageDesignError("pooled inputs must have shapes (N, T, 6) and (N, T)")
    n_assets, n_obs, _n_coef = x.shape
    gram = np.zeros((N_COEF, N_COEF), dtype=float)
    xy = np.zeros(N_COEF, dtype=float)
    # Accumulate the common Gram over assets. Do not treat asset-days as independent.
    for asset in range(n_assets):
        gram = gram + x[asset].T @ x[asset]
        xy = xy + x[asset].T @ y[asset]
    if int(np.linalg.matrix_rank(gram)) < N_COEF:
        raise LeverageDesignError("pooled leverage design is rank deficient. No ridge was applied.")
    coefficients = np.linalg.solve(gram, xy)
    scores = np.zeros((n_obs, N_COEF), dtype=float)
    for asset in range(n_assets):
        residual = y[asset] - x[asset] @ coefficients
        scores = scores + x[asset] * residual[:, None]
    omega, lag = hac_long_run_covariance(scores)
    covariance = np.linalg.solve(gram, np.linalg.solve(gram, n_obs * omega).T).T
    contrast = np.zeros(N_COEF, dtype=float)
    contrast[4] = 1.0
    contrast[5] = 1.0
    gap_variance = float(contrast @ covariance @ contrast)
    if not np.isfinite(gap_variance) or gap_variance < 0.0:
        raise LeverageDesignError("HAC variance of the pooled leverage gap is not finite and nonnegative")
    gap = float(coefficients[4] + coefficients[5])
    gap_se = float(np.sqrt(gap_variance))
    g_pos = float(coefficients[4])
    g_neg = float(coefficients[5])
    g_neg_negative = bool(g_neg < 0.0)
    larger = bool(abs(g_neg) > g_pos)
    return LeverageFit(
        coefficients=np.asarray(coefficients, dtype=float),
        gap=gap,
        gap_se=gap_se,
        gap_ci_low=gap - Z_CRITICAL * gap_se,
        gap_ci_high=gap + Z_CRITICAL * gap_se,
        hac_lag=int(lag),
        n_observations=int(n_obs),
        g_neg_negative=g_neg_negative,
        abs_g_neg_exceeds_g_pos=larger,
        leverage_label=bool(g_neg_negative and larger),
    )
