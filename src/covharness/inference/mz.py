"""Pooled-vech Mincer-Zarnowitz calibration of covariance forecasts.

The primary representation uses one common intercept and one common slope
on every unique covariance entry i <= j,

    S_ij,t = alpha + beta H_ij,t + e_ij,t,

with null (alpha, beta) = (0, 1). This is the Patton-Sheppard
common-coefficient pooled-vech regression. Stacked unique entries are not
treated as independent. Daily scores sum pair-level contributions within
each date, and the sandwich uses Bartlett / Newey-West HAC on that daily
score sequence.

State-augmented MZ adds origin-measurable market-state regressors with
common gamma coefficients. That pooled state restriction is project-
specific and is not Patton-Sheppard's elementwise augmented MZ.

Approximate Patton-Sheppard equation-21 weights use

    s_ij,t = sqrt(H_ii,t * H_jj,t)

and divide the dependent variable and every regressor by the same scale.
The label is ``approximate_ps21``. It is not exact GLS.

Deterministic exact-fit cases keep Wald 0 or inf and do not return a
singular sandwich as a sampling covariance. Then ``covariance`` is None
and ``covariance_degenerate`` is True.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from covharness.inference.exceptions import RankDeficientMZDesignError
from covharness.inference.gw import (
    BonferroniFamilyResult,
    bonferroni_pair,
    market_state_augmenting_instruments,
)
from covharness.inference.hac import (
    KERNEL_BARTLETT,
    bartlett_weights,
    resolve_hac_lag,
)
from covharness.losses.contracts import (
    InvalidCovarianceMatrixError,
    as_square_finite,
    require_matching_square,
    require_symmetric,
)

MZ_ALPHA = 0.05
WEIGHTING_NONE = "none"
WEIGHTING_APPROXIMATE_PS21 = "approximate_ps21"
REPRESENTATION_POOLED_VECH = "pooled_vech"
SUBSET_ALL = "all"
SUBSET_DIAGONAL = "diagonal"
SUBSET_OFF_DIAGONAL = "off_diagonal"
INFERENCE_REGULAR = "regular"
INFERENCE_DETERMINISTIC_NULL = "deterministic_null"
INFERENCE_DETERMINISTIC_ALTERNATIVE = "deterministic_alternative"


@dataclass(frozen=True)
class MZSubsetResult:
    """Pooled coefficients on a subset of unique covariance entries."""

    subset: str
    n_entries: int
    alpha: float
    beta: float
    wald_statistic: float | None
    wald_df: int | None
    p_value: float | None


@dataclass(frozen=True)
class MZResult:
    """Pooled-vech Mincer-Zarnowitz calibration of one forecast.

    ``covariance`` is the daily-score HAC sandwich when inference is regular.
    Deterministic exact-fit cases set ``covariance`` to None rather than
    returning a zero or numerically singular matrix.
    """

    n_observations: int
    n_assets: int
    n_unique_entries: int
    alpha: float
    beta: float
    gamma: NDArray[np.floating]
    coefficients: NDArray[np.floating]
    covariance: NDArray[np.floating] | None
    covariance_degenerate: bool
    inference_case: str
    wald_statistic: float
    wald_df: int
    p_value: float
    alpha_level: float
    hac_lags: int
    hac_kernel: str
    weighting_rule: str
    representation: str
    diagonal: MZSubsetResult
    off_diagonal: MZSubsetResult | None


def as_covariance_panel(
    matrices: ArrayLike, name: str
) -> NDArray[np.floating]:
    """Copy a finite ``(T, N, N)`` covariance panel. Inputs are not mutated."""
    array = np.array(matrices, dtype=float, copy=True)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise InvalidCovarianceMatrixError(
            f"{name} must have shape (T, N, N); got {array.shape}"
        )
    if array.shape[0] < 3:
        raise InvalidCovarianceMatrixError(
            f"{name} requires at least three evaluation dates"
        )
    if array.shape[1] < 1:
        raise InvalidCovarianceMatrixError(f"{name} requires at least one asset")
    if not np.isfinite(array).all():
        raise InvalidCovarianceMatrixError(
            f"{name} must be finite (NaN and inf are rejected)"
        )
    # Check each date. Do not repair.
    for time in range(array.shape[0]):
        matrix = as_square_finite(array[time], f"{name}[{time}]")
        require_symmetric(matrix, f"{name}[{time}]")
        array[time] = matrix
    return array


def unique_entry_indices(n_assets: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return ``i <= j`` indices for the unique covariance entries."""
    return np.triu_indices(n_assets)


def _ps21_scales(
    forecast: NDArray[np.floating], row_index: NDArray[np.intp], col_index: NDArray[np.intp]
) -> NDArray[np.floating]:
    """Return ``s_ij,t = sqrt(H_ii,t * H_jj,t)`` for the requested pairs."""
    diag = np.diagonal(forecast, axis1=1, axis2=2)
    if np.any(diag <= 0.0) or (not np.isfinite(diag).all()):
        raise InvalidCovarianceMatrixError(
            "approximate PS21 scales require finite strictly positive forecast "
            "diagonals. No epsilon floor was applied."
        )
    # Scale by sqrt of the two forecast variances.
    return np.sqrt(diag[:, row_index] * diag[:, col_index])


def _design_and_response(
    proxy: NDArray[np.floating],
    forecast: NDArray[np.floating],
    row_index: NDArray[np.intp],
    col_index: NDArray[np.intp],
    instruments: NDArray[np.floating] | None,
    weighting_rule: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return daily design cubes ``X[t, p, k]`` and responses ``y[t, p]``."""
    response = proxy[:, row_index, col_index]
    forecast_entries = forecast[:, row_index, col_index]
    n_obs, n_entries = response.shape
    n_state = 0 if instruments is None else instruments.shape[1]
    n_coef = 2 + n_state
    design = np.empty((n_obs, n_entries, n_coef), dtype=float)
    design[:, :, 0] = 1.0
    design[:, :, 1] = forecast_entries
    if instruments is not None:
        # Common state regressors are repeated across unique entries.
        design[:, :, 2:] = instruments[:, np.newaxis, :]

    if weighting_rule == WEIGHTING_NONE:
        return design, response
    if weighting_rule != WEIGHTING_APPROXIMATE_PS21:
        raise ValueError(
            f"weighting_rule must be {WEIGHTING_NONE!r} or "
            f"{WEIGHTING_APPROXIMATE_PS21!r}; got {weighting_rule!r}"
        )
    scales = _ps21_scales(forecast, row_index, col_index)
    # Divide y and every X column by the same scale.
    return design / scales[:, :, np.newaxis], response / scales


def hac_long_run_covariance(
    scores: NDArray[np.floating],
    *,
    maxlags: int | None = None,
) -> tuple[NDArray[np.floating], int]:
    """Bartlett / Newey-West long-run covariance of a daily score sequence.

    Autocovariances use the ``1/T`` divisor on the demeaned scores. The
    estimator is the matrix analogue of Block 3A HAC. It is not the SPA
    geometric kernel.
    """
    n_obs, n_coef = scores.shape
    if n_obs < 2:
        raise ValueError("HAC requires at least two daily scores")
    if n_coef < 1:
        raise ValueError("daily scores must have at least one coordinate")
    lag, _lag_rule = resolve_hac_lag(n_obs, maxlags)
    centered = scores - scores.mean(axis=0, keepdims=True)
    omega = (centered.T @ centered) / n_obs
    weights = bartlett_weights(lag)
    for lag_index in range(1, lag + 1):
        gamma = (centered[lag_index:].T @ centered[:-lag_index]) / n_obs
        omega = omega + weights[lag_index - 1] * (gamma + gamma.T)
    if not np.isfinite(omega).all():
        raise RankDeficientMZDesignError(
            "HAC long-run covariance of the daily MZ score is not finite. "
            "No repair was applied."
        )
    return omega, lag


def _fit_pooled_vech(
    proxy: NDArray[np.floating],
    forecast: NDArray[np.floating],
    row_index: NDArray[np.intp],
    col_index: NDArray[np.intp],
    instruments: NDArray[np.floating] | None,
    weighting_rule: str,
    maxlags: int | None,
    restriction: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating] | None,
    float,
    int,
    int,
    bool,
    str,
]:
    """Return coefficients, optional sandwich, Wald, df, lag, and fit case."""
    design, response = _design_and_response(
        proxy, forecast, row_index, col_index, instruments, weighting_rule
    )
    n_obs, n_entries, n_coef = design.shape
    # Accumulate X'X and X'y over dates.
    gram = np.zeros((n_coef, n_coef), dtype=float)
    xy = np.zeros(n_coef, dtype=float)
    for time in range(n_obs):
        x_t = design[time]
        y_t = response[time]
        gram = gram + x_t.T @ x_t
        xy = xy + x_t.T @ y_t
    gram_rank = int(np.linalg.matrix_rank(gram))
    if gram_rank < n_coef:
        raise RankDeficientMZDesignError(
            "the pooled Mincer-Zarnowitz design is rank deficient. "
            "No ridge or dropped column was applied."
        )
    coefficients = np.linalg.solve(gram, xy)
    # Sum unique-entry scores by date.
    scores = np.empty((n_obs, n_coef), dtype=float)
    for time in range(n_obs):
        residual = response[time] - design[time] @ coefficients
        scores[time] = design[time].T @ residual
    omega, lag = hac_long_run_covariance(scores, maxlags=maxlags)
    gap = coefficients - restriction
    scores_zero = bool(np.allclose(scores, 0.0, atol=1e-10, rtol=0.0))
    # Exact fit. Do not return a vanishing sandwich.
    if np.allclose(gap, 0.0, atol=1e-10, rtol=0.0):
        return (
            coefficients,
            None,
            0.0,
            n_coef,
            lag,
            True,
            INFERENCE_DETERMINISTIC_NULL,
        )
    if scores_zero:
        return (
            coefficients,
            None,
            float(np.inf),
            n_coef,
            lag,
            True,
            INFERENCE_DETERMINISTIC_ALTERNATIVE,
        )
    covariance = np.linalg.solve(gram, np.linalg.solve(gram, n_obs * omega).T).T
    cov_rank = int(np.linalg.matrix_rank(covariance))
    if cov_rank < n_coef:
        return (
            coefficients,
            None,
            float(np.inf),
            n_coef,
            lag,
            True,
            INFERENCE_DETERMINISTIC_ALTERNATIVE,
        )
    mid = np.linalg.solve(covariance, gap)
    statistic = float(gap @ mid)
    return (
        coefficients,
        covariance,
        statistic,
        n_coef,
        lag,
        False,
        INFERENCE_REGULAR,
    )


def _subset_result(
    proxy: NDArray[np.floating],
    forecast: NDArray[np.floating],
    row_index: NDArray[np.intp],
    col_index: NDArray[np.intp],
    weighting_rule: str,
    maxlags: int | None,
    subset: str,
) -> MZSubsetResult:
    """Descriptive pooled intercept and slope on a pair subset."""
    restriction = np.array([0.0, 1.0], dtype=float)
    try:
        coefficients, _cov, statistic, df, _lag, _deg, _case = _fit_pooled_vech(
            proxy,
            forecast,
            row_index,
            col_index,
            None,
            weighting_rule,
            maxlags,
            restriction,
        )
    except RankDeficientMZDesignError:
        return MZSubsetResult(
            subset=subset,
            n_entries=int(row_index.size),
            alpha=float("nan"),
            beta=float("nan"),
            wald_statistic=None,
            wald_df=None,
            p_value=None,
        )
    return MZSubsetResult(
        subset=subset,
        n_entries=int(row_index.size),
        alpha=float(coefficients[0]),
        beta=float(coefficients[1]),
        wald_statistic=statistic,
        wald_df=df,
        p_value=float(chi2.sf(statistic, df=df)),
    )


def mincer_zarnowitz_pooled_vech(
    proxy: ArrayLike,
    forecast: ArrayLike,
    *,
    origin_covariances: ArrayLike | None = None,
    instruments: ArrayLike | None = None,
    weighting: str = WEIGHTING_NONE,
    maxlags: int | None = None,
    alpha: float = MZ_ALPHA,
) -> MZResult:
    """Pooled-vech MZ, optionally state-augmented or approximately weighted.

    ``proxy`` is the target-day realized covariance. ``origin_covariances``,
    when supplied, are origin-day matrices used only to build market-state
    augmenting instruments. Those instruments are not the target-day proxy.
    """
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if weighting not in (WEIGHTING_NONE, WEIGHTING_APPROXIMATE_PS21):
        raise ValueError(
            f"weighting must be {WEIGHTING_NONE!r} or "
            f"{WEIGHTING_APPROXIMATE_PS21!r}; got {weighting!r}"
        )
    if origin_covariances is not None and instruments is not None:
        raise ValueError(
            "supply origin_covariances or instruments, not both"
        )

    proxy_panel = as_covariance_panel(proxy, "proxy S")
    forecast_panel = as_covariance_panel(forecast, "forecast H")
    if proxy_panel.shape != forecast_panel.shape:
        require_matching_square(proxy_panel[0], forecast_panel[0])
        raise InvalidCovarianceMatrixError(
            "proxy S and forecast H panels must have the same shape; "
            f"got {proxy_panel.shape} and {forecast_panel.shape}"
        )
    n_obs, n_assets, _ = proxy_panel.shape
    row_all, col_all = unique_entry_indices(n_assets)

    state: NDArray[np.floating] | None
    if origin_covariances is not None:
        origin_panel = as_covariance_panel(origin_covariances, "origin S")
        if origin_panel.shape[0] != n_obs:
            raise ValueError(
                "origin_covariances and the target proxy must have the same "
                f"number of dates; got {origin_panel.shape[0]} and {n_obs}"
            )
        if origin_panel.shape[1] != n_assets:
            raise ValueError(
                "origin_covariances and the target proxy must have the same "
                "cross-section"
            )
        state = market_state_augmenting_instruments(origin_panel)
    elif instruments is not None:
        state_array = np.array(instruments, dtype=float, copy=True)
        if state_array.ndim != 2 or state_array.shape[0] != n_obs:
            raise ValueError(
                "instruments must have shape (T, q) aligned with the proxy"
            )
        if state_array.shape[1] < 1:
            raise ValueError("instruments must have at least one column")
        if not np.isfinite(state_array).all():
            raise ValueError("instruments must be finite")
        state = state_array
    else:
        state = None

    n_state = 0 if state is None else state.shape[1]
    restriction = np.zeros(2 + n_state, dtype=float)
    restriction[1] = 1.0
    coefficients, covariance, statistic, df, lag, degenerate, inference_case = (
        _fit_pooled_vech(
            proxy_panel,
            forecast_panel,
            row_all,
            col_all,
            state,
            weighting,
            maxlags,
            restriction,
        )
    )
    gamma = (
        np.zeros(0, dtype=float)
        if state is None
        else np.asarray(coefficients[2:], dtype=float)
    )
    diagonal_rows, diagonal_cols = np.diag_indices(n_assets)
    diagonal = _subset_result(
        proxy_panel,
        forecast_panel,
        diagonal_rows,
        diagonal_cols,
        weighting,
        maxlags,
        SUBSET_DIAGONAL,
    )
    off_diagonal: MZSubsetResult | None
    if n_assets >= 2:
        off_rows, off_cols = np.triu_indices(n_assets, k=1)
        off_diagonal = _subset_result(
            proxy_panel,
            forecast_panel,
            off_rows,
            off_cols,
            weighting,
            maxlags,
            SUBSET_OFF_DIAGONAL,
        )
    else:
        off_diagonal = None

    return MZResult(
        n_observations=n_obs,
        n_assets=n_assets,
        n_unique_entries=int(row_all.size),
        alpha=float(coefficients[0]),
        beta=float(coefficients[1]),
        gamma=gamma,
        coefficients=coefficients,
        covariance=covariance,
        covariance_degenerate=degenerate,
        inference_case=inference_case,
        wald_statistic=statistic,
        wald_df=df,
        p_value=float(chi2.sf(statistic, df=df)),
        alpha_level=float(alpha),
        hac_lags=lag,
        hac_kernel=KERNEL_BARTLETT,
        weighting_rule=weighting,
        representation=REPRESENTATION_POOLED_VECH,
        diagonal=diagonal,
        off_diagonal=off_diagonal,
    )


def mincer_zarnowitz_two_finalists(
    result_a: MZResult,
    result_b: MZResult,
    *,
    family_alpha: float = MZ_ALPHA,
    label_a: str = "finalist_a",
    label_b: str = "finalist_b",
) -> BonferroniFamilyResult:
    """Bonferroni family of two finalist MZ tests of the same specification."""
    return bonferroni_pair(
        result_a.p_value,
        result_b.p_value,
        family_alpha=family_alpha,
        label_a=label_a,
        label_b=label_b,
    )
