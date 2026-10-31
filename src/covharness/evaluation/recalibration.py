"""Origin-time recalibration of a saved covariance forecast sequence.

R1 rescales a forecast by the mean whitened trace ratio on past
out-of-sample pairs. R2 shrinks that forecast toward the trailing proxy
mean and then applies the same closed-form scale. Neither map refits a
model. The headline forecast remains the raw matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.losses.qlike import reduced_qlike_loss, whitened_trace_ratio

MIN_RECALIBRATION_PAIRS = 21
MAX_RECALIBRATION_PAIRS = 250
LAMBDA_GRID: tuple[float, ...] = tuple(i / 20 for i in range(21))


@dataclass(frozen=True)
class RecalibrationResult:
    """Transformed SCREEN or synthetic path and the coefficients that made it.

    ``r1`` and ``r2`` have shape ``(T, N, N)``. Coefficient paths have shape
    ``(T,)``. An identity origin stores scale 1 and, for R2, lambda 1.
    ``hbar_failure_origins`` lists origins where the proxy mean was not
    strictly positive definite, so R2 used the identity there.
    """

    r1: NDArray[np.floating]
    r2: NDArray[np.floating]
    scale_r1: NDArray[np.floating]
    scale_r2: NDArray[np.floating]
    lambda_r2: NDArray[np.floating]
    identity_r1: NDArray[np.bool_]
    identity_r2: NDArray[np.bool_]
    hbar_failure_origins: tuple[str, ...]


def fitting_indices(
    origin_dates: Sequence[str],
    target_dates: Sequence[str],
    segments: Sequence[str],
    origin_index: int,
    *,
    max_pairs: int = MAX_RECALIBRATION_PAIRS,
) -> NDArray[np.int64]:
    """Trailing past pairs whose target is on or before this origin.

    A pair enters only when its target date is on or before the origin and
    its target segment equals the segment of the target being forecast.
    The current row is excluded by that date rule on a one-day-ahead
    schedule. The returned indices are chronological and at most
    ``max_pairs`` long.
    """
    origin = origin_dates[origin_index]
    segment = segments[origin_index]
    eligible = [
        index
        for index, target in enumerate(target_dates)
        if target <= origin and segments[index] == segment
    ]
    window = eligible[-max_pairs:]
    return np.asarray(window, dtype=np.int64)


def scalar_multiplier(
    forecasts: NDArray[np.floating],
    proxies: NDArray[np.floating],
) -> float:
    """Mean of tr(H inverse S) / N. This is the R1 closed form."""
    ratios = [
        whitened_trace_ratio(proxies[index], forecasts[index])
        for index in range(forecasts.shape[0])
    ]
    return float(np.mean(ratios))


def _mean_reduced_qlike(
    forecasts: NDArray[np.floating],
    proxies: NDArray[np.floating],
) -> float:
    losses = [
        reduced_qlike_loss(proxies[index], forecasts[index])
        for index in range(forecasts.shape[0])
    ]
    return float(np.mean(losses))


def _proxy_mean(proxies: NDArray[np.floating]) -> NDArray[np.floating]:
    # Symmetrize before the positive-definiteness test. No eigenvalue repair.
    mean = np.mean(proxies, axis=0)
    return 0.5 * (mean + mean.T)


def _strictly_positive_definite(matrix: NDArray[np.floating]) -> bool:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def select_shrinkage(
    forecasts: NDArray[np.floating],
    proxies: NDArray[np.floating],
    *,
    lambda_grid: Sequence[float] = LAMBDA_GRID,
) -> tuple[float, float, NDArray[np.floating]] | None:
    """Choose (c, lambda, H bar) by fitting-set reduced QLIKE.

    Returns None when H bar is not strictly positive definite. Lambda 1
    rebuilds the R1 forecast on this same fitting set.
    """
    hbar = _proxy_mean(proxies)
    if not _strictly_positive_definite(hbar):
        return None
    best_loss = np.inf
    best: tuple[float, float] | None = None
    for lam in lambda_grid:
        # Shrink every fitting-set forecast, then apply the closed-form scale.
        shrunk = lam * forecasts + (1.0 - lam) * hbar
        scale = scalar_multiplier(shrunk, proxies)
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        loss = _mean_reduced_qlike(scale * shrunk, proxies)
        if loss < best_loss:
            best_loss = loss
            best = (scale, float(lam))
    if best is None:
        return None
    return best[0], best[1], hbar


def apply_recalibrated(
    forecast: NDArray[np.floating],
    scale: float,
    lam: float,
    hbar: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    """Apply one origin's map. Lambda 1 ignores H bar and matches R1."""
    if lam == 1.0 or hbar is None:
        mapped = scale * forecast
    else:
        mapped = scale * (lam * forecast + (1.0 - lam) * hbar)
    # Drop the floating-point residual of a symmetric combination.
    return 0.5 * (mapped + mapped.T)


def recalibrate_forecast_sequence(
    forecasts: ArrayLike,
    proxies: ArrayLike,
    *,
    origin_dates: Sequence[str],
    target_dates: Sequence[str],
    segments: Sequence[str],
    refit: ArrayLike,
    min_pairs: int = MIN_RECALIBRATION_PAIRS,
    max_pairs: int = MAX_RECALIBRATION_PAIRS,
    lambda_grid: Sequence[float] = LAMBDA_GRID,
) -> RecalibrationResult:
    """Recalibrate each origin using only earlier same-segment pairs.

    Parameters
    ----------
    forecasts, proxies:
        Shape ``(T, N, N)``. Row ``t`` is the forecast formed at
        ``origin_dates[t]`` for ``target_dates[t]``.
    segments:
        Named segment of each target. A fitting pair must share the
        segment of the target being forecast.
    refit:
        True at origins where the coefficients are re-estimated.
    """
    forecast_cube = np.asarray(forecasts, dtype=float)
    proxy_cube = np.asarray(proxies, dtype=float)
    refit_flags = np.asarray(refit, dtype=bool)
    n_rows = forecast_cube.shape[0]
    if forecast_cube.shape != proxy_cube.shape or forecast_cube.ndim != 3:
        raise ValueError("forecasts and proxies must share shape (T, N, N)")
    if not (
        len(origin_dates)
        == len(target_dates)
        == len(segments)
        == refit_flags.shape[0]
        == n_rows
    ):
        raise ValueError("dates, segments, and refit must have length T")

    r1 = np.empty_like(forecast_cube)
    r2 = np.empty_like(forecast_cube)
    scale_r1 = np.ones(n_rows, dtype=float)
    scale_r2 = np.ones(n_rows, dtype=float)
    lambda_r2 = np.ones(n_rows, dtype=float)
    identity_r1 = np.ones(n_rows, dtype=bool)
    identity_r2 = np.ones(n_rows, dtype=bool)
    failures: list[str] = []
    held_r1: float | None = None
    held_r2: tuple[float, float, NDArray[np.floating]] | None = None

    for origin_index in range(n_rows):
        r2_identity_this_origin = False
        if refit_flags[origin_index]:
            indices = fitting_indices(
                origin_dates,
                target_dates,
                segments,
                origin_index,
                max_pairs=max_pairs,
            )
            if indices.shape[0] >= min_pairs:
                past_h = forecast_cube[indices]
                past_s = proxy_cube[indices]
                # Estimate R1 even when the proxy mean later fails for R2.
                held_r1 = scalar_multiplier(past_h, past_s)
                selected = select_shrinkage(
                    past_h, past_s, lambda_grid=lambda_grid
                )
                if selected is None:
                    failures.append(str(origin_dates[origin_index]))
                    r2_identity_this_origin = True
                else:
                    held_r2 = selected
        current = forecast_cube[origin_index]
        if held_r1 is not None and np.isfinite(held_r1) and held_r1 > 0.0:
            scale_r1[origin_index] = held_r1
            identity_r1[origin_index] = False
            r1[origin_index] = held_r1 * current
        else:
            r1[origin_index] = current
        # A failed proxy mean uses the identity here and does not replace
        # the last successful R2 coefficients.
        if held_r2 is not None and not r2_identity_this_origin:
            scale, lam, hbar = held_r2
            scale_r2[origin_index] = scale
            lambda_r2[origin_index] = lam
            identity_r2[origin_index] = False
            r2[origin_index] = apply_recalibrated(current, scale, lam, hbar)
        else:
            r2[origin_index] = current

    return RecalibrationResult(
        r1=r1,
        r2=r2,
        scale_r1=scale_r1,
        scale_r2=scale_r2,
        lambda_r2=lambda_r2,
        identity_r1=identity_r1,
        identity_r2=identity_r2,
        hbar_failure_origins=tuple(failures),
    )
