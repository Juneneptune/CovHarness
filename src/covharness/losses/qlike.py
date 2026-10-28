"""Reduced multivariate QLIKE and full Stein loss.

Reduced QLIKE is the primary ranking loss because it remains defined for a
singular PSD proxy. Full Stein additionally requires SPD ``S`` because of
``logdet(S)``. Neither function repairs a forecast.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.losses.contracts import (
    as_square_finite,
    cholesky_factor,
    logdet_from_cholesky,
    require_matching_square,
    require_psd,
    require_symmetric,
    trace_solve_from_cholesky,
)


def reduced_qlike_loss(proxy: ArrayLike, forecast: ArrayLike) -> float:
    """Reduced multivariate QLIKE ``L_Q(S, H) = logdet(H) + trace(H^{-1} S)``.

    ``S`` must be square, finite, symmetric, and PSD. It may be singular.
    ``H`` must be square, finite, symmetric, and strictly PD.

    ``H = L @ L.T`` is obtained by Cholesky. The same factor is used for
    ``logdet(H) = 2 * sum(log(diag(L)))`` and for the solves that compute
    ``trace(H^{-1} S)``. ``inv(H)`` is not formed. A failed Cholesky of ``H``
    is the positive-definiteness failure. No jitter or eigenvalue repair is
    applied.
    """
    # Copy and enforce the documented proxy and forecast contracts.
    proxy_s = as_square_finite(proxy, "proxy S")
    forecast_h = as_square_finite(forecast, "forecast H")
    require_matching_square(proxy_s, forecast_h)
    require_symmetric(proxy_s, "proxy S")
    require_symmetric(forecast_h, "forecast H")
    require_psd(proxy_s, "proxy S")

    # Strict PD of H. Failure is raised; the forecast is not repaired.
    chol_h = cholesky_factor(forecast_h, "forecast H")
    logdet_h = logdet_from_cholesky(chol_h)
    trace_term = trace_solve_from_cholesky(chol_h, proxy_s)
    return float(logdet_h + trace_term)


def full_stein_loss(proxy: ArrayLike, forecast: ArrayLike) -> float:
    """Full Stein loss when both ``S`` and ``H`` are SPD.

    ``L_S(S, H) = trace(H^{-1} S) - logdet(H^{-1} S) - N``.

    Equivalently ``L_S = L_Q(S, H) - logdet(S) - N``. Reduced QLIKE and full
    Stein are not numerically equal. For a common SPD target they differ by a
    target-only term, so model rankings and pairwise loss differentials agree.

    Singular ``S`` is rejected because ``logdet(S)`` is undefined. Use
    reduced QLIKE for a rank-deficient PSD proxy.
    """
    # Copy and enforce symmetry before any factorization.
    proxy_s = as_square_finite(proxy, "proxy S")
    forecast_h = as_square_finite(forecast, "forecast H")
    require_matching_square(proxy_s, forecast_h)
    require_symmetric(proxy_s, "proxy S")
    require_symmetric(forecast_h, "forecast H")

    # SPD proxy. Singular S fails here rather than through a repaired logdet.
    chol_s = cholesky_factor(proxy_s, "proxy S")
    chol_h = cholesky_factor(forecast_h, "forecast H")
    n_assets = proxy_s.shape[0]
    logdet_s = logdet_from_cholesky(chol_s)
    logdet_h = logdet_from_cholesky(chol_h)
    trace_term = trace_solve_from_cholesky(chol_h, proxy_s)
    reduced = logdet_h + trace_term
    return float(reduced - logdet_s - n_assets)


def whitened_trace_ratio(proxy: ArrayLike, forecast: ArrayLike) -> float:
    """Return ``tr(H^{-1} S) / N`` from the QLIKE Cholesky solves.

    ``inv(H)`` is not formed. ``H`` must be strictly PD. The ratio is a
    calibration diagnostic. It is not a ranking loss.
    """
    # Copy and enforce the same finite, symmetric, PSD contracts as reduced QLIKE.
    proxy_s = as_square_finite(proxy, "proxy S")
    forecast_h = as_square_finite(forecast, "forecast H")
    require_matching_square(proxy_s, forecast_h)
    require_symmetric(proxy_s, "proxy S")
    require_symmetric(forecast_h, "forecast H")
    require_psd(proxy_s, "proxy S")
    # Reuse the QLIKE Cholesky solves. Do not form inv(H).
    chol_h = cholesky_factor(forecast_h, "forecast H")
    trace_term = trace_solve_from_cholesky(chol_h, proxy_s)
    return float(trace_term / proxy_s.shape[0])


def whitened_trace_ratio_panel(
    proxy: ArrayLike,
    forecast: ArrayLike,
) -> NDArray[np.floating]:
    """Return the date-wise whitened ratio for panels of shape ``(T, N, N)``."""
    proxy_panel = np.asarray(proxy, dtype=float)
    forecast_panel = np.asarray(forecast, dtype=float)
    if proxy_panel.ndim != 3 or forecast_panel.ndim != 3:
        raise ValueError("proxy and forecast panels must have shape (T, N, N)")
    if proxy_panel.shape != forecast_panel.shape:
        raise ValueError("proxy and forecast panels must have the same shape")
    n_obs = int(proxy_panel.shape[0])
    ratios = np.empty(n_obs, dtype=float)
    # Score one origin-target date at a time with the same Cholesky path.
    for time in range(n_obs):
        ratios[time] = whitened_trace_ratio(proxy_panel[time], forecast_panel[time])
    return ratios
