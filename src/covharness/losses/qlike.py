"""Reduced multivariate QLIKE and full Stein loss.

Reduced QLIKE is the primary ranking loss because it remains defined for a
singular PSD proxy. Full Stein additionally requires SPD ``S`` because of
``logdet(S)``. Neither function repairs a forecast.
"""

from __future__ import annotations

from numpy.typing import ArrayLike

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
