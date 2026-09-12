"""Descriptive variance-versus-correlation localization diagnostics.

These quantities are not additive components of squared Frobenius or QLIKE.
Correlation normalization is nonlinear, so an unbiased covariance proxy does
not imply an unbiased correlation proxy. The diagnostics do not inherit the
Patton / Laurent–Rombouts–Violante ranking-consistency guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.losses.contracts import (
    InvalidCovarianceMatrixError,
    as_square_finite,
    require_matching_square,
    require_symmetric,
)


@dataclass(frozen=True)
class CovarianceLocalizationDiagnostic:
    """Descriptive localization of a covariance forecast error.

    ``variance_squared_error`` is ``sum_i (S_ii - H_ii)^2``.

    ``correlation_frobenius_squared`` is ``||R(S) - R(H)||_F^2`` where
    ``R(A) = D(A)^{-1} A D(A)^{-1}`` and ``D(A) = diag(sqrt(A_ii))``.

    Neither term is a proxy-robust ranking loss.
    """

    variance_squared_error: float
    correlation_frobenius_squared: float
    n_assets: int
    proxy_variances: NDArray[np.floating]
    forecast_variances: NDArray[np.floating]
    proxy_correlation: NDArray[np.floating]
    forecast_correlation: NDArray[np.floating]


def implied_correlation(matrix: NDArray[np.floating], name: str) -> NDArray[np.floating]:
    """``R = D^{-1} A D^{-1}`` with ``D = diag(sqrt(diag(A)))``.

    Zero or negative diagonal entries are rejected. They are not clipped.
    """
    variances = np.diag(matrix).astype(float, copy=True)
    if np.any(variances <= 0.0):
        raise InvalidCovarianceMatrixError(
            f"{name} has a nonpositive diagonal entry, so implied correlation "
            "is undefined. The diagnostic does not repair the diagonal."
        )
    # Scale rows and columns by inverse standard deviations. No clipping.
    scales = 1.0 / np.sqrt(variances)
    return matrix * np.outer(scales, scales)


def covariance_localization_diagnostic(
    proxy: ArrayLike,
    forecast: ArrayLike,
) -> CovarianceLocalizationDiagnostic:
    """Variance-space and correlation-space discrepancies. Not a ranking loss."""
    proxy_s = as_square_finite(proxy, "proxy S")
    forecast_h = as_square_finite(forecast, "forecast H")
    require_matching_square(proxy_s, forecast_h)
    require_symmetric(proxy_s, "proxy S")
    require_symmetric(forecast_h, "forecast H")

    proxy_var = np.diag(proxy_s).astype(float, copy=True)
    forecast_var = np.diag(forecast_h).astype(float, copy=True)
    # Sum of squared diagonal (variance) errors.
    variance_sse = float(np.sum((proxy_var - forecast_var) ** 2))
    proxy_corr = implied_correlation(proxy_s, "proxy S")
    forecast_corr = implied_correlation(forecast_h, "forecast H")
    corr_resid = proxy_corr - forecast_corr
    return CovarianceLocalizationDiagnostic(
        variance_squared_error=variance_sse,
        correlation_frobenius_squared=float(np.sum(corr_resid * corr_resid)),
        n_assets=int(proxy_s.shape[0]),
        proxy_variances=proxy_var,
        forecast_variances=forecast_var,
        proxy_correlation=proxy_corr,
        forecast_correlation=forecast_corr,
    )
