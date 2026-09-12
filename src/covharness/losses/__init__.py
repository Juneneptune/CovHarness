"""Covariance-space evaluation losses.

Primary ranking loss is reduced multivariate QLIKE. Squared Frobenius is the
complementary robust criterion. Full Stein is available when the proxy is SPD.
Ordinary unsquared Frobenius is retained only as a labeled non-robust contrast.
"""

from covharness.losses.contracts import (
    ForecastNotPositiveDefiniteError,
    InvalidCovarianceMatrixError,
    ProxyNotPositiveDefiniteError,
    PSD_ATOL,
    SYMMETRY_ATOL,
)
from covharness.losses.frobenius import squared_frobenius_loss, unsquared_frobenius_loss
from covharness.losses.localization import (
    CovarianceLocalizationDiagnostic,
    covariance_localization_diagnostic,
)
from covharness.losses.qlike import full_stein_loss, reduced_qlike_loss
from covharness.losses.robustness import (
    AnalyticExpectedLosses,
    RobustnessMonteCarloResult,
    analytic_expected_losses,
    monte_carlo_proxy_ranking,
    plot_proxy_robustness,
)

__all__ = [
    "AnalyticExpectedLosses",
    "CovarianceLocalizationDiagnostic",
    "ForecastNotPositiveDefiniteError",
    "InvalidCovarianceMatrixError",
    "PSD_ATOL",
    "ProxyNotPositiveDefiniteError",
    "RobustnessMonteCarloResult",
    "SYMMETRY_ATOL",
    "analytic_expected_losses",
    "covariance_localization_diagnostic",
    "full_stein_loss",
    "monte_carlo_proxy_ranking",
    "plot_proxy_robustness",
    "reduced_qlike_loss",
    "squared_frobenius_loss",
    "unsquared_frobenius_loss",
]
