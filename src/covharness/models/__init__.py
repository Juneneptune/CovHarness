"""One-day-ahead covariance forecasting models.

The common contract is :class:`CovarianceModel`. Realized-covariance
baselines inherit :class:`RealizedCovarianceModel` and consume a
``(T, N, N)`` origin window. Daily-return models will use the same
forecast object later. Models do not inspect protocol block labels.
"""

from covharness.models.base import (
    CovarianceForecast,
    CovarianceModel,
    ForecastDiagnostics,
    ModelIdentity,
    RealizedCovarianceModel,
    as_realized_covariance_history,
    forecast_diagnostics,
)
from covharness.models.ewma import EWMARealizedCovariance
from covharness.models.exceptions import (
    InvalidModelConfigurationError,
    InvalidModelInputError,
)
from covharness.models.random_walk import RandomWalkRealizedCovariance

__all__ = [
    "CovarianceForecast",
    "CovarianceModel",
    "EWMARealizedCovariance",
    "ForecastDiagnostics",
    "InvalidModelConfigurationError",
    "InvalidModelInputError",
    "ModelIdentity",
    "RandomWalkRealizedCovariance",
    "RealizedCovarianceModel",
    "as_realized_covariance_history",
    "forecast_diagnostics",
]
