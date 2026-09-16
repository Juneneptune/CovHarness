"""One-day-ahead covariance forecasting models.

The common contract is :class:`CovarianceModel`. Realized-covariance
baselines inherit :class:`RealizedCovarianceModel` and consume a
``(T, N, N)`` origin window. HARQ-DRD uses the same forecast object
and a ``fit`` that also takes a ``(T, N)`` per-asset quarticity window.
Daily-return models will use the same forecast object later. Models
do not inspect protocol block labels.
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
    InvalidModelForecastError,
    InvalidModelInputError,
)
from covharness.models.har_drd import HARDRDRealizedCovariance
from covharness.models.harq_drd import HARQDRDRealizedCovariance
from covharness.models.random_walk import RandomWalkRealizedCovariance

__all__ = [
    "CovarianceForecast",
    "CovarianceModel",
    "EWMARealizedCovariance",
    "ForecastDiagnostics",
    "HARDRDRealizedCovariance",
    "HARQDRDRealizedCovariance",
    "InvalidModelConfigurationError",
    "InvalidModelForecastError",
    "InvalidModelInputError",
    "ModelIdentity",
    "RandomWalkRealizedCovariance",
    "RealizedCovarianceModel",
    "as_realized_covariance_history",
    "forecast_diagnostics",
]
