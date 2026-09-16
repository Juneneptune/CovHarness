"""One-day-ahead covariance forecasting models.

The common contract is :class:`CovarianceModel`. Realized-covariance
baselines inherit :class:`RealizedCovarianceModel` and consume a
``(T, N, N)`` origin window. HARQ-DRD uses the same forecast object
and a ``fit`` that also takes a ``(T, N)`` per-asset quarticity window.
Ridge-DRD consumes the same covariance cube as HAR-DRD and adds an
explicit shared ridge penalty on the three HAR slopes. XGBoost-DRD
consumes that same cube after the same within transform and Ridge RMS
scaling, replacing the linear ridge learner with two pooled boosters.
Standalone Ledoit-Wolf models consume a ``(T, N)`` daily-return window.
DCC, DCC-NL, and LSTM-BEKK consume the same daily-return window and add a
daily ``update`` without re-estimating parameters. Rolling cadence is
declared by :class:`ModelCapabilities`. Models do not inspect
protocol block labels.
"""

from covharness.models.base import (
    CovarianceForecast,
    CovarianceModel,
    ForecastDiagnostics,
    ModelIdentity,
    RealizedCovarianceModel,
    as_daily_return_history,
    as_realized_covariance_history,
    as_realized_covariance_matrix,
    forecast_diagnostics,
)
from covharness.models.capabilities import (
    FitInput,
    ModelCapabilities,
    RollingCadence,
    UpdateObservable,
    capabilities_of,
)
from covharness.models.ewma import EWMARealizedCovariance
from covharness.models.exceptions import (
    InvalidModelConfigurationError,
    InvalidModelForecastError,
    InvalidModelInputError,
)
from covharness.models.har_drd import HARDRDRealizedCovariance
from covharness.models.harq_drd import HARQDRDRealizedCovariance
from covharness.models.ridge_drd import RidgeDRDRealizedCovariance, RidgeDRDFitState
from covharness.models.xgboost_drd import (
    XGBoostDRDFitState,
    XGBoostDRDRealizedCovariance,
)
from covharness.models.dcc import DCCCovariance, DCCFitState, DCCNonlinearCovariance
from covharness.models.lstm_bekk import LSTMBEKKCovariance, LSTMBEKKFitState
from covharness.models.ledoit_wolf import (
    LedoitWolfLinearCovariance,
    LedoitWolfLinearFitState,
    LedoitWolfNonlinearCovariance,
    LedoitWolfNonlinearFitState,
    apply_linear_identity_shrinkage,
    centered_return_moments,
    ledoit_wolf_2004b_shrinkage_coefficient,
)
from covharness.models.random_walk import RandomWalkRealizedCovariance

__all__ = [
    "CovarianceForecast",
    "CovarianceModel",
    "DCCCovariance",
    "DCCFitState",
    "DCCNonlinearCovariance",
    "EWMARealizedCovariance",
    "FitInput",
    "ForecastDiagnostics",
    "HARDRDRealizedCovariance",
    "HARQDRDRealizedCovariance",
    "RidgeDRDFitState",
    "RidgeDRDRealizedCovariance",
    "XGBoostDRDFitState",
    "XGBoostDRDRealizedCovariance",
    "InvalidModelConfigurationError",
    "InvalidModelForecastError",
    "InvalidModelInputError",
    "LedoitWolfLinearCovariance",
    "LedoitWolfLinearFitState",
    "LedoitWolfNonlinearCovariance",
    "LedoitWolfNonlinearFitState",
    "LSTMBEKKCovariance",
    "LSTMBEKKFitState",
    "ModelCapabilities",
    "ModelIdentity",
    "RandomWalkRealizedCovariance",
    "RealizedCovarianceModel",
    "RollingCadence",
    "UpdateObservable",
    "apply_linear_identity_shrinkage",
    "as_daily_return_history",
    "as_realized_covariance_history",
    "as_realized_covariance_matrix",
    "capabilities_of",
    "centered_return_moments",
    "forecast_diagnostics",
    "ledoit_wolf_2004b_shrinkage_coefficient",
]
