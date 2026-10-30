"""SHAR-DRD one-day-ahead realized-covariance model.

The variance equation replaces the HAR daily lag with positive and
negative realized semivariances, each with a shared slope. Weekly and
monthly variance lags, and the correlation equation, are the HAR-DRD
objects. Estimation uses the imported within transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.models.base import (
    CovarianceForecast,
    CovarianceModel,
    ModelIdentity,
    as_realized_covariance_history,
    pack_forecast,
)
from covharness.models.capabilities import (
    FitInput,
    ModelCapabilities,
    RollingCadence,
    UpdateObservable,
)
from covharness.models.exceptions import InvalidModelForecastError, InvalidModelInputError
from covharness.models.har_drd import (
    HAR_SLOPES,
    HARDRDRawValidity,
    LAG_CONVENTION,
    MIN_WINDOW_LENGTH,
    PAIR_ORDERING,
    RESPONSE_START,
    _drd_panel,
    fixed_effects_shared_slopes,
    forecast_predictors,
    har_predictors,
    har_regression_indices,
    har_response_design,
    headline_repaired_forecast,
    pair_count,
    pair_panel_from_correlations,
)

MODEL_NAME = "shar_drd"
SHAR_VARIANCE_SLOPES = 4
VARIANCE_SLOPE_ORDER = ("rs_positive", "rs_negative", "weekly", "monthly")


@dataclass(frozen=True)
class SHARDRDFitState:
    """Auditable SHAR-DRD coefficients and window metadata."""

    alpha_S: NDArray[np.floating]
    beta_S: NDArray[np.floating]
    alpha_R: NDArray[np.floating]
    beta_R: NDArray[np.floating]
    n_assets: int
    n_pairs: int
    window_length: int
    n_regression_dates: int
    variance_within_rank: int
    correlation_within_rank: int
    pair_ordering: str
    lag_convention: str
    window_mean: NDArray[np.floating]


class SHARDRDRealizedCovariance(CovarianceModel):
    """One-day-ahead SHAR-DRD on covariance and semivariance windows."""

    model_name = MODEL_NAME
    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.WINDOW_STATE,
        fit_input=FitInput.REALIZED_COVARIANCE_AND_SEMIVARIANCE,
        update_observable=UpdateObservable.REALIZED_COVARIANCE_AND_SEMIVARIANCE_WINDOW,
    )

    def __init__(self) -> None:
        self._fit: SHARDRDFitState | None = None
        self._variance_panel: NDArray[np.floating] | None = None
        self._pair_panel: NDArray[np.floating] | None = None
        self._rs_positive: NDArray[np.floating] | None = None
        self._rs_negative: NDArray[np.floating] | None = None
        self._raw_validity: HARDRDRawValidity | None = None

    @property
    def identity(self) -> ModelIdentity:
        configuration: dict[str, object] = {
            "lag_convention": LAG_CONVENTION,
            "pair_ordering": PAIR_ORDERING,
            "variance_slope_order": VARIANCE_SLOPE_ORDER,
            "specification_id": "SHARDRD01",
        }
        if self._fit is not None:
            configuration.update(
                {
                    "n_assets": self._fit.n_assets,
                    "window_length": self._fit.window_length,
                    "variance_within_rank": self._fit.variance_within_rank,
                    "correlation_within_rank": self._fit.correlation_within_rank,
                }
            )
        if self._raw_validity is not None:
            configuration["repaired"] = self._raw_validity.repaired
            configuration["repair_method"] = self._raw_validity.repair_method
        return ModelIdentity(name=self.model_name, configuration=configuration)

    @property
    def fit_state(self) -> SHARDRDFitState:
        """Return the coefficient state from the latest ``fit``."""
        return _require_fit(self._fit)

    def fit(
        self,
        realized_covariances: ArrayLike,
        semivariance_positive: ArrayLike,
        semivariance_negative: ArrayLike,
    ) -> SHARDRDRealizedCovariance:
        """Estimate SHAR-DRD on the origin window. Inputs are not mutated."""
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _ = history.shape
        if n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError(
                f"SHAR-DRD requires at least {MIN_WINDOW_LENGTH} matrices; got T={n_times}"
            )
        positive = _as_semivariance(semivariance_positive, n_times, n_assets, "RS+")
        negative = _as_semivariance(semivariance_negative, n_times, n_assets, "RS-")
        # Split the covariance into variances and correlations.
        variances, correlations = _drd_panel(history)
        pair_panel = pair_panel_from_correlations(correlations)
        response_index = har_regression_indices(n_times)
        variance_y, variance_x = shar_variance_design(
            variances, positive, negative, response_index
        )
        alpha_s, beta_s, var_rank, var_width = fixed_effects_shared_slopes(variance_y, variance_x)
        correlation_y, correlation_x = har_response_design(pair_panel, response_index)
        alpha_r, beta_r, corr_rank, corr_width = fixed_effects_shared_slopes(
            correlation_y, correlation_x
        )
        if var_width != SHAR_VARIANCE_SLOPES or corr_width != HAR_SLOPES:
            raise InvalidModelForecastError("SHAR-DRD within design has an unexpected width")
        self._variance_panel = variances
        self._pair_panel = pair_panel
        self._rs_positive = positive
        self._rs_negative = negative
        self._fit = SHARDRDFitState(
            alpha_S=np.array(alpha_s, dtype=float, copy=True),
            beta_S=np.array(beta_s, dtype=float, copy=True),
            alpha_R=np.array(alpha_r, dtype=float, copy=True),
            beta_R=np.array(beta_r, dtype=float, copy=True),
            n_assets=n_assets,
            n_pairs=pair_count(n_assets),
            window_length=n_times,
            n_regression_dates=int(response_index.size),
            variance_within_rank=int(var_rank),
            correlation_within_rank=int(corr_rank),
            pair_ordering=PAIR_ORDERING,
            lag_convention=LAG_CONVENTION,
            window_mean=np.mean(history, axis=0),
        )
        self._raw_validity = None
        return self

    def forecast(self) -> CovarianceForecast:
        """Return the one-step SHAR-DRD forecast. The fallback is recorded."""
        state = _require_fit(self._fit)
        variance_z, pair_z = self._origin_predictors()
        # Apply the fitted maps. Do not repair inside the linear predictor.
        v_raw = state.alpha_S + variance_z @ state.beta_S
        x_raw = state.alpha_R + pair_z @ state.beta_R
        matrix, validity, _r_raw, _h_raw = headline_repaired_forecast(
            v_raw, x_raw, state.n_assets, state.window_mean, fallback_name="SHAR-DRD"
        )
        self._raw_validity = validity
        return pack_forecast(matrix, self.identity)

    def update_window(
        self,
        realized_covariances: ArrayLike,
        semivariance_positive: ArrayLike,
        semivariance_negative: ArrayLike,
    ) -> SHARDRDRealizedCovariance:
        """Rebuild origin features. Coefficients stay at the last ``fit``."""
        state = _require_fit(self._fit)
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _ = history.shape
        if n_assets != state.n_assets or n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError("SHAR-DRD update_window shape does not match the fit")
        positive = _as_semivariance(semivariance_positive, n_times, n_assets, "RS+")
        negative = _as_semivariance(semivariance_negative, n_times, n_assets, "RS-")
        variances, correlations = _drd_panel(history)
        self._variance_panel = variances
        self._pair_panel = pair_panel_from_correlations(correlations)
        self._rs_positive = positive
        self._rs_negative = negative
        self._fit = replace(state, window_length=n_times, window_mean=np.mean(history, axis=0))
        self._raw_validity = None
        return self

    def _origin_predictors(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Origin predictors. The target day is not in these panels."""
        if (
            self._variance_panel is None
            or self._pair_panel is None
            or self._rs_positive is None
            or self._rs_negative is None
        ):
            raise InvalidModelInputError("SHAR-DRD has no stored origin panels")
        variance_z = shar_forecast_predictors(
            self._variance_panel, self._rs_positive, self._rs_negative
        ).T
        pair_z = forecast_predictors(self._pair_panel).T
        return variance_z, pair_z


def _as_semivariance(
    values: ArrayLike, n_times: int, n_assets: int, name: str
) -> NDArray[np.floating]:
    """Copy a finite nonnegative ``(T, N)`` semivariance panel."""
    array = np.array(values, dtype=float, copy=True)
    if array.shape != (n_times, n_assets):
        raise InvalidModelInputError(f"{name} must have shape {(n_times, n_assets)}; got {array.shape}")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise InvalidModelInputError(f"{name} must be finite and nonnegative")
    return array


def shar_variance_predictors(
    variances: NDArray[np.floating],
    rs_positive: NDArray[np.floating],
    rs_negative: NDArray[np.floating],
    index: int,
) -> NDArray[np.floating]:
    """Return four variance predictors at response ``index``.

    Columns are lagged RS+, lagged RS-, the four-day weekly variance mean,
    and the seventeen-day monthly variance mean. Row ``index`` is not read.
    """
    if index < RESPONSE_START:
        raise InvalidModelInputError(f"SHAR response index must be at least {RESPONSE_START}; got {index}")
    weekly_monthly = har_predictors(variances, index)
    return np.stack(
        (rs_positive[index - 1], rs_negative[index - 1], weekly_monthly[1], weekly_monthly[2]),
        axis=0,
    )


def shar_forecast_predictors(
    variances: NDArray[np.floating],
    rs_positive: NDArray[np.floating],
    rs_negative: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return origin variance predictors for the target after the window."""
    n_times = int(variances.shape[0])
    if n_times < MIN_WINDOW_LENGTH:
        raise InvalidModelInputError(f"SHAR forecast predictors require T >= {MIN_WINDOW_LENGTH}")
    weekly = variances[n_times - 5 : n_times - 1].mean(axis=0)
    monthly = variances[n_times - 22 : n_times - 5].mean(axis=0)
    return np.stack((rs_positive[n_times - 1], rs_negative[n_times - 1], weekly, monthly), axis=0)


def shar_variance_design(
    variances: NDArray[np.floating],
    rs_positive: NDArray[np.floating],
    rs_negative: NDArray[np.floating],
    response_index: NDArray[np.intp],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``y`` of shape ``(N, K)`` and ``X`` of shape ``(N, K, 4)``."""
    n_assets = int(variances.shape[1])
    n_dates = int(response_index.size)
    y = np.empty((n_assets, n_dates), dtype=float)
    design = np.empty((n_assets, n_dates, SHAR_VARIANCE_SLOPES), dtype=float)
    for position, time_index in enumerate(response_index):
        # Predictors stop at the previous day. The response row is not a feature.
        predictors = shar_variance_predictors(variances, rs_positive, rs_negative, int(time_index))
        y[:, position] = variances[time_index]
        design[:, position, :] = np.asarray(predictors.T, dtype=float)
    return y, design


def _require_fit(state: SHARDRDFitState | None) -> SHARDRDFitState:
    """Reject a forecast issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError("SHAR-DRD has no fitted state. Call fit first.")
    return state
