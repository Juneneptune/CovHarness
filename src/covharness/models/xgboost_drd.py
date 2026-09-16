"""XGBoost-DRD one-day-ahead realized-covariance architecture control.

XGBoost-DRD completes the controlled HAR-DRD to Ridge-DRD to XGBoost-DRD
ladder. Targets, non-overlapping 1/4/17 predictors, dummy-free within
transformation, Ridge RMS-scaled predictor representation, pair order,
rolling cadence, covariance reconstruction, and the headline repair
rule are the HAR/Ridge objects. The linear ridge learner is replaced by
two pooled squared-error boosted trees.

Scaling is imposed so that the numerical predictor representation
matches Ridge exactly. It is not imposed because trees require scaling.
There is no asset or pair identity feature, no group-ID column, no
quarticity term, no cross-section, no log-variance transform, and no
Fisher transform. Headline fitting is deterministic CPU hist.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import xgboost
from numpy.typing import ArrayLike, NDArray
from xgboost import XGBRegressor

from covharness.models.base import (
    CovarianceForecast,
    ModelIdentity,
    RealizedCovarianceModel,
    as_realized_covariance_history,
    pack_forecast,
)
from covharness.models.capabilities import (
    FitInput,
    ModelCapabilities,
    RollingCadence,
    UpdateObservable,
)
from covharness.models.exceptions import (
    InvalidModelConfigurationError,
    InvalidModelForecastError,
    InvalidModelInputError,
)
from covharness.models.har_drd import (
    HAR_SLOPES,
    HARDRDRawValidity,
    LAG_CONVENTION,
    MIN_WINDOW_LENGTH,
    PAIR_ORDERING,
    RESPONSE_START,
    _drd_panel,
    forecast_predictors,
    har_regression_indices,
    har_response_design,
    headline_repaired_forecast,
    pair_count,
    pair_panel_from_correlations,
)
from covharness.models.ridge_drd import (
    SCALING_CONVENTION,
    scale_within_predictors,
    within_predictor_scales,
    within_transformed_arrays,
)

MODEL_NAME = "xgboost_drd"
PINNED_XGBOOST_VERSION = "3.2.0"
WITHIN_CONVENTION = "group_demean_no_dummies"
OBJECTIVE = "reg:squarederror"
BOOSTER_TYPE = "gbtree"
TREE_METHOD = "hist"
DEVICE = "cpu"
N_JOBS = 1
SUBSAMPLE = 1.0
COLSAMPLE_BYTREE = 1.0
COLSAMPLE_BYLEVEL = 1.0
COLSAMPLE_BYNODE = 1.0
GROW_POLICY = "depthwise"
MAX_BIN = 256
BASE_SCORE = 0.0
RANDOM_STATE = 0
EARLY_STOPPING = False
N_BOOSTERS = 2
SLOPE_NAMES = ("daily", "weekly", "monthly")


@dataclass(frozen=True)
class XGBoostDRDFitState:
    """Auditable XGBoost-DRD group means, scales, boosters, and window metadata."""

    ybar_D: NDArray[np.floating]
    Xbar_D: NDArray[np.floating]
    scale_D: NDArray[np.floating]
    ybar_R: NDArray[np.floating]
    Xbar_R: NDArray[np.floating]
    scale_R: NDArray[np.floating]
    origin_X_D: NDArray[np.floating]
    origin_X_R: NDArray[np.floating]
    variance_booster: XGBRegressor
    correlation_booster: XGBRegressor
    n_assets: int
    n_pairs: int
    window_length: int
    n_regression_dates: int
    pair_ordering: str
    lag_convention: str
    within_convention: str
    scaling_convention: str
    xgboost_version: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: float
    reg_lambda: float
    reg_alpha: float
    gamma: float
    objective: str
    booster_type: str
    tree_method: str
    device: str
    n_jobs: int
    subsample: float
    colsample_bytree: float
    colsample_bylevel: float
    colsample_bynode: float
    grow_policy: str
    max_bin: int
    base_score: float
    random_state: int
    early_stopping: bool
    n_boosters: int
    window_mean: NDArray[np.floating]


class XGBoostDRDRealizedCovariance(RealizedCovarianceModel):
    """One-day-ahead XGBoost-DRD of a realized-covariance window."""

    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.WINDOW_STATE,
        fit_input=FitInput.REALIZED_COVARIANCE,
        update_observable=UpdateObservable.REALIZED_COVARIANCE_WINDOW,
    )

    def __init__(
        self,
        *,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        min_child_weight: float,
        reg_lambda: float,
        reg_alpha: float,
        gamma: float,
    ) -> None:
        # Require the seven public hyperparameters. Frozen package constants stay implicit.
        self._n_estimators = _require_positive_int("n_estimators", n_estimators)
        self._max_depth = _require_positive_int("max_depth", max_depth)
        self._learning_rate = _require_finite_float(
            "learning_rate", learning_rate, exclusive_minimum=0.0
        )
        self._min_child_weight = _require_finite_float(
            "min_child_weight", min_child_weight, minimum=0.0
        )
        self._reg_lambda = _require_finite_float("reg_lambda", reg_lambda, minimum=0.0)
        self._reg_alpha = _require_finite_float("reg_alpha", reg_alpha, minimum=0.0)
        self._gamma = _require_finite_float("gamma", gamma, minimum=0.0)
        self._fit: XGBoostDRDFitState | None = None
        self._raw_validity: HARDRDRawValidity | None = None
        self._v_raw: NDArray[np.floating] | None = None
        self._x_raw: NDArray[np.floating] | None = None
        self._R_raw: NDArray[np.floating] | None = None
        self._H_raw: NDArray[np.floating] | None = None

    @property
    def identity(self) -> ModelIdentity:
        configuration: dict[str, object] = {
            "n_estimators": int(self._n_estimators),
            "max_depth": int(self._max_depth),
            "learning_rate": float(self._learning_rate),
            "min_child_weight": float(self._min_child_weight),
            "reg_lambda": float(self._reg_lambda),
            "reg_alpha": float(self._reg_alpha),
            "gamma": float(self._gamma),
            "objective": OBJECTIVE,
            "booster": BOOSTER_TYPE,
            "tree_method": TREE_METHOD,
            "device": DEVICE,
            "n_jobs": N_JOBS,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "colsample_bylevel": COLSAMPLE_BYLEVEL,
            "colsample_bynode": COLSAMPLE_BYNODE,
            "grow_policy": GROW_POLICY,
            "max_bin": MAX_BIN,
            "base_score": BASE_SCORE,
            "random_state": RANDOM_STATE,
            "early_stopping": EARLY_STOPPING,
            "n_boosters": N_BOOSTERS,
            "lag_convention": LAG_CONVENTION,
            "pair_ordering": PAIR_ORDERING,
            "within_convention": WITHIN_CONVENTION,
            "scaling_convention": SCALING_CONVENTION,
            "n_slope_columns": HAR_SLOPES,
            "slope_names": SLOPE_NAMES,
            "responses_standardized": False,
            "pinned_xgboost_version": PINNED_XGBOOST_VERSION,
        }
        if self._fit is not None:
            configuration.update(
                {
                    "n_assets": self._fit.n_assets,
                    "n_pairs": self._fit.n_pairs,
                    "window_length": self._fit.window_length,
                    "n_regression_dates": self._fit.n_regression_dates,
                    "xgboost_version": self._fit.xgboost_version,
                }
            )
        if self._raw_validity is not None:
            configuration.update(
                {
                    "raw_nonfinite_variance": self._raw_validity.raw_nonfinite_variance,
                    "raw_nonpositive_variance": self._raw_validity.raw_nonpositive_variance,
                    "raw_correlation_out_of_bounds": (
                        self._raw_validity.raw_correlation_out_of_bounds
                    ),
                    "raw_correlation_not_pd": self._raw_validity.raw_correlation_not_pd,
                    "raw_covariance_not_pd": self._raw_validity.raw_covariance_not_pd,
                    "repaired": self._raw_validity.repaired,
                    "repair_method": self._raw_validity.repair_method,
                }
            )
        return ModelIdentity(name=MODEL_NAME, configuration=configuration)

    @property
    def fit_state(self) -> XGBoostDRDFitState:
        """Return the stored maps. ``fit`` must have been called."""
        return _require_fit(self._fit)

    @property
    def raw_validity(self) -> HARDRDRawValidity | None:
        """Return flags from the most recent ``forecast`` call, if any."""
        return self._raw_validity

    def fit(self, realized_covariances: ArrayLike) -> XGBoostDRDRealizedCovariance:
        """Estimate XGBoost-DRD on the origin window. Inputs are not mutated."""
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _ = history.shape
        if n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError(
                "XGBoost-DRD requires at least "
                f"{MIN_WINDOW_LENGTH} origin-window matrices so that "
                f"response dates {RESPONSE_START},...,T-1 exist; got T={n_times}"
            )
        if n_assets < 2:
            raise InvalidModelInputError(
                "XGBoost-DRD requires at least two assets so that a correlation "
                f"HAR is defined; got N={n_assets}"
            )
        # Reuse HAR-DRD DRD panels. Do not form a second decomposition.
        variances, correlations = _drd_panel(history)
        n_pairs = pair_count(n_assets)
        pair_panel = pair_panel_from_correlations(correlations)
        response_index = har_regression_indices(n_times)
        n_dates = int(response_index.size)

        # Build the same non-overlapping 1/4/17 designs as HAR and Ridge.
        variance_y, variance_x = har_response_design(variances, response_index)
        correlation_y, correlation_x = har_response_design(pair_panel, response_index)
        ybar_D, Xbar_D, ytilde_D, z_D, scale_D = scaled_within_design(
            variance_y, variance_x
        )
        ybar_R, Xbar_R, ytilde_R, z_R, scale_R = scaled_within_design(
            correlation_y, correlation_x
        )
        if z_D.shape[1] != HAR_SLOPES or z_R.shape[1] != HAR_SLOPES:
            raise InvalidModelForecastError(
                "XGBoost-DRD within designs must have exactly three slope columns"
            )

        # Train exactly two pooled boosters. Group IDs are not passed.
        variance_booster = fit_squared_error_booster(
            z_D,
            ytilde_D,
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            min_child_weight=self._min_child_weight,
            reg_lambda=self._reg_lambda,
            reg_alpha=self._reg_alpha,
            gamma=self._gamma,
        )
        correlation_booster = fit_squared_error_booster(
            z_R,
            ytilde_R,
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            min_child_weight=self._min_child_weight,
            reg_lambda=self._reg_lambda,
            reg_alpha=self._reg_alpha,
            gamma=self._gamma,
        )
        origin_X_D = np.asarray(forecast_predictors(variances).T, dtype=float)
        origin_X_R = np.asarray(forecast_predictors(pair_panel).T, dtype=float)
        self._fit = XGBoostDRDFitState(
            ybar_D=np.array(ybar_D, dtype=float, copy=True),
            Xbar_D=np.array(Xbar_D, dtype=float, copy=True),
            scale_D=np.array(scale_D, dtype=float, copy=True),
            ybar_R=np.array(ybar_R, dtype=float, copy=True),
            Xbar_R=np.array(Xbar_R, dtype=float, copy=True),
            scale_R=np.array(scale_R, dtype=float, copy=True),
            origin_X_D=np.array(origin_X_D, dtype=float, copy=True),
            origin_X_R=np.array(origin_X_R, dtype=float, copy=True),
            variance_booster=variance_booster,
            correlation_booster=correlation_booster,
            n_assets=n_assets,
            n_pairs=n_pairs,
            window_length=n_times,
            n_regression_dates=n_dates,
            pair_ordering=PAIR_ORDERING,
            lag_convention=LAG_CONVENTION,
            within_convention=WITHIN_CONVENTION,
            scaling_convention=SCALING_CONVENTION,
            xgboost_version=str(xgboost.__version__),
            n_estimators=int(self._n_estimators),
            max_depth=int(self._max_depth),
            learning_rate=float(self._learning_rate),
            min_child_weight=float(self._min_child_weight),
            reg_lambda=float(self._reg_lambda),
            reg_alpha=float(self._reg_alpha),
            gamma=float(self._gamma),
            objective=OBJECTIVE,
            booster_type=BOOSTER_TYPE,
            tree_method=TREE_METHOD,
            device=DEVICE,
            n_jobs=N_JOBS,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            colsample_bylevel=COLSAMPLE_BYLEVEL,
            colsample_bynode=COLSAMPLE_BYNODE,
            grow_policy=GROW_POLICY,
            max_bin=MAX_BIN,
            base_score=BASE_SCORE,
            random_state=RANDOM_STATE,
            early_stopping=EARLY_STOPPING,
            n_boosters=N_BOOSTERS,
            window_mean=np.mean(history, axis=0),
        )
        self._raw_validity = None
        self._v_raw = None
        self._x_raw = None
        self._R_raw = None
        self._H_raw = None
        return self

    def forecast(self) -> CovarianceForecast:
        """Return the one-step XGBoost-DRD forecast, repairing only when required."""
        state = _require_fit(self._fit)
        v_raw, x_raw = _raw_components(state)
        H_final, validity, R_raw, H_raw = headline_repaired_forecast(
            v_raw,
            x_raw,
            state.n_assets,
            state.window_mean,
            fallback_name="XGBoost-DRD",
        )
        self._v_raw = np.array(v_raw, dtype=float, copy=True)
        self._x_raw = np.array(x_raw, dtype=float, copy=True)
        self._R_raw = np.array(R_raw, dtype=float, copy=True)
        self._H_raw = None if H_raw is None else np.array(H_raw, dtype=float, copy=True)
        self._raw_validity = validity
        return pack_forecast(H_final, self.identity)

    def update_window(
        self, realized_covariances: ArrayLike
    ) -> XGBoostDRDRealizedCovariance:
        """Rebuild origin HAR features and the current-window fallback mean.

        Group means, RMS scales, hyperparameters, and both boosters remain
        the values stored at the last ``fit``.
        """
        state = _require_fit(self._fit)
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _n_cols = history.shape
        if n_assets != state.n_assets:
            raise InvalidModelInputError(
                "XGBoost-DRD update_window N must equal the fitted width "
                f"{state.n_assets}; got N={n_assets}"
            )
        if n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError(
                "XGBoost-DRD update_window requires T >= "
                f"{MIN_WINDOW_LENGTH}; got T={n_times}"
            )
        # Rebuild origin panels only. Do not call booster.fit.
        variances, correlations = _drd_panel(history)
        pair_panel = pair_panel_from_correlations(correlations)
        origin_X_D = np.asarray(forecast_predictors(variances).T, dtype=float)
        origin_X_R = np.asarray(forecast_predictors(pair_panel).T, dtype=float)
        self._fit = replace(
            state,
            origin_X_D=np.array(origin_X_D, dtype=float, copy=True),
            origin_X_R=np.array(origin_X_R, dtype=float, copy=True),
            window_length=n_times,
            window_mean=np.mean(history, axis=0),
        )
        self._raw_validity = None
        self._v_raw = None
        self._x_raw = None
        self._R_raw = None
        self._H_raw = None
        return self


def scaled_within_design(
    response: NDArray[np.floating],
    design: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Return group means, within residual, Ridge RMS scales, and scaled Z.

    ``response`` has shape ``(G, K)``. ``design`` has shape ``(G, K, S)``.
    The response is not standardized.
    """
    y_bar, x_bar, y_tilde, x_tilde = within_transformed_arrays(response, design)
    scales = within_predictor_scales(x_tilde)
    z_scaled = scale_within_predictors(x_tilde, scales)
    return y_bar, x_bar, y_tilde, z_scaled, scales


def fit_squared_error_booster(
    predictors: NDArray[np.floating],
    response: NDArray[np.floating],
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    min_child_weight: float,
    reg_lambda: float,
    reg_alpha: float,
    gamma: float,
) -> XGBRegressor:
    """Fit one pooled CPU hist booster. No eval set and no early stopping."""
    design = np.asarray(predictors, dtype=float)
    target = np.asarray(response, dtype=float).reshape(-1)
    if design.ndim != 2 or design.shape[0] != target.shape[0]:
        raise InvalidModelInputError(
            "XGBoost-DRD booster design must have shape (M, S) aligned with y"
        )
    # Freeze every headline package constant. Do not rely on silent defaults.
    booster = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        min_child_weight=float(min_child_weight),
        reg_lambda=float(reg_lambda),
        reg_alpha=float(reg_alpha),
        gamma=float(gamma),
        objective=OBJECTIVE,
        booster=BOOSTER_TYPE,
        tree_method=TREE_METHOD,
        device=DEVICE,
        n_jobs=N_JOBS,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        colsample_bylevel=COLSAMPLE_BYLEVEL,
        colsample_bynode=COLSAMPLE_BYNODE,
        grow_policy=GROW_POLICY,
        max_bin=MAX_BIN,
        base_score=BASE_SCORE,
        random_state=RANDOM_STATE,
        validate_parameters=True,
        verbosity=0,
        early_stopping_rounds=None,
    )
    booster.fit(design, target, verbose=False)
    return booster


def booster_raw_bytes(booster: XGBRegressor) -> bytes:
    """Return a copy of the fitted booster dump for freeze comparisons."""
    return bytes(booster.get_booster().save_raw(raw_format="ubj"))


def origin_scaled_predictors(
    origin_x: NDArray[np.floating],
    x_bar: NDArray[np.floating],
    scales: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Center origin HAR predictors with frozen means, then apply frozen RMS scales."""
    centered = np.asarray(origin_x, dtype=float) - np.asarray(x_bar, dtype=float)
    return scale_within_predictors(centered, scales)


def _raw_components(
    state: XGBoostDRDFitState,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply frozen maps. Predictions are group mean plus booster residual."""
    z_d = origin_scaled_predictors(state.origin_X_D, state.Xbar_D, state.scale_D)
    z_r = origin_scaled_predictors(state.origin_X_R, state.Xbar_R, state.scale_R)
    v_raw = np.asarray(state.ybar_D, dtype=float) + np.asarray(
        state.variance_booster.predict(z_d), dtype=float
    )
    x_raw = np.asarray(state.ybar_R, dtype=float) + np.asarray(
        state.correlation_booster.predict(z_r), dtype=float
    )
    return v_raw, x_raw


def _require_positive_int(name: str, value: object) -> int:
    """Reject a non-integer or non-positive constructor argument."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidModelConfigurationError(
            f"XGBoost-DRD {name} must be an integer >= 1; got {value!r}"
        )
    if value < 1:
        raise InvalidModelConfigurationError(
            f"XGBoost-DRD {name} must be an integer >= 1; got {value!r}"
        )
    return int(value)


def _require_finite_float(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
) -> float:
    """Reject a nonfinite constructor float outside the required domain."""
    if isinstance(value, bool):
        raise InvalidModelConfigurationError(
            f"XGBoost-DRD {name} must be a finite float; got {value!r}"
        )
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise InvalidModelConfigurationError(
            f"XGBoost-DRD {name} must be a finite float; got {value!r}"
        ) from exc
    if not np.isfinite(number):
        raise InvalidModelConfigurationError(
            f"XGBoost-DRD {name} must be a finite float; got {value!r}"
        )
    if minimum is not None and number < minimum:
        raise InvalidModelConfigurationError(
            f"XGBoost-DRD {name} must be >= {minimum}; got {value!r}"
        )
    if exclusive_minimum is not None and number <= exclusive_minimum:
        raise InvalidModelConfigurationError(
            f"XGBoost-DRD {name} must be > {exclusive_minimum}; got {value!r}"
        )
    return number


def _require_fit(state: XGBoostDRDFitState | None) -> XGBoostDRDFitState:
    """Reject a method call issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError(
            f"{MODEL_NAME} has no fitted origin-window state. Call fit before forecast."
        )
    return state
