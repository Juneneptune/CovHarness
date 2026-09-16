"""Ridge-DRD one-day-ahead realized-covariance control.

Ridge-DRD is the regularized HAR-DRD control. Targets, non-overlapping
1/4/17 predictors, asset and pair fixed effects, pair order, and the
headline repair rule are the HAR-DRD objects. The only change is L2
penalization of the three shared slopes after a within transform.

The penalty is one explicit ``lambda >= 0`` shared by the variance and
correlation maps. Intercepts are not penalized. Predictors are scaled
only after within demeaning. Responses are not scaled. ``lambda = 0``
nests the implemented HAR-DRD OLS problem. There is no quarticity term,
cross-section, log-variance transform, or Fisher transform.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve as scipy_solve

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
    fixed_effects_shared_slopes,
    forecast_predictors,
    har_regression_indices,
    har_response_design,
    headline_repaired_forecast,
    pair_count,
    pair_panel_from_correlations,
)

MODEL_NAME = "ridge_drd"
SCALING_CONVENTION = "within_column_rms_after_demeaning"
PENALTY_CONVENTION = "sse_plus_lambda_l2_on_scaled_slopes"
SLOPE_NAMES = ("daily", "weekly", "monthly")


@dataclass(frozen=True)
class RidgeDRDFitState:
    """Auditable Ridge-DRD coefficients, scales, and window metadata."""

    ridge_lambda: float
    alpha_D: NDArray[np.floating]
    gamma_D: NDArray[np.floating]
    beta_D: NDArray[np.floating]
    scale_D: NDArray[np.floating]
    alpha_R: NDArray[np.floating]
    gamma_R: NDArray[np.floating]
    beta_R: NDArray[np.floating]
    scale_R: NDArray[np.floating]
    n_assets: int
    n_pairs: int
    window_length: int
    n_regression_dates: int
    variance_within_rank: int
    correlation_within_rank: int
    pair_ordering: str
    lag_convention: str
    scaling_convention: str
    penalty_convention: str
    window_mean: NDArray[np.floating]


class RidgeDRDRealizedCovariance(RealizedCovarianceModel):
    """One-day-ahead Ridge-DRD of a realized-covariance window."""

    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.WINDOW_STATE,
        fit_input=FitInput.REALIZED_COVARIANCE,
        update_observable=UpdateObservable.REALIZED_COVARIANCE_WINDOW,
    )

    def __init__(self, *, lambda_: float) -> None:
        self._lambda = _require_ridge_lambda(lambda_)
        self._fit: RidgeDRDFitState | None = None
        self._variance_panel: NDArray[np.floating] | None = None
        self._pair_panel: NDArray[np.floating] | None = None
        self._raw_validity: HARDRDRawValidity | None = None
        self._v_raw: NDArray[np.floating] | None = None
        self._x_raw: NDArray[np.floating] | None = None
        self._R_raw: NDArray[np.floating] | None = None
        self._H_raw: NDArray[np.floating] | None = None

    @property
    def identity(self) -> ModelIdentity:
        configuration: dict[str, object] = {
            "lambda": float(self._lambda),
            "lag_convention": LAG_CONVENTION,
            "pair_ordering": PAIR_ORDERING,
            "scaling_convention": SCALING_CONVENTION,
            "penalty_convention": PENALTY_CONVENTION,
            "n_slope_columns": HAR_SLOPES,
            "slope_names": SLOPE_NAMES,
            "responses_standardized": False,
        }
        if self._fit is not None:
            configuration.update(
                {
                    "n_assets": self._fit.n_assets,
                    "n_pairs": self._fit.n_pairs,
                    "window_length": self._fit.window_length,
                    "n_regression_dates": self._fit.n_regression_dates,
                    "variance_within_rank": self._fit.variance_within_rank,
                    "correlation_within_rank": self._fit.correlation_within_rank,
                }
            )
        if self._raw_validity is not None:
            configuration.update(
                {
                    "raw_nonfinite_variance": self._raw_validity.raw_nonfinite_variance,
                    "raw_nonpositive_variance": self._raw_validity.raw_nonpositive_variance,
                    "raw_nonfinite_correlation": (
                        self._raw_validity.raw_nonfinite_correlation
                    ),
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
    def fit_state(self) -> RidgeDRDFitState:
        """Return the stored coefficients. ``fit`` must have been called."""
        return _require_fit(self._fit)

    @property
    def raw_validity(self) -> HARDRDRawValidity | None:
        """Return flags from the most recent ``forecast`` call, if any."""
        return self._raw_validity

    def fit(self, realized_covariances: ArrayLike) -> RidgeDRDRealizedCovariance:
        """Estimate Ridge-DRD on the origin window. Inputs are not mutated."""
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _ = history.shape
        if n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError(
                "Ridge-DRD requires at least "
                f"{MIN_WINDOW_LENGTH} origin-window matrices so that "
                f"response dates {RESPONSE_START},...,T-1 exist; got T={n_times}"
            )
        if n_assets < 2:
            raise InvalidModelInputError(
                "Ridge-DRD requires at least two assets so that a correlation "
                f"HAR is defined; got N={n_assets}"
            )
        # Reuse HAR-DRD DRD panels. Do not form a second decomposition.
        variances, correlations = _drd_panel(history)
        n_pairs = pair_count(n_assets)
        pair_panel = pair_panel_from_correlations(correlations)
        response_index = har_regression_indices(n_times)
        n_dates = int(response_index.size)

        variance_y, variance_x = har_response_design(variances, response_index)
        alpha_D, gamma_D, beta_D, scale_D, var_rank, var_width = (
            ridge_fixed_effects_shared_slopes(variance_y, variance_x, self._lambda)
        )
        correlation_y, correlation_x = har_response_design(pair_panel, response_index)
        alpha_R, gamma_R, beta_R, scale_R, corr_rank, corr_width = (
            ridge_fixed_effects_shared_slopes(
                correlation_y, correlation_x, self._lambda
            )
        )
        if var_width != HAR_SLOPES or corr_width != HAR_SLOPES:
            raise InvalidModelForecastError(
                "Ridge-DRD within designs must have exactly three slope columns"
            )
        self._variance_panel = variances
        self._pair_panel = pair_panel
        self._fit = RidgeDRDFitState(
            ridge_lambda=float(self._lambda),
            alpha_D=np.array(alpha_D, dtype=float, copy=True),
            gamma_D=np.array(gamma_D, dtype=float, copy=True),
            beta_D=np.array(beta_D, dtype=float, copy=True),
            scale_D=np.array(scale_D, dtype=float, copy=True),
            alpha_R=np.array(alpha_R, dtype=float, copy=True),
            gamma_R=np.array(gamma_R, dtype=float, copy=True),
            beta_R=np.array(beta_R, dtype=float, copy=True),
            scale_R=np.array(scale_R, dtype=float, copy=True),
            n_assets=n_assets,
            n_pairs=n_pairs,
            window_length=n_times,
            n_regression_dates=n_dates,
            variance_within_rank=int(var_rank),
            correlation_within_rank=int(corr_rank),
            pair_ordering=PAIR_ORDERING,
            lag_convention=LAG_CONVENTION,
            scaling_convention=SCALING_CONVENTION,
            penalty_convention=PENALTY_CONVENTION,
            window_mean=np.mean(history, axis=0),
        )
        self._raw_validity = None
        self._v_raw = None
        self._x_raw = None
        self._R_raw = None
        self._H_raw = None
        return self

    def forecast(self) -> CovarianceForecast:
        """Return the one-step Ridge-DRD forecast, repairing only when required."""
        state = _require_fit(self._fit)
        v_raw, x_raw = _raw_components(state, self._origin_predictors())
        H_final, validity, R_raw, H_raw = headline_repaired_forecast(
            v_raw, x_raw, state.n_assets, state.window_mean, fallback_name="Ridge-DRD"
        )
        self._v_raw = np.array(v_raw, dtype=float, copy=True)
        self._x_raw = np.array(x_raw, dtype=float, copy=True)
        self._R_raw = np.array(R_raw, dtype=float, copy=True)
        self._H_raw = None if H_raw is None else np.array(H_raw, dtype=float, copy=True)
        self._raw_validity = validity
        return pack_forecast(H_final, self.identity)

    def update_window(
        self, realized_covariances: ArrayLike
    ) -> RidgeDRDRealizedCovariance:
        """Rebuild origin HAR features and the current-window fallback mean.

        Lambda, within scales, and coefficients remain the values stored at
        the last ``fit``. Origin features stay in raw units. Scales are not
        recomputed from the new window.
        """
        state = _require_fit(self._fit)
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _n_cols = history.shape
        if n_assets != state.n_assets:
            raise InvalidModelInputError(
                "Ridge-DRD update_window N must equal the fitted width "
                f"{state.n_assets}; got N={n_assets}"
            )
        if n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError(
                "Ridge-DRD update_window requires T >= "
                f"{MIN_WINDOW_LENGTH}; got T={n_times}"
            )
        # Rebuild origin panels. Do not touch lambda, scales, or slopes.
        variances, correlations = _drd_panel(history)
        self._variance_panel = variances
        self._pair_panel = pair_panel_from_correlations(correlations)
        self._fit = replace(
            state,
            window_length=n_times,
            window_mean=np.mean(history, axis=0),
        )
        self._raw_validity = None
        self._v_raw = None
        self._x_raw = None
        self._R_raw = None
        self._H_raw = None
        return self

    def _origin_predictors(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """HAR predictors at the supplied origin. Target-day values are unused."""
        _require_fit(self._fit)
        if self._variance_panel is None or self._pair_panel is None:
            raise InvalidModelInputError(
                f"{MODEL_NAME} has no stored origin panels. Call fit before forecast."
            )
        # Origin slices. The target is not in these arrays.
        variance_z = forecast_predictors(self._variance_panel).T
        pair_z = forecast_predictors(self._pair_panel).T
        return variance_z, pair_z


def within_transformed_arrays(
    response: NDArray[np.floating],
    design: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return group means and stacked within arrays. Dummy columns are not built.

    ``response`` has shape ``(G, K)``. ``design`` has shape ``(G, K, S)``.
    Returned ``y_bar`` has shape ``(G,)``. ``x_bar`` has shape ``(G, S)``.
    ``y_tilde`` has shape ``(G*K,)``. ``x_tilde`` has shape ``(G*K, S)``.
    """
    if response.ndim != 2:
        raise InvalidModelInputError("fixed-effects response must have shape (G, K)")
    if design.ndim != 3 or design.shape[:2] != response.shape:
        raise InvalidModelInputError(
            "fixed-effects design must have shape (G, K, S) aligned with y"
        )
    n_groups, n_dates, n_slopes = design.shape
    if n_slopes < 1:
        raise InvalidModelInputError("Ridge within design must have at least one slope column")
    if n_groups < 1 or n_dates < 1:
        raise InvalidModelInputError("fixed-effects design must be non-empty")
    # Demean within groups. Do not make dummy columns.
    y_bar = response.mean(axis=1)
    x_bar = design.mean(axis=1)
    y_tilde = (response - y_bar[:, None]).reshape(n_groups * n_dates)
    x_tilde = (design - x_bar[:, None, :]).reshape(n_groups * n_dates, n_slopes)
    return y_bar, x_bar, y_tilde, x_tilde


def within_predictor_scales(x_tilde: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return column RMS of the within design. Exact zeros become scale 1."""
    if x_tilde.ndim != 2 or x_tilde.shape[0] < 1 or x_tilde.shape[1] < 1:
        raise InvalidModelInputError("within predictor matrix must have shape (M, S)")
    mean_squares = np.mean(x_tilde * x_tilde, axis=0)
    scales = np.sqrt(mean_squares)
    # Exact zero only. Do not drop the column.
    scales = np.where(scales == 0.0, 1.0, scales)
    return np.asarray(scales, dtype=float)


def scale_within_predictors(
    x_tilde: NDArray[np.floating],
    scales: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return ``Xtilde / scale``. The response is not scaled."""
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or scales.shape[0] != x_tilde.shape[1]:
        raise InvalidModelInputError("predictor scales must have length equal to S")
    if np.any(scales <= 0.0) or (not np.isfinite(scales).all()):
        raise InvalidModelForecastError("Ridge predictor scales must be finite and strictly positive")
    return x_tilde / scales


def ridge_scaled_slopes(
    x_scaled: NDArray[np.floating],
    y_tilde: NDArray[np.floating],
    penalty: float,
) -> NDArray[np.floating]:
    """Solve the scaled-space ridge problem. ``lambda=0`` uses ``lstsq``."""
    penalty = _require_ridge_lambda(penalty)
    n_slopes = int(x_scaled.shape[1])
    if penalty == 0.0:
        gamma, _residuals, _rank, _singular = np.linalg.lstsq(
            x_scaled, y_tilde, rcond=None
        )
        return np.asarray(gamma, dtype=float)
    # (X'X + lambda I) gamma = X'y. Do not form an inverse.
    gram = x_scaled.T @ x_scaled
    gram = gram + penalty * np.eye(n_slopes)
    rhs = x_scaled.T @ y_tilde
    gamma = scipy_solve(gram, rhs, assume_a="pos")
    return np.asarray(gamma, dtype=float).reshape(n_slopes)


def ridge_fixed_effects_shared_slopes(
    response: NDArray[np.floating],
    design: NDArray[np.floating],
    penalty: float,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    int,
    int,
]:
    """Return intercepts, scaled slopes, raw slopes, scales, rank, and width.

    Rank is the unregularized within-design rank. Dummy intercept columns
    are not built. Intercepts are recovered after the ridge slopes and
    are not penalized.
    """
    penalty = _require_ridge_lambda(penalty)
    y_bar, x_bar, y_tilde, x_tilde = within_transformed_arrays(response, design)
    n_columns = int(x_tilde.shape[1])
    scales = within_predictor_scales(x_tilde)
    if penalty == 0.0:
        # Nest HAR lstsq on the unscaled within design, including rank deficiency.
        intercepts, beta, har_rank, har_width = fixed_effects_shared_slopes(
            response, design
        )
        gamma = beta * scales
        return intercepts, gamma, beta, scales, har_rank, har_width
    x_scaled = scale_within_predictors(x_tilde, scales)
    gamma = ridge_scaled_slopes(x_scaled, y_tilde, penalty)
    beta = gamma / scales
    intercepts = y_bar - x_bar @ beta
    # Unregularized within rank, before the penalty is applied.
    _beta_ols, _residuals, rank_value, _singular = np.linalg.lstsq(
        x_tilde, y_tilde, rcond=None
    )
    rank = int(rank_value)
    return intercepts, gamma, beta, scales, rank, n_columns


def _raw_components(
    state: RidgeDRDFitState,
    predictors: tuple[NDArray[np.floating], NDArray[np.floating]],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply raw-space maps. Scaled coefficients are not used here."""
    variance_z, pair_z = predictors
    v_raw = state.alpha_D + variance_z @ state.beta_D
    x_raw = state.alpha_R + pair_z @ state.beta_R
    return v_raw, x_raw


def _require_ridge_lambda(penalty: float) -> float:
    """Reject a nonfinite or negative ridge penalty."""
    value = float(penalty)
    if (not np.isfinite(value)) or value < 0.0:
        raise InvalidModelConfigurationError(
            f"Ridge-DRD lambda must be finite and >= 0; got {penalty}"
        )
    return value


def _require_fit(state: RidgeDRDFitState | None) -> RidgeDRDFitState:
    """Reject a method call issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError(
            f"{MODEL_NAME} has no fitted origin-window state. Call fit before forecast."
        )
    return state
