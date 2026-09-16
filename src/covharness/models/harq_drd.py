"""HARQ-DRD one-day-ahead realized-covariance baseline.

HARQ-DRD is HAR-DRD plus one daily per-asset quarticity interaction on
the variance equation. Correlations are the unchanged HAR-DRD three-slope
map. Estimation uses the within transformation. The headline model is in
levels. There is no log-variance transform, Fisher transform, ridge,
graph term, weekly quarticity term, or monthly quarticity term.

The model consumes a precomputed ``(T, N)`` realized-quarticity window
aligned with the covariance history. It does not read raw intraday
returns. The documented per-asset definition is ``RQ_i=(M/3) sum_l r_{i,l}^4``.
That is not the cross-sectional aggregate used by the second Giacomini-White
state.
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
    har_regression_indices,
    har_response_design,
    headline_repaired_forecast,
    pair_count,
    pair_panel_from_correlations,
)

MODEL_NAME = "harq_drd"
HARQ_VARIANCE_SLOPES = 4
VARIANCE_SLOPE_ORDER = (
    "daily",
    "quarticity_interaction",
    "weekly",
    "monthly",
)
QUARTICITY_DEFINITION = "per_asset_rq_m_over_3_sum_r4"
QUARTICITY_DEFINITION_FORMULA = "(M/3) * sum_l r_{i,l}^4"


@dataclass(frozen=True)
class HARQDRDFitState:
    """Auditable HARQ-DRD coefficients and window metadata."""

    alpha_Q: NDArray[np.floating]
    beta_Q: NDArray[np.floating]
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
    quarticity_definition: str
    window_mean: NDArray[np.floating]


class HARQDRDRealizedCovariance(CovarianceModel):
    """One-day-ahead HARQ-DRD of a realized-covariance and per-asset RQ window.

    ``fit`` takes realized covariances and a matching ``(T, N)`` quarticity
    matrix. The class does not inherit ``RealizedCovarianceModel`` because
    that parent ``fit`` accepts covariance history only.
    """

    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.WINDOW_STATE,
        fit_input=FitInput.REALIZED_COVARIANCE_AND_RQ,
        update_observable=UpdateObservable.REALIZED_COVARIANCE_AND_RQ_WINDOW,
    )

    def __init__(self) -> None:
        self._fit: HARQDRDFitState | None = None
        self._variance_panel: NDArray[np.floating] | None = None
        self._pair_panel: NDArray[np.floating] | None = None
        self._quarticity_panel: NDArray[np.floating] | None = None
        self._raw_validity: HARDRDRawValidity | None = None
        self._v_raw: NDArray[np.floating] | None = None
        self._x_raw: NDArray[np.floating] | None = None
        self._R_raw: NDArray[np.floating] | None = None
        self._H_raw: NDArray[np.floating] | None = None

    @property
    def identity(self) -> ModelIdentity:
        configuration: dict[str, object] = {
            "lag_convention": LAG_CONVENTION,
            "pair_ordering": PAIR_ORDERING,
            "quarticity_definition": QUARTICITY_DEFINITION,
            "variance_slope_order": VARIANCE_SLOPE_ORDER,
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
    def fit_state(self) -> HARQDRDFitState:
        """Return the stored coefficients. ``fit`` must have been called."""
        return _require_fit(self._fit)

    @property
    def raw_validity(self) -> HARDRDRawValidity | None:
        """Return flags from the most recent ``forecast`` call, if any."""
        return self._raw_validity

    def fit(
        self,
        realized_covariances: ArrayLike,
        realized_quarticity: ArrayLike,
    ) -> HARQDRDRealizedCovariance:
        """Estimate HARQ-DRD on the origin window. Inputs are not mutated."""
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _ = history.shape
        if n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError(
                "HARQ-DRD requires at least "
                f"{MIN_WINDOW_LENGTH} origin-window matrices so that "
                f"response dates {RESPONSE_START},...,T-1 exist; got T={n_times}"
            )
        if n_assets < 2:
            raise InvalidModelInputError(
                "HARQ-DRD requires at least two assets so that a correlation "
                f"HAR is defined; got N={n_assets}"
            )
        quarticity = as_realized_quarticity_history(
            realized_quarticity, n_times=n_times, n_assets=n_assets
        )
        # Require a strictly positive diagonal on every covariance slice.
        variances, correlations = _drd_panel(history)
        n_pairs = pair_count(n_assets)
        pair_panel = pair_panel_from_correlations(correlations)
        response_index = har_regression_indices(n_times)
        n_dates = int(response_index.size)

        variance_y, variance_x = harq_variance_design(
            variances, quarticity, response_index
        )
        alpha_Q, beta_Q, var_rank, var_width = fixed_effects_shared_slopes(
            variance_y, variance_x
        )
        correlation_y, correlation_x = har_response_design(pair_panel, response_index)
        alpha_R, beta_R, corr_rank, corr_width = fixed_effects_shared_slopes(
            correlation_y, correlation_x
        )
        if var_width != HARQ_VARIANCE_SLOPES:
            raise InvalidModelForecastError(
                "HARQ-DRD variance within design must have exactly four slope columns"
            )
        if corr_width != HAR_SLOPES:
            raise InvalidModelForecastError(
                "HARQ-DRD correlation within design must have exactly three slope columns"
            )
        self._variance_panel = variances
        self._pair_panel = pair_panel
        self._quarticity_panel = quarticity
        self._fit = HARQDRDFitState(
            alpha_Q=np.array(alpha_Q, dtype=float, copy=True),
            beta_Q=np.array(beta_Q, dtype=float, copy=True),
            alpha_R=np.array(alpha_R, dtype=float, copy=True),
            beta_R=np.array(beta_R, dtype=float, copy=True),
            n_assets=n_assets,
            n_pairs=n_pairs,
            window_length=n_times,
            n_regression_dates=n_dates,
            variance_within_rank=int(var_rank),
            correlation_within_rank=int(corr_rank),
            pair_ordering=PAIR_ORDERING,
            lag_convention=LAG_CONVENTION,
            quarticity_definition=QUARTICITY_DEFINITION,
            window_mean=np.mean(history, axis=0),
        )
        self._raw_validity = None
        self._v_raw = None
        self._x_raw = None
        self._R_raw = None
        self._H_raw = None
        return self

    def forecast(self) -> CovarianceForecast:
        """Return the one-step HARQ-DRD forecast, repairing only when required."""
        state = _require_fit(self._fit)
        v_raw, x_raw = _raw_components(state, self._origin_predictors())
        H_final, validity, R_raw, H_raw = headline_repaired_forecast(
            v_raw, x_raw, state.n_assets, state.window_mean, fallback_name="HARQ-DRD"
        )
        self._v_raw = np.array(v_raw, dtype=float, copy=True)
        self._x_raw = np.array(x_raw, dtype=float, copy=True)
        self._R_raw = np.array(R_raw, dtype=float, copy=True)
        self._H_raw = None if H_raw is None else np.array(H_raw, dtype=float, copy=True)
        self._raw_validity = validity
        return pack_forecast(H_final, self.identity)

    def update_window(
        self,
        realized_covariances: ArrayLike,
        realized_quarticity: ArrayLike,
    ) -> HARQDRDRealizedCovariance:
        """Rebuild origin HARQ features and the current-window fallback mean.

        Coefficients remain the values stored at the last ``fit``. Origin-day
        per-asset RQ is taken from the current window. Aggregate GW RQ is not
        used.
        """
        state = _require_fit(self._fit)
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _n_cols = history.shape
        if n_assets != state.n_assets:
            raise InvalidModelInputError(
                "HARQ-DRD update_window N must equal the fitted width "
                f"{state.n_assets}; got N={n_assets}"
            )
        if n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError(
                "HARQ-DRD update_window requires T >= "
                f"{MIN_WINDOW_LENGTH}; got T={n_times}"
            )
        quarticity = as_realized_quarticity_history(
            realized_quarticity, n_times=n_times, n_assets=n_assets
        )
        # Rebuild origin panels from the current windows only.
        variances, correlations = _drd_panel(history)
        self._variance_panel = variances
        self._pair_panel = pair_panel_from_correlations(correlations)
        self._quarticity_panel = quarticity
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
        """HARQ predictors at the supplied origin. Target-day values are unused."""
        _require_fit(self._fit)
        if (
            self._variance_panel is None
            or self._pair_panel is None
            or self._quarticity_panel is None
        ):
            raise InvalidModelInputError(
                f"{MODEL_NAME} has no stored origin panels. Call fit before forecast."
            )
        # Origin slices. The target covariance and RQ are not in these arrays.
        variance_z = harq_forecast_predictors(
            self._variance_panel, self._quarticity_panel
        ).T
        pair_z = forecast_predictors(self._pair_panel).T
        return variance_z, pair_z


def as_realized_quarticity_history(
    realized_quarticity: ArrayLike,
    *,
    n_times: int,
    n_assets: int,
) -> NDArray[np.floating]:
    """Copy a finite nonnegative ``(T, N)`` RQ window. Inputs are not mutated.

    Zero entries are allowed. Negative and non-finite entries are rejected.
    The array is not annualized, winsorized, clipped, or standardized.
    """
    array = np.array(realized_quarticity, dtype=float, copy=True)
    if array.ndim != 2:
        raise InvalidModelInputError(
            "realized_quarticity must have shape (T, N); "
            f"got {array.shape}"
        )
    n_rows, n_cols = array.shape
    if n_rows != n_times:
        raise InvalidModelInputError(
            "realized_quarticity T must match realized_covariances T; "
            f"got RQ T={n_rows} and covariance T={n_times}"
        )
    if n_cols != n_assets:
        raise InvalidModelInputError(
            "realized_quarticity N must match realized_covariances N; "
            f"got RQ N={n_cols} and covariance N={n_assets}"
        )
    if not np.isfinite(array).all():
        raise InvalidModelInputError(
            "realized_quarticity must be finite (NaN and inf are rejected)"
        )
    if np.any(array < 0.0):
        raise InvalidModelInputError(
            "realized_quarticity must be nonnegative. Negative RQ is rejected. "
            "Zero is allowed."
        )
    return array


def harq_variance_predictors(
    variances: NDArray[np.floating],
    quarticity: NDArray[np.floating],
    index: int,
) -> NDArray[np.floating]:
    """Return four HARQ variance predictors at response ``index``.

    Columns are daily ``v``, ``sqrt(RQ)*v``, weekly mean of four observations,
    and monthly mean of seventeen observations. Quarticity is per asset.
    """
    if index < RESPONSE_START:
        raise InvalidModelInputError(
            f"HARQ response index must be at least {RESPONSE_START}; got {index}"
        )
    daily = variances[index - 1]
    interaction = np.sqrt(quarticity[index - 1]) * daily
    weekly = variances[index - 5 : index - 1].mean(axis=0)
    monthly = variances[index - 22 : index - 5].mean(axis=0)
    return np.stack((daily, interaction, weekly, monthly), axis=0)


def harq_forecast_predictors(
    variances: NDArray[np.floating],
    quarticity: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return origin HARQ variance predictors for the target after the window."""
    n_times = int(variances.shape[0])
    if n_times < MIN_WINDOW_LENGTH:
        raise InvalidModelInputError(
            "HARQ-DRD forecast predictors require T >= "
            f"{MIN_WINDOW_LENGTH}; got T={n_times}"
        )
    daily = variances[n_times - 1]
    interaction = np.sqrt(quarticity[n_times - 1]) * daily
    weekly = variances[n_times - 5 : n_times - 1].mean(axis=0)
    monthly = variances[n_times - 22 : n_times - 5].mean(axis=0)
    return np.stack((daily, interaction, weekly, monthly), axis=0)


def harq_variance_design(
    variances: NDArray[np.floating],
    quarticity: NDArray[np.floating],
    response_index: NDArray[np.intp],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``y`` of shape ``(N, K)`` and ``X`` of shape ``(N, K, 4)``."""
    n_assets = int(variances.shape[1])
    n_dates = int(response_index.size)
    y = np.empty((n_assets, n_dates), dtype=float)
    design = np.empty((n_assets, n_dates, HARQ_VARIANCE_SLOPES), dtype=float)
    for position, time_index in enumerate(response_index):
        # Predictors for this response date. Do not read before index 0.
        predictors = harq_variance_predictors(variances, quarticity, int(time_index))
        y[:, position] = variances[time_index]
        design[:, position, :] = np.asarray(predictors.T, dtype=float)
    return y, design


def _raw_components(
    state: HARQDRDFitState, predictors: tuple[NDArray[np.floating], NDArray[np.floating]]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply fitted HARQ variance and HAR correlation maps at the origin."""
    variance_z, pair_z = predictors
    v_raw = state.alpha_Q + variance_z @ state.beta_Q
    x_raw = state.alpha_R + pair_z @ state.beta_R
    return v_raw, x_raw


def _require_fit(state: HARQDRDFitState | None) -> HARQDRDFitState:
    """Reject a method call issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError(
            f"{MODEL_NAME} has no fitted origin-window state. Call fit before forecast."
        )
    return state
