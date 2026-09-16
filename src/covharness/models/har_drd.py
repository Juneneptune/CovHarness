"""HAR-DRD one-day-ahead realized-covariance baseline.

The headline specification is Zhang-style non-overlapping HAR on the
Oh-Patton DRD split. Variances have asset-specific intercepts and three
shared slopes. Correlations have pair-specific intercepts and three
shared slopes. Estimation uses the within transformation, not dummy
columns. The headline model is in levels. There is no log-variance
transform, Fisher transform, ridge, graph term, or HARQ term.

Pair vectorization uses the repository's existing unique-pair order.
That order is the strict upper triangle ``i < j`` in the supplied asset
order, matching ``np.triu_indices(N, k=1)`` in Epps and GW.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.losses.contracts import (
    SYMMETRY_ATOL,
    InvalidCovarianceMatrixError,
    require_symmetric,
)
from covharness.losses.localization import implied_correlation
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
from covharness.models.exceptions import InvalidModelForecastError, InvalidModelInputError

MODEL_NAME = "har_drd"
PAIR_ORDERING = "strict_upper_triangle_i_lt_j"
LAG_CONVENTION = "nonoverlapping_1_4_17"
REPAIR_METHOD = "estimation_window_mean"
RESPONSE_START = 22
WEEKLY_WIDTH = 4
MONTHLY_WIDTH = 17
MIN_WINDOW_LENGTH = RESPONSE_START + 1
HAR_SLOPES = 3


@dataclass(frozen=True)
class HARDRDRawValidity:
    """Reportable raw-forecast failure flags. Repair is never silent."""

    raw_nonfinite_variance: bool
    raw_nonpositive_variance: bool
    raw_nonfinite_correlation: bool
    raw_correlation_out_of_bounds: bool
    raw_correlation_not_pd: bool
    raw_covariance_not_pd: bool
    repaired: bool
    repair_method: str | None


@dataclass(frozen=True)
class HARDRDFitState:
    """Auditable HAR-DRD coefficients and window metadata."""

    alpha_D: NDArray[np.floating]
    beta_D: NDArray[np.floating]
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


class HARDRDRealizedCovariance(RealizedCovarianceModel):
    """One-day-ahead HAR-DRD of a realized-covariance window."""

    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.WINDOW_STATE,
        fit_input=FitInput.REALIZED_COVARIANCE,
        update_observable=UpdateObservable.REALIZED_COVARIANCE_WINDOW,
    )

    def __init__(self) -> None:
        self._fit: HARDRDFitState | None = None
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
            "lag_convention": LAG_CONVENTION,
            "pair_ordering": PAIR_ORDERING,
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
    def fit_state(self) -> HARDRDFitState:
        """Return the stored coefficients. ``fit`` must have been called."""
        return _require_fit(self._fit)

    @property
    def raw_validity(self) -> HARDRDRawValidity | None:
        """Return flags from the most recent ``forecast`` call, if any."""
        return self._raw_validity

    def fit(self, realized_covariances: ArrayLike) -> HARDRDRealizedCovariance:
        """Estimate HAR-DRD on the origin window. Inputs are not mutated."""
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _ = history.shape
        if n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError(
                "HAR-DRD requires at least "
                f"{MIN_WINDOW_LENGTH} origin-window matrices so that "
                f"response dates {RESPONSE_START},...,T-1 exist; got T={n_times}"
            )
        if n_assets < 2:
            raise InvalidModelInputError(
                "HAR-DRD requires at least two assets so that a correlation "
                f"HAR is defined; got N={n_assets}"
            )
        # Require a strictly positive diagonal on every slice.
        variances, correlations = _drd_panel(history)
        n_pairs = pair_count(n_assets)
        pair_panel = pair_panel_from_correlations(correlations)
        response_index = har_regression_indices(n_times)
        n_dates = int(response_index.size)

        variance_y, variance_x = har_response_design(variances, response_index)
        alpha_D, beta_D, var_rank, var_width = fixed_effects_shared_slopes(
            variance_y, variance_x
        )
        correlation_y, correlation_x = har_response_design(pair_panel, response_index)
        alpha_R, beta_R, corr_rank, corr_width = fixed_effects_shared_slopes(
            correlation_y, correlation_x
        )
        if var_width != HAR_SLOPES or corr_width != HAR_SLOPES:
            raise InvalidModelForecastError(
                "HAR-DRD within designs must have exactly three slope columns"
            )
        self._variance_panel = variances
        self._pair_panel = pair_panel
        self._fit = HARDRDFitState(
            alpha_D=np.array(alpha_D, dtype=float, copy=True),
            beta_D=np.array(beta_D, dtype=float, copy=True),
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
            window_mean=np.mean(history, axis=0),
        )
        self._raw_validity = None
        self._v_raw = None
        self._x_raw = None
        self._R_raw = None
        self._H_raw = None
        return self

    def forecast(self) -> CovarianceForecast:
        """Return the one-step HAR-DRD forecast, repairing only when required."""
        state = _require_fit(self._fit)
        v_raw, x_raw = _raw_components(state, self._origin_predictors())
        H_final, validity, R_raw, H_raw = headline_repaired_forecast(
            v_raw, x_raw, state.n_assets, state.window_mean, fallback_name="HAR-DRD"
        )
        self._v_raw = np.array(v_raw, dtype=float, copy=True)
        self._x_raw = np.array(x_raw, dtype=float, copy=True)
        self._R_raw = np.array(R_raw, dtype=float, copy=True)
        self._H_raw = None if H_raw is None else np.array(H_raw, dtype=float, copy=True)
        self._raw_validity = validity
        return pack_forecast(H_final, self.identity)

    def update_window(
        self, realized_covariances: ArrayLike
    ) -> HARDRDRealizedCovariance:
        """Rebuild origin HAR features and the current-window fallback mean.

        Coefficients remain the values stored at the last ``fit``. This method
        does not re-estimate intercepts or slopes.
        """
        state = _require_fit(self._fit)
        history = as_realized_covariance_history(realized_covariances)
        n_times, n_assets, _n_cols = history.shape
        if n_assets != state.n_assets:
            raise InvalidModelInputError(
                "HAR-DRD update_window N must equal the fitted width "
                f"{state.n_assets}; got N={n_assets}"
            )
        if n_times < MIN_WINDOW_LENGTH:
            raise InvalidModelInputError(
                "HAR-DRD update_window requires T >= "
                f"{MIN_WINDOW_LENGTH}; got T={n_times}"
            )
        # Rebuild origin panels from the current window only.
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


def unique_pair_indices(n_assets: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return strict upper-triangle indices ``i < j`` in asset order."""
    if n_assets < 2:
        raise InvalidModelInputError("unique pairs require at least two assets")
    return np.triu_indices(n_assets, k=1)


def pair_count(n_assets: int) -> int:
    """Return ``P = N(N-1)/2``."""
    rows, _cols = unique_pair_indices(n_assets)
    return int(rows.size)


def pair_vector_from_correlation(correlation: NDArray[np.floating]) -> NDArray[np.floating]:
    """Stack unique off-diagonal correlations in the frozen pair order."""
    n_assets = int(correlation.shape[0])
    rows, cols = unique_pair_indices(n_assets)
    return np.asarray(correlation[rows, cols], dtype=float)


def correlation_from_pair_vector(
    pair_values: ArrayLike, n_assets: int
) -> NDArray[np.floating]:
    """Rebuild a symmetric correlation matrix from the frozen pair order."""
    values = np.asarray(pair_values, dtype=float).reshape(-1)
    rows, cols = unique_pair_indices(n_assets)
    if values.size != rows.size:
        raise InvalidModelInputError(
            "pair vector length must equal N(N-1)/2; "
            f"got {values.size} for N={n_assets}"
        )
    correlation = np.eye(n_assets, dtype=float)
    # Fill both off-diagonal triangles from the same vector.
    correlation[rows, cols] = values
    correlation[cols, rows] = values
    return correlation


def drd_components(matrix: ArrayLike) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``(v, R)`` from one covariance. The input is not mutated."""
    array = np.array(matrix, dtype=float, copy=True)
    variances = np.diag(array).astype(float, copy=True)
    if np.any(variances <= 0.0):
        raise InvalidModelInputError(
            "HAR-DRD realized variances must be strictly positive. "
            "A zero diagonal leaves R undefined. The matrix is not repaired."
        )
    try:
        correlation = implied_correlation(array, "HAR-DRD S")
    except InvalidCovarianceMatrixError as exc:
        raise InvalidModelInputError(str(exc)) from exc
    return variances, correlation


def covariance_from_drd(
    variances: ArrayLike, correlation: ArrayLike
) -> NDArray[np.floating]:
    """Return ``D R D`` with ``D = diag(sqrt(v))``."""
    v = np.asarray(variances, dtype=float).reshape(-1)
    corr = np.array(correlation, dtype=float, copy=True)
    if np.any(v <= 0.0) or (not np.isfinite(v).all()):
        raise InvalidModelInputError(
            "DRD reconstruction requires finite strictly positive variances"
        )
    scales = np.sqrt(v)
    return corr * np.outer(scales, scales)


def har_regression_indices(n_times: int) -> NDArray[np.intp]:
    """Return response indices ``22, ..., T-1``."""
    if n_times < MIN_WINDOW_LENGTH:
        raise InvalidModelInputError(
            "HAR-DRD regression dates require T >= "
            f"{MIN_WINDOW_LENGTH}; got T={n_times}"
        )
    return np.arange(RESPONSE_START, n_times, dtype=np.intp)


def har_predictors(values: NDArray[np.floating], index: int) -> NDArray[np.floating]:
    """Return daily, weekly, and monthly predictors at response ``index``.

    Weekly is ``values[index-5:index-1]`` (4 observations). Monthly is
    ``values[index-22:index-5]`` (17 observations). The blocks do not overlap.
    """
    if index < RESPONSE_START:
        raise InvalidModelInputError(
            f"HAR response index must be at least {RESPONSE_START}; got {index}"
        )
    daily = values[index - 1]
    weekly = values[index - 5 : index - 1].mean(axis=0)
    monthly = values[index - 22 : index - 5].mean(axis=0)
    return np.stack((daily, weekly, monthly), axis=0)


def forecast_predictors(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return origin predictors for the target immediately after the window.

    Daily is ``values[T-1]``. Weekly is ``values[T-5:T-1]``. Monthly is
    ``values[T-22:T-5]``.
    """
    n_times = int(values.shape[0])
    if n_times < MIN_WINDOW_LENGTH:
        raise InvalidModelInputError(
            "HAR-DRD forecast predictors require T >= "
            f"{MIN_WINDOW_LENGTH}; got T={n_times}"
        )
    daily = values[n_times - 1]
    weekly = values[n_times - 5 : n_times - 1].mean(axis=0)
    monthly = values[n_times - 22 : n_times - 5].mean(axis=0)
    return np.stack((daily, weekly, monthly), axis=0)


def har_response_design(
    panel: NDArray[np.floating], response_index: NDArray[np.intp]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``y`` of shape ``(G, K)`` and ``X`` of shape ``(G, K, 3)``."""
    n_groups = int(panel.shape[1])
    n_dates = int(response_index.size)
    y = np.empty((n_groups, n_dates), dtype=float)
    design = np.empty((n_groups, n_dates, HAR_SLOPES), dtype=float)
    for position, time_index in enumerate(response_index):
        # Predictors for this response date. Do not read before index 0.
        predictors = har_predictors(panel, int(time_index))
        y[:, position] = panel[time_index]
        design[:, position, :] = np.asarray(predictors.T, dtype=float)
    return y, design


def fixed_effects_shared_slopes(
    response: NDArray[np.floating],
    design: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], int, int]:
    """Estimate shared slopes by the within transformation.

    ``response`` has shape ``(G, K)``. ``design`` has shape ``(G, K, S)``.
    Group intercepts are recovered from group means after the slopes.
    The demeaned design has ``S`` columns. Dummy intercept columns are
    not built. HAR-DRD uses ``S=3``. HARQ-DRD variance uses ``S=4``.
    """
    if response.ndim != 2:
        raise InvalidModelInputError("fixed-effects response must have shape (G, K)")
    if design.ndim != 3 or design.shape[:2] != response.shape:
        raise InvalidModelInputError(
            "fixed-effects design must have shape (G, K, S) aligned with y"
        )
    n_groups, n_dates, n_slopes = design.shape
    if n_slopes < 1:
        raise InvalidModelInputError("HAR within design must have at least one slope column")
    if n_groups < 1 or n_dates < 1:
        raise InvalidModelInputError("fixed-effects design must be non-empty")
    # Demean within groups. Do not make dummy columns.
    y_bar = response.mean(axis=1, keepdims=True)
    x_bar = design.mean(axis=1, keepdims=True)
    y_tilde = (response - y_bar).reshape(n_groups * n_dates)
    x_tilde = (design - x_bar).reshape(n_groups * n_dates, n_slopes)
    n_columns = int(x_tilde.shape[1])
    beta, _residuals, rank_value, _singular = np.linalg.lstsq(x_tilde, y_tilde, rcond=None)
    rank = int(rank_value)
    intercepts = y_bar[:, 0] - x_bar[:, 0, :] @ beta
    return intercepts, beta, rank, n_columns


def dummy_fixed_effects_shared_slopes(
    response: NDArray[np.floating],
    design: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Reference dummy-variable OLS for small-N equivalence tests only.

    This helper is not used by ``HARDRDRealizedCovariance``. It exists so
    tests can compare the within estimator with an explicit intercept
    dummy design at small N.
    """
    n_groups, n_dates, n_slopes = design.shape
    y_stack = response.reshape(n_groups * n_dates)
    dummy = np.kron(np.eye(n_groups), np.ones((n_dates, 1)))
    x_stack = design.reshape(n_groups * n_dates, n_slopes)
    stacked = np.hstack((dummy, x_stack))
    coef, _residuals, _rank, _singular = np.linalg.lstsq(stacked, y_stack, rcond=None)
    return coef[:n_groups], coef[n_groups:]


def raw_forecast_validity(
    variances: NDArray[np.floating],
    pair_values: NDArray[np.floating],
    correlation: NDArray[np.floating],
    covariance: NDArray[np.floating] | None,
) -> HARDRDRawValidity:
    """Evaluate the frozen raw-validity screen. No matrix is repaired here."""
    nonfinite_v = bool(not np.isfinite(variances).all())
    nonpositive_v = bool(np.any(variances <= 0.0) or nonfinite_v)
    nonfinite_x = bool(not np.isfinite(pair_values).all())
    out_of_bounds = bool(nonfinite_x or np.any(np.abs(pair_values) > 1.0))
    corr_not_pd = not _strict_pd(correlation)
    cov_not_pd = covariance is None or (not _strict_pd(covariance))
    valid = not (
        nonfinite_v
        or nonpositive_v
        or nonfinite_x
        or out_of_bounds
        or corr_not_pd
        or cov_not_pd
    )
    if valid:
        return HARDRDRawValidity(
            raw_nonfinite_variance=False,
            raw_nonpositive_variance=False,
            raw_nonfinite_correlation=False,
            raw_correlation_out_of_bounds=False,
            raw_correlation_not_pd=False,
            raw_covariance_not_pd=False,
            repaired=False,
            repair_method=None,
        )
    return HARDRDRawValidity(
        raw_nonfinite_variance=nonfinite_v,
        raw_nonpositive_variance=nonpositive_v,
        raw_nonfinite_correlation=nonfinite_x,
        raw_correlation_out_of_bounds=out_of_bounds,
        raw_correlation_not_pd=corr_not_pd,
        raw_covariance_not_pd=cov_not_pd,
        repaired=True,
        repair_method=REPAIR_METHOD,
    )


def pair_panel_from_correlations(correlations: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return a ``(T, P)`` unique-pair panel from a ``(T, N, N)`` correlation cube."""
    n_times, n_assets, _ = correlations.shape
    n_pairs = pair_count(n_assets)
    panel = np.empty((n_times, n_pairs), dtype=float)
    for time_index in range(n_times):
        panel[time_index] = pair_vector_from_correlation(correlations[time_index])
    return panel


def _drd_panel(
    history: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Decompose each origin-window matrix. Inputs are not mutated."""
    n_times, n_assets, _ = history.shape
    variances = np.empty((n_times, n_assets), dtype=float)
    correlations = np.empty((n_times, n_assets, n_assets), dtype=float)
    for time_index in range(n_times):
        variances[time_index], correlations[time_index] = drd_components(
            history[time_index]
        )
    return variances, correlations


def _raw_components(
    state: HARDRDFitState, predictors: tuple[NDArray[np.floating], NDArray[np.floating]]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply fitted HAR maps to origin variance and correlation predictors."""
    variance_z, pair_z = predictors
    v_raw = state.alpha_D + variance_z @ state.beta_D
    x_raw = state.alpha_R + pair_z @ state.beta_R
    return v_raw, x_raw


def _raw_covariance(
    variances: NDArray[np.floating], correlation: NDArray[np.floating]
) -> NDArray[np.floating] | None:
    """Build ``H_raw`` when variances allow a real diagonal scale."""
    if (not np.isfinite(variances).all()) or np.any(variances <= 0.0):
        return None
    if not np.isfinite(correlation).all():
        return None
    return covariance_from_drd(variances, correlation)


def headline_repaired_forecast(
    variances: NDArray[np.floating],
    pair_values: NDArray[np.floating],
    n_assets: int,
    window_mean: NDArray[np.floating],
    fallback_name: str = "HAR-DRD",
) -> tuple[
    NDArray[np.floating],
    HARDRDRawValidity,
    NDArray[np.floating],
    NDArray[np.floating] | None,
]:
    """Assemble ``H_raw`` and apply the frozen BPQ estimation-window-mean filter."""
    correlation = correlation_from_pair_vector(pair_values, n_assets)
    covariance = _raw_covariance(variances, correlation)
    validity = raw_forecast_validity(variances, pair_values, correlation, covariance)
    if validity.repaired:
        headline = _require_pd_fallback(window_mean, name=fallback_name)
    else:
        assert covariance is not None
        headline = covariance
    return headline, validity, correlation, covariance


def _require_pd_fallback(
    window_mean: NDArray[np.floating], name: str = "HAR-DRD"
) -> NDArray[np.floating]:
    """Return the window mean if it is strictly PD. Do not repair again."""
    fallback = np.array(window_mean, dtype=float, copy=True)
    label = f"{name} estimation-window mean"
    if not np.isfinite(fallback).all():
        raise InvalidModelForecastError(
            f"{label} is not finite. No second repair was applied."
        )
    try:
        require_symmetric(fallback, label, atol=SYMMETRY_ATOL)
    except InvalidCovarianceMatrixError as exc:
        raise InvalidModelForecastError(
            f"{label} is not symmetric. No second repair was applied."
        ) from exc
    if not _strict_pd(fallback):
        raise InvalidModelForecastError(
            f"{label} is not strictly positive definite. "
            "No second repair was applied."
        )
    return fallback


def _strict_pd(matrix: NDArray[np.floating]) -> bool:
    """True iff the matrix is finite, square, and Cholesky succeeds."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.isfinite(matrix).all():
        return False
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def _require_fit(state: HARDRDFitState | None) -> HARDRDFitState:
    """Reject a method call issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError(
            f"{MODEL_NAME} has no fitted origin-window state. Call fit before forecast."
        )
    return state
