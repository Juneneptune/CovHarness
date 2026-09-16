"""Original Engle (2002) DCC and DCC-NL on daily-return windows.

Headline models use original DCC, not cDCC. Stage-one volatilities are
independent ZeroMean Gaussian GARCH(1,1) fits on in-window demeaned
returns. The second stage estimates ``(alpha, beta)`` by all-pairs
bivariate composite Gaussian quasi-likelihood.

DCC-NL changes only the intercept ``C``. It applies the pinned
analytical Ledoit-Wolf 2020 estimator to standardized residuals with
``k=0`` and divisor ``T``, then renormalizes the diagonal. It does not
shrink ``H_t`` or ``R_t`` after the recursion.

Indexing. Window rows are ``t = 0, ..., T-1``. ``Q_t`` is the DCC state
used to form ``R_t`` that scores the observed standardized residual
``s_t``. ``Q_0 = C_star``. After ``s_t`` is observed,

    Q_{t+1} = (1-alpha-beta) C_star + alpha s_t s_t' + beta Q_t.

Stored ``current_q`` is ``Q_{T-1}``, the state for the last observed
return. ``forecast()`` forms ``Q_{T|T-1}`` from that state and does not
mutate it. ``update`` advances the observed-return state by one day
without re-estimating parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import metadata

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from covharness.losses.contracts import (
    SYMMETRY_ATOL,
    InvalidCovarianceMatrixError,
    require_symmetric,
)
from covharness.models.base import (
    CovarianceForecast,
    CovarianceModel,
    ModelIdentity,
    as_daily_return_history,
    pack_forecast,
)
from covharness.models.capabilities import (
    FitInput,
    ModelCapabilities,
    RollingCadence,
    UpdateObservable,
)
from covharness.models.exceptions import InvalidModelForecastError, InvalidModelInputError

DCC_MODEL_NAME = "dcc"
DCC_NL_MODEL_NAME = "dcc_nl"
GARCH_MEAN_RULE = "fit_window_mean_then_zero_mean"
GARCH_BACKCAST_RULE = "mean_squared_centered_residual_divisor_T"
GARCH_SPECIFICATION = "zero_mean_gaussian_garch_1_1"
ARCH_PACKAGE = "arch"
PAIR_ORDERING = "strict_upper_triangle_i_lt_j"
PAIR_LIKELIHOOD = "all_pairs_composite"
OPTIMIZER_NAME = "SLSQP"
DCC_ALPHA0 = 0.05
DCC_BETA0 = 0.90
TARGET_DIVISOR = "T"
SAMPLE_TARGET_TYPE = "sample_standardized_residual_correlation"
NL_TARGET_TYPE = "analytical_nonlinear_standardized_residual_correlation"
NL_REFERENCE_PACKAGE = "nonlinshrink"
NL_K = 0
# nonlinshrink.shrink_cov requires n_eff >= 12. With k=0, n_eff = T.
NL_MIN_T = 12
UNIT_DIAGONAL_ATOL = 1e-10


@dataclass(frozen=True)
class StageOneGARCHFit:
    """In-window ZeroMean GARCH(1,1) paths. Rows are observations ``t=0..T-1``."""

    fit_mean: NDArray[np.floating]
    omega: NDArray[np.floating]
    garch_alpha: NDArray[np.floating]
    garch_beta: NDArray[np.floating]
    garch_backcast: NDArray[np.floating]
    residuals: NDArray[np.floating]
    variances: NDArray[np.floating]
    standardized: NDArray[np.floating]
    garch_success: NDArray[np.bool_]
    garch_nfev: NDArray[np.int_]
    n_observations: int
    n_assets: int
    arch_version: str


@dataclass(frozen=True)
class DCCFitState:
    """Observed-origin DCC filter state. ``current_q`` scores the last return."""

    n_observations: int
    n_assets: int
    n_pairs: int
    fit_mean: NDArray[np.floating]
    omega: NDArray[np.floating]
    garch_alpha: NDArray[np.floating]
    garch_beta: NDArray[np.floating]
    garch_backcast: NDArray[np.floating]
    current_variance: NDArray[np.floating]
    current_residual: NDArray[np.floating]
    current_standardized: NDArray[np.floating]
    dcc_alpha: float
    dcc_beta: float
    target_c: NDArray[np.floating]
    current_q: NDArray[np.floating]
    garch_mean_rule: str
    garch_backcast_rule: str
    garch_specification: str
    arch_package: str
    arch_version: str
    target_type: str
    target_divisor: str
    pair_likelihood_type: str
    pair_ordering: str
    optimizer: str
    optimizer_start: tuple[float, float]
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_objective: float
    optimizer_nfev: int
    garch_success: NDArray[np.bool_]
    garch_nfev: NDArray[np.int_]
    nonlinear_reference_package: str | None
    nonlinear_reference_version: str | None
    nonlinear_k: int | None


class _EngleDCCBase(CovarianceModel):
    """Shared original-DCC engine. Subclasses supply only the intercept ``C``."""

    model_name: str
    target_type: str
    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.RECURSIVE_STATE,
        fit_input=FitInput.DAILY_RETURN,
        update_observable=UpdateObservable.DAILY_RETURN,
    )

    def __init__(self) -> None:
        self._fit: DCCFitState | None = None

    @property
    def identity(self) -> ModelIdentity:
        configuration: dict[str, object] = {
            "dcc_variant": "engle_2002_original",
            "garch_mean_rule": GARCH_MEAN_RULE,
            "garch_backcast_rule": GARCH_BACKCAST_RULE,
            "garch_specification": GARCH_SPECIFICATION,
            "pair_likelihood_type": PAIR_LIKELIHOOD,
            "pair_ordering": PAIR_ORDERING,
            "optimizer": OPTIMIZER_NAME,
            "optimizer_start": (DCC_ALPHA0, DCC_BETA0),
            "target_type": self.target_type,
            "target_divisor": TARGET_DIVISOR,
        }
        if self._fit is not None:
            configuration.update(_identity_from_fit(self._fit))
        return ModelIdentity(name=self.model_name, configuration=configuration)

    @property
    def fit_state(self) -> DCCFitState:
        """Return the stored origin-window filter state."""
        return _require_dcc_fit(self._fit, self.model_name)

    def fit(self, returns: ArrayLike) -> _EngleDCCBase:
        """Estimate GARCH and DCC parameters on the supplied window."""
        history = _dcc_return_window(returns)
        stage_one = fit_stage_one_garch(history)
        # Replay GARCH with frozen parameters so targeting and the filter share s_t.
        variances = garch_variance_paths(
            stage_one.residuals,
            stage_one.omega,
            stage_one.garch_alpha,
            stage_one.garch_beta,
            stage_one.garch_backcast,
        )
        standardized = standardized_residuals(stage_one.residuals, variances)
        target = self._build_target(standardized)
        dcc_alpha, dcc_beta, opt_audit = estimate_dcc_parameters(target, standardized)
        observed_q = dcc_observed_q_path(target, dcc_alpha, dcc_beta, standardized)[-1]
        self._fit = DCCFitState(
            n_observations=stage_one.n_observations,
            n_assets=stage_one.n_assets,
            n_pairs=pair_count(stage_one.n_assets),
            fit_mean=stage_one.fit_mean.copy(),
            omega=stage_one.omega.copy(),
            garch_alpha=stage_one.garch_alpha.copy(),
            garch_beta=stage_one.garch_beta.copy(),
            garch_backcast=stage_one.garch_backcast.copy(),
            current_variance=variances[-1].copy(),
            current_residual=stage_one.residuals[-1].copy(),
            current_standardized=standardized[-1].copy(),
            dcc_alpha=float(dcc_alpha),
            dcc_beta=float(dcc_beta),
            target_c=np.array(target, dtype=float, copy=True),
            current_q=np.array(observed_q, dtype=float, copy=True),
            garch_mean_rule=GARCH_MEAN_RULE,
            garch_backcast_rule=GARCH_BACKCAST_RULE,
            garch_specification=GARCH_SPECIFICATION,
            arch_package=ARCH_PACKAGE,
            arch_version=stage_one.arch_version,
            target_type=self.target_type,
            target_divisor=TARGET_DIVISOR,
            pair_likelihood_type=PAIR_LIKELIHOOD,
            pair_ordering=PAIR_ORDERING,
            optimizer=OPTIMIZER_NAME,
            optimizer_start=(DCC_ALPHA0, DCC_BETA0),
            optimizer_success=bool(opt_audit["success"]),
            optimizer_status=int(opt_audit["status"]),
            optimizer_message=str(opt_audit["message"]),
            optimizer_objective=float(opt_audit["objective"]),
            optimizer_nfev=int(opt_audit["nfev"]),
            garch_success=stage_one.garch_success.copy(),
            garch_nfev=stage_one.garch_nfev.copy(),
            nonlinear_reference_package=self._nl_package(),
            nonlinear_reference_version=self._nl_version(),
            nonlinear_k=self._nl_k(),
        )
        return self

    def forecast(self) -> CovarianceForecast:
        """Return ``H_{t+1|t}`` from the current observed-origin state.

        The stored filter is not mutated. Repeated calls before ``update``
        are identical.
        """
        state = _require_dcc_fit(self._fit, self.model_name)
        forecast_h, _h_next, _q_next = one_step_forecast_from_state(state)
        _reject_invalid_forecast(forecast_h, self.model_name)
        return pack_forecast(forecast_h, self.identity)

    def update(self, new_return: ArrayLike) -> _EngleDCCBase:
        """Advance GARCH and DCC states by one observed return. No refit."""
        state = _require_dcc_fit(self._fit, self.model_name)
        observed = _as_new_return(new_return, state.n_assets)
        h_new, q_new = next_conditional_states(state)
        # Center with the frozen fit-window mean.
        epsilon_new = observed - state.fit_mean
        if np.any(h_new <= 0.0) or not np.isfinite(h_new).all():
            raise InvalidModelForecastError(
                f"{self.model_name} update produced a nonpositive GARCH variance"
            )
        s_new = epsilon_new / np.sqrt(h_new)
        if not np.isfinite(s_new).all():
            raise InvalidModelForecastError(
                f"{self.model_name} update produced a nonfinite standardized residual"
            )
        _reject_invalid_q(q_new, f"{self.model_name} update Q")
        self._fit = replace(
            state,
            current_variance=np.asarray(h_new, dtype=float),
            current_residual=np.asarray(epsilon_new, dtype=float),
            current_standardized=np.asarray(s_new, dtype=float),
            current_q=np.array(q_new, dtype=float, copy=True),
        )
        return self

    def _build_target(self, standardized: NDArray[np.floating]) -> NDArray[np.floating]:
        raise NotImplementedError

    def _nl_package(self) -> str | None:
        return None

    def _nl_version(self) -> str | None:
        return None

    def _nl_k(self) -> int | None:
        return None


class DCCCovariance(_EngleDCCBase):
    """Original DCC with sample-correlation targeting of standardized residuals."""

    model_name = DCC_MODEL_NAME
    target_type = SAMPLE_TARGET_TYPE

    def _build_target(self, standardized: NDArray[np.floating]) -> NDArray[np.floating]:
        return sample_standardized_residual_correlation(standardized)


class DCCNonlinearCovariance(_EngleDCCBase):
    """Original DCC with analytical LW 2020 targeting of standardized residuals."""

    model_name = DCC_NL_MODEL_NAME
    target_type = NL_TARGET_TYPE

    def _build_target(self, standardized: NDArray[np.floating]) -> NDArray[np.floating]:
        return nonlinear_standardized_residual_correlation(standardized)

    def _nl_package(self) -> str | None:
        return NL_REFERENCE_PACKAGE

    def _nl_version(self) -> str | None:
        return metadata.version(NL_REFERENCE_PACKAGE)

    def _nl_k(self) -> int | None:
        return NL_K


def _dcc_return_window(returns: ArrayLike) -> NDArray[np.floating]:
    """Copy a finite ``(T, N)`` window and require at least two assets."""
    history = as_daily_return_history(returns)
    if history.shape[1] < 2:
        raise InvalidModelInputError(
            f"DCC requires at least two assets; got N={history.shape[1]}"
        )
    return history


def fit_stage_one_garch(returns: NDArray[np.floating]) -> StageOneGARCHFit:
    """Demean the window and fit N independent ZeroMean GARCH(1,1) models."""
    from arch.univariate import arch_model

    n_observations, n_assets = returns.shape
    fit_mean = returns.mean(axis=0)
    residuals = returns - fit_mean
    backcast = np.mean(residuals * residuals, axis=0)
    if np.any(backcast <= 0.0) or not np.isfinite(backcast).all():
        raise InvalidModelInputError(
            "GARCH backcast h_i,0 = mean(epsilon^2) must be finite and strictly positive"
        )
    omega = np.empty(n_assets, dtype=float)
    garch_alpha = np.empty(n_assets, dtype=float)
    garch_beta = np.empty(n_assets, dtype=float)
    variances = np.empty((n_observations, n_assets), dtype=float)
    success = np.empty(n_assets, dtype=bool)
    nfev = np.empty(n_assets, dtype=int)
    arch_version = metadata.version(ARCH_PACKAGE)
    # Fit each asset independently. Do not drop failures.
    for asset in range(n_assets):
        series = residuals[:, asset]
        try:
            model = arch_model(
                series,
                mean="Zero",
                vol="GARCH",
                p=1,
                o=0,
                q=1,
                dist="normal",
                rescale=False,
            )
            result = model.fit(disp="off", backcast=float(backcast[asset]))
        except Exception as exc:
            raise InvalidModelForecastError(
                f"ZeroMean GARCH(1,1) failed for asset {asset}; {exc}"
            ) from exc
        opt = result.optimization_result
        if opt is None or not bool(opt.success):
            message = getattr(opt, "message", "no optimizer result")
            raise InvalidModelForecastError(
                f"ZeroMean GARCH(1,1) did not converge for asset {asset}; {message}"
            )
        try:
            omega_i = float(result.params["omega"])
            alpha_i = float(result.params["alpha[1]"])
            beta_i = float(result.params["beta[1]"])
        except Exception as exc:
            raise InvalidModelForecastError(
                f"ZeroMean GARCH(1,1) returned unreadable parameters for asset {asset}"
            ) from exc
        if not np.isfinite([omega_i, alpha_i, beta_i]).all():
            raise InvalidModelForecastError(
                f"ZeroMean GARCH(1,1) returned nonfinite parameters for asset {asset}"
            )
        if not (omega_i > 0.0 and alpha_i >= 0.0 and beta_i >= 0.0 and alpha_i + beta_i < 1.0):
            raise InvalidModelForecastError(
                "ZeroMean GARCH(1,1) parameters violate omega>0, a>=0, b>=0, "
                f"a+b<1 for asset {asset}; got omega={omega_i}, a={alpha_i}, b={beta_i}"
            )
        # arch reports conditional standard deviations, not variances.
        std = np.asarray(result.conditional_volatility, dtype=float)
        variance = std * std
        if variance.shape != (n_observations,) or np.any(variance <= 0.0) or not np.isfinite(variance).all():
            raise InvalidModelForecastError(
                f"ZeroMean GARCH(1,1) produced an invalid variance path for asset {asset}"
            )
        omega[asset] = omega_i
        garch_alpha[asset] = alpha_i
        garch_beta[asset] = beta_i
        variances[:, asset] = variance
        success[asset] = True
        nfev[asset] = int(getattr(opt, "nfev", 0))
    standardized = standardized_residuals(residuals, variances)
    return StageOneGARCHFit(
        fit_mean=fit_mean,
        omega=omega,
        garch_alpha=garch_alpha,
        garch_beta=garch_beta,
        garch_backcast=backcast,
        residuals=residuals,
        variances=variances,
        standardized=standardized,
        garch_success=success,
        garch_nfev=nfev,
        n_observations=n_observations,
        n_assets=n_assets,
        arch_version=arch_version,
    )


def garch_variance_paths(
    residuals: NDArray[np.floating],
    omega: NDArray[np.floating],
    garch_alpha: NDArray[np.floating],
    garch_beta: NDArray[np.floating],
    backcast: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Replay GARCH(1,1) with ``h_0 = omega + (a+b) backcast``.

    ``residuals`` has shape ``(T, N)``. The first in-window variance uses
    the explicit sample-second-moment backcast in place of the missing
    presample residual and variance.
    """
    n_observations, n_assets = residuals.shape
    variances = np.empty((n_observations, n_assets), dtype=float)
    variances[0] = omega + (garch_alpha + garch_beta) * backcast
    for time_index in range(1, n_observations):
        lagged = residuals[time_index - 1]
        variances[time_index] = (
            omega
            + garch_alpha * lagged * lagged
            + garch_beta * variances[time_index - 1]
        )
    if np.any(variances <= 0.0) or not np.isfinite(variances).all():
        raise InvalidModelForecastError(
            "replayed GARCH(1,1) variances are not finite and strictly positive"
        )
    return variances


def standardized_residuals(
    residuals: NDArray[np.floating],
    variances: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return ``epsilon / sqrt(h)``. Never divide by ``h`` itself."""
    if np.any(variances <= 0.0):
        raise InvalidModelForecastError(
            "standardized residuals require strictly positive GARCH variances"
        )
    standardized = residuals / np.sqrt(variances)
    if not np.isfinite(standardized).all():
        raise InvalidModelForecastError("standardized residuals are not finite")
    return standardized


def pair_indices(n_assets: int) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Return the frozen unique-pair order ``i < j``."""
    return np.triu_indices(n_assets, k=1)


def pair_count(n_assets: int) -> int:
    """Return ``N(N-1)/2``."""
    return int(n_assets * (n_assets - 1) // 2)


def correlation_from_covariance(covariance: NDArray[np.floating]) -> NDArray[np.floating]:
    """Renormalize a covariance to a correlation. The matrix is not repaired."""
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0.0) or not np.isfinite(diagonal).all():
        raise InvalidModelForecastError(
            "DCC target covariance diagonal must be finite and strictly positive"
        )
    scales = np.sqrt(diagonal)
    correlation = covariance / np.outer(scales, scales)
    _reject_invalid_correlation(correlation, "DCC target correlation")
    return correlation


def sample_standardized_residual_correlation(
    standardized: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return ``C`` from ``C_tilde = S'std Sstd / T`` then diagonal renormalization.

    Standardized residuals are not demeaned again. The divisor is ``T``.
    """
    n_observations, n_assets = standardized.shape
    gram = standardized.T @ standardized / n_observations
    if n_assets > n_observations:
        raise InvalidModelForecastError(
            "plain DCC sample target is rank-deficient when N>T; "
            f"got T={n_observations}, N={n_assets}"
        )
    return correlation_from_covariance(gram)


def nonlinear_standardized_residual_correlation(
    standardized: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Analytical LW 2020 target on standardized residuals with ``k=0``."""
    import nonlinshrink as nls

    n_observations, n_assets = standardized.shape
    if n_observations < NL_MIN_T:
        raise InvalidModelInputError(
            "DCC-NL analytical targeting requires T >= "
            f"{NL_MIN_T} because nonlinshrink with k=0 rejects smaller n_eff; "
            f"got T={n_observations}"
        )
    try:
        shrunk = np.asarray(
            nls.shrink_cov(np.array(standardized, copy=True), k=NL_K),
            dtype=float,
        )
    except Exception as exc:
        raise InvalidModelForecastError(
            "DCC-NL analytical target failed in nonlinshrink; "
            f"{exc}"
        ) from exc
    if shrunk.shape != (n_assets, n_assets):
        raise InvalidModelForecastError(
            f"DCC-NL analytical target has shape {shrunk.shape}, expected {(n_assets, n_assets)}"
        )
    if not np.isfinite(shrunk).all():
        raise InvalidModelForecastError("DCC-NL analytical target is nonfinite")
    try:
        require_symmetric(shrunk, "DCC-NL analytical covariance", atol=SYMMETRY_ATOL)
    except InvalidCovarianceMatrixError as exc:
        raise InvalidModelForecastError(str(exc)) from exc
    _require_strictly_pd(shrunk, "DCC-NL analytical covariance")
    return correlation_from_covariance(shrunk)


def q_to_correlation(q_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return ``R = diag(Q)^{-1/2} Q diag(Q)^{-1/2}``. Rho is not clipped."""
    diagonal = np.diag(q_matrix)
    if np.any(diagonal <= 0.0) or not np.isfinite(diagonal).all():
        raise InvalidModelForecastError(
            "DCC Q diagonal must be finite and strictly positive before normalization"
        )
    scales = np.sqrt(diagonal)
    return q_matrix / np.outer(scales, scales)


def dcc_q_update(
    q_matrix: NDArray[np.floating],
    shock: NDArray[np.floating],
    target: NDArray[np.floating],
    alpha: float,
    beta: float,
) -> NDArray[np.floating]:
    """Return ``(1-alpha-beta) C + alpha s s' + beta Q``."""
    return (1.0 - alpha - beta) * target + alpha * np.outer(shock, shock) + beta * q_matrix


def dcc_observed_q_path(
    target: NDArray[np.floating],
    alpha: float,
    beta: float,
    standardized: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return ``Q_t`` for each observed residual, starting at ``Q_0 = C``.

    Output has shape ``(T, N, N)``. Slice ``t`` is the state that scores
    ``s_t``. The unused one-step-ahead ``Q_T`` is not stored.
    """
    n_observations, n_assets = standardized.shape
    path = np.empty((n_observations, n_assets, n_assets), dtype=float)
    q_state = np.array(target, dtype=float, copy=True)
    for time_index in range(n_observations):
        path[time_index] = q_state
        q_state = dcc_q_update(
            q_state, standardized[time_index], target, alpha, beta
        )
    return path


def bivariate_correlation_nll(
    rho: float,
    s_i: float,
    s_j: float,
) -> float:
    """Return one pair's Gaussian correlation negative log-likelihood term."""
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        return float("inf")
    one_minus = 1.0 - rho * rho
    quad = s_i * s_i - 2.0 * rho * s_i * s_j + s_j * s_j
    return 0.5 * (float(np.log(one_minus)) + quad / one_minus)


def all_pairs_composite_nll(
    alpha: float,
    beta: float,
    target: NDArray[np.floating],
    standardized: NDArray[np.floating],
) -> float:
    """Sum all-pairs bivariate correlation NLL over ``t=0..T-1``.

    Pair order is the strict upper triangle. The objective is invariant
    to a common permutation of asset columns and ``C_star``.
    """
    if not np.isfinite([alpha, beta]).all():
        return float("inf")
    if alpha < 0.0 or beta < 0.0 or alpha + beta > 1.0:
        return float("inf")
    n_observations, n_assets = standardized.shape
    rows, cols = pair_indices(n_assets)
    q_state = np.array(target, dtype=float, copy=True)
    intercept = (1.0 - alpha - beta) * target
    total = 0.0
    for time_index in range(n_observations):
        diagonal = np.diag(q_state)
        if np.any(diagonal <= 0.0) or not np.isfinite(q_state).all():
            return float("inf")
        scales = np.sqrt(diagonal)
        correlation = q_state / np.outer(scales, scales)
        rho = correlation[rows, cols]
        if np.any(~np.isfinite(rho)) or np.any(np.abs(rho) >= 1.0):
            return float("inf")
        shock = standardized[time_index]
        s_i = shock[rows]
        s_j = shock[cols]
        one_minus = 1.0 - rho * rho
        quad = s_i * s_i - 2.0 * rho * s_i * s_j + s_j * s_j
        contribution = 0.5 * np.sum(np.log(one_minus) + quad / one_minus)
        if not np.isfinite(contribution):
            return float("inf")
        total += float(contribution)
        if time_index < n_observations - 1:
            q_state = intercept + alpha * np.outer(shock, shock) + beta * q_state
    return float(total)


def estimate_dcc_parameters(
    target: NDArray[np.floating],
    standardized: NDArray[np.floating],
) -> tuple[float, float, dict[str, object]]:
    """Estimate ``(alpha, beta)`` by deterministic SLSQP on the all-pairs NLL."""

    def objective(parameters: NDArray[np.floating]) -> float:
        return all_pairs_composite_nll(
            float(parameters[0]),
            float(parameters[1]),
            target,
            standardized,
        )

    result = minimize(
        objective,
        x0=np.array([DCC_ALPHA0, DCC_BETA0], dtype=float),
        method=OPTIMIZER_NAME,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        constraints={"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1]},
    )
    alpha = float(result.x[0]) if result.x is not None and result.x.size == 2 else float("nan")
    beta = float(result.x[1]) if result.x is not None and result.x.size == 2 else float("nan")
    audit = {
        "success": bool(result.success),
        "status": int(getattr(result, "status", -1)),
        "message": str(getattr(result, "message", "")),
        "objective": float(result.fun) if np.isscalar(result.fun) else float("nan"),
        "nfev": int(getattr(result, "nfev", 0)),
    }
    accepted = (
        bool(result.success)
        and np.isfinite(audit["objective"])
        and np.isfinite(alpha)
        and np.isfinite(beta)
        and alpha >= 0.0
        and beta >= 0.0
        and alpha + beta < 1.0
    )
    if not accepted:
        raise InvalidModelForecastError(
            "DCC SLSQP failed the accepted stationarity domain "
            f"alpha>=0, beta>=0, alpha+beta<1; got alpha={alpha}, beta={beta}, "
            f"success={result.success}, message={result.message}"
        )
    return alpha, beta, audit


def next_conditional_states(
    state: DCCFitState,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``(h_{t+1|t}, Q_{t+1|t})`` from the current observed-origin state."""
    h_next = (
        state.omega
        + state.garch_alpha * state.current_residual * state.current_residual
        + state.garch_beta * state.current_variance
    )
    q_next = dcc_q_update(
        state.current_q,
        state.current_standardized,
        state.target_c,
        state.dcc_alpha,
        state.dcc_beta,
    )
    return h_next, q_next


def one_step_forecast_from_state(
    state: DCCFitState,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return ``(H_{t+1|t}, h_{t+1|t}, Q_{t+1|t})`` without mutating state."""
    h_next, q_next = next_conditional_states(state)
    if np.any(h_next <= 0.0) or not np.isfinite(h_next).all():
        raise InvalidModelForecastError(
            "one-step GARCH variances are not finite and strictly positive"
        )
    _reject_invalid_q(q_next, "one-step DCC Q")
    correlation = q_to_correlation(q_next)
    scale = np.sqrt(h_next)
    forecast_h = correlation * np.outer(scale, scale)
    return forecast_h, h_next, q_next


def _as_new_return(new_return: ArrayLike, n_assets: int) -> NDArray[np.floating]:
    """Copy a finite length-``N`` update vector."""
    array = np.array(new_return, dtype=float, copy=True)
    if array.ndim != 1 or array.shape[0] != n_assets:
        raise InvalidModelInputError(
            f"DCC update requires shape (N,) with N={n_assets}; got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise InvalidModelInputError("DCC update return must be finite")
    return array


def _reject_invalid_correlation(matrix: NDArray[np.floating], name: str) -> None:
    """Reject a nonfinite, asymmetric, non-unit-diagonal, or non-strictly-PD correlation."""
    if not np.isfinite(matrix).all():
        raise InvalidModelForecastError(f"{name} is nonfinite")
    try:
        require_symmetric(matrix, name, atol=SYMMETRY_ATOL)
    except InvalidCovarianceMatrixError as exc:
        raise InvalidModelForecastError(str(exc)) from exc
    diagonal_error = float(np.max(np.abs(np.diag(matrix) - 1.0)))
    if diagonal_error > UNIT_DIAGONAL_ATOL:
        raise InvalidModelForecastError(
            f"{name} must have unit diagonal; max |diag-1| is {diagonal_error}"
        )
    _require_strictly_pd(matrix, name)


def _reject_invalid_q(matrix: NDArray[np.floating], name: str) -> None:
    """Reject a nonfinite, asymmetric, or non-strictly-PD Q matrix."""
    if not np.isfinite(matrix).all():
        raise InvalidModelForecastError(f"{name} is nonfinite")
    try:
        require_symmetric(matrix, name, atol=SYMMETRY_ATOL)
    except InvalidCovarianceMatrixError as exc:
        raise InvalidModelForecastError(str(exc)) from exc
    _require_strictly_pd(matrix, name)


def _reject_invalid_forecast(matrix: NDArray[np.floating], name: str) -> None:
    """Reject a nonfinite, asymmetric, or non-strictly-PD forecast. No repair."""
    if not np.isfinite(matrix).all():
        raise InvalidModelForecastError(f"{name} forecast is nonfinite")
    try:
        require_symmetric(matrix, f"{name} forecast", atol=SYMMETRY_ATOL)
    except InvalidCovarianceMatrixError as exc:
        raise InvalidModelForecastError(str(exc)) from exc
    _require_strictly_pd(matrix, f"{name} forecast")


def _require_strictly_pd(matrix: NDArray[np.floating], name: str) -> None:
    """Require a successful Cholesky factor. Failure is not repaired."""
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise InvalidModelForecastError(
            f"{name} is not strictly positive definite"
        ) from exc


def _require_dcc_fit(state: DCCFitState | None, name: str) -> DCCFitState:
    """Reject a filter read issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError(
            f"{name} has no fitted origin-window state. Call fit before forecast or update."
        )
    return state


def _identity_from_fit(state: DCCFitState) -> dict[str, object]:
    """Copy auditable fitted scalars into the identity mapping."""
    configuration: dict[str, object] = {
        "n_observations": state.n_observations,
        "n_assets": state.n_assets,
        "n_pairs": state.n_pairs,
        "dcc_alpha": state.dcc_alpha,
        "dcc_beta": state.dcc_beta,
        "arch_package": state.arch_package,
        "arch_version": state.arch_version,
        "optimizer_success": state.optimizer_success,
        "optimizer_status": state.optimizer_status,
        "optimizer_message": state.optimizer_message,
        "optimizer_objective": state.optimizer_objective,
        "optimizer_nfev": state.optimizer_nfev,
    }
    if state.nonlinear_reference_package is not None:
        configuration.update(
            {
                "nonlinear_reference_package": state.nonlinear_reference_package,
                "nonlinear_reference_version": state.nonlinear_reference_version,
                "nonlinear_k": state.nonlinear_k,
            }
        )
    return configuration
