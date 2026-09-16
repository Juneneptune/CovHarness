"""Standalone Ledoit-Wolf linear and analytical nonlinear shrinkage.

Both estimators consume a caller-supplied daily-return window of shape
``(T, N)``. They do not consume realized-covariance histories and they
do not shrink a DCC targeting matrix. The covariance estimated on the
current window is the one-day-ahead forecast ``H_{t+1|t} = Sigma_hat_t``.

Returns are demeaned inside the supplied window. The shared sample
covariance is the centered Gram matrix with divisor ``T-1``.

Headline linear shrinkage is Ledoit-Wolf 2004b rotation-equivariant
shrinkage toward ``mu I``, with ``mu = tr(S)/N``. It is not the 2004a
Honey / equicorrelation-target estimator. The shrinkage intensity is
estimated. It is not a user-chosen rho.

Headline nonlinear shrinkage is the Ledoit-Wolf 2020 analytical
estimator, wrapped from the pinned ``nonlinshrink`` 0.7 port of that
formula. It is not QuEST and not QIS. The wrapper does not transcribe
the kernel or Hilbert formulas. Invalid reference output is rejected
rather than repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
    require_fitted_state,
)
from covharness.models.capabilities import (
    FitInput,
    ModelCapabilities,
    RollingCadence,
    UpdateObservable,
)
from covharness.models.exceptions import InvalidModelForecastError, InvalidModelInputError

LINEAR_MODEL_NAME = "lw_linear"
NONLINEAR_MODEL_NAME = "lw_nl"
COVARIANCE_DIVISOR = "T_minus_1"
LINEAR_ESTIMATOR = "ledoit_wolf_2004b_mu_identity"
NONLINEAR_ESTIMATOR = "ledoit_wolf_2020_analytical"
NONLINEAR_REFERENCE_PACKAGE = "nonlinshrink"
# nonlinshrink.shrink_cov requires n_eff >= 12 after subtracting k.
NONLINEAR_MIN_N_EFF = 12


@dataclass(frozen=True)
class CenteredReturnMoments:
    """In-window centered returns and the ``T-1`` sample covariance.

    ``returns`` has shape ``(T, N)``. ``centered`` has the same shape.
    ``mean`` has shape ``(N,)``. ``sample_covariance`` has shape
    ``(N, N)``. ``n_eff`` is ``T-1``.
    """

    returns: NDArray[np.floating]
    centered: NDArray[np.floating]
    mean: NDArray[np.floating]
    n_observations: int
    n_assets: int
    n_eff: int
    sample_covariance: NDArray[np.floating]
    sample_eigenvalues: NDArray[np.floating]
    sample_eigenvectors: NDArray[np.floating]


@dataclass(frozen=True)
class LedoitWolfLinearFitState:
    """Auditable Ledoit-Wolf 2004b fit metadata. Eigenvectors are not stored."""

    mean: NDArray[np.floating]
    n_observations: int
    n_assets: int
    n_eff: int
    centering: bool
    covariance_divisor: str
    rho: float
    mu: float
    sample_eigenvalues: NDArray[np.floating]


@dataclass(frozen=True)
class LedoitWolfNonlinearFitState:
    """Auditable Ledoit-Wolf 2020 analytical fit metadata."""

    mean: NDArray[np.floating]
    n_observations: int
    n_assets: int
    n_eff: int
    centering: bool
    covariance_divisor: str
    reference_package: str
    reference_version: str
    analytical_method: str
    sample_eigenvalues: NDArray[np.floating]
    shrunk_eigenvalues: NDArray[np.floating]


class LedoitWolfLinearCovariance(CovarianceModel):
    """One-day-ahead Ledoit-Wolf 2004b shrinkage of a daily-return window."""

    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.REFIT_HOLD,
        fit_input=FitInput.DAILY_RETURN,
        update_observable=UpdateObservable.NONE,
    )

    def __init__(self) -> None:
        self._fit: LedoitWolfLinearFitState | None = None
        self._state: NDArray[np.floating] | None = None

    @property
    def identity(self) -> ModelIdentity:
        configuration: dict[str, object] = {
            "estimator": LINEAR_ESTIMATOR,
            "centering": True,
            "covariance_divisor": COVARIANCE_DIVISOR,
            "target": "mu_identity",
        }
        if self._fit is not None:
            configuration.update(
                {
                    "n_observations": self._fit.n_observations,
                    "n_assets": self._fit.n_assets,
                    "n_eff": self._fit.n_eff,
                    "rho": self._fit.rho,
                    "mu": self._fit.mu,
                }
            )
        return ModelIdentity(name=LINEAR_MODEL_NAME, configuration=configuration)

    @property
    def fit_state(self) -> LedoitWolfLinearFitState:
        """Return stored linear-shrinkage metadata. ``fit`` must have been called."""
        return _require_linear_fit(self._fit)

    def fit(self, returns: ArrayLike) -> LedoitWolfLinearCovariance:
        """Estimate 2004b identity-target shrinkage. Inputs are not mutated."""
        moments = centered_return_moments(returns)
        # Match sklearn's 1/T Gram matrix to our S = Y'Y/(T-1).
        scaled_centered = np.sqrt(
            moments.n_observations / moments.n_eff
        ) * moments.centered
        rho = ledoit_wolf_2004b_shrinkage_coefficient(scaled_centered)
        mu = float(np.trace(moments.sample_covariance) / moments.n_assets)
        sigma = apply_linear_identity_shrinkage(
            moments.sample_covariance, rho=rho, mu=mu
        )
        self._fit = LedoitWolfLinearFitState(
            mean=moments.mean.copy(),
            n_observations=moments.n_observations,
            n_assets=moments.n_assets,
            n_eff=moments.n_eff,
            centering=True,
            covariance_divisor=COVARIANCE_DIVISOR,
            rho=float(rho),
            mu=mu,
            sample_eigenvalues=moments.sample_eigenvalues.copy(),
        )
        self._state = sigma
        return self

    def forecast(self) -> CovarianceForecast:
        """Return the window estimate as ``H_{t+1|t}``."""
        state = require_fitted_state(self._state, LINEAR_MODEL_NAME)
        return pack_forecast(state, self.identity)


class LedoitWolfNonlinearCovariance(CovarianceModel):
    """One-day-ahead Ledoit-Wolf 2020 analytical nonlinear shrinkage."""

    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.REFIT_HOLD,
        fit_input=FitInput.DAILY_RETURN,
        update_observable=UpdateObservable.NONE,
    )

    def __init__(self) -> None:
        self._fit: LedoitWolfNonlinearFitState | None = None
        self._state: NDArray[np.floating] | None = None

    @property
    def identity(self) -> ModelIdentity:
        configuration: dict[str, object] = {
            "estimator": NONLINEAR_ESTIMATOR,
            "centering": True,
            "covariance_divisor": COVARIANCE_DIVISOR,
            "reference_package": NONLINEAR_REFERENCE_PACKAGE,
        }
        if self._fit is not None:
            configuration.update(
                {
                    "n_observations": self._fit.n_observations,
                    "n_assets": self._fit.n_assets,
                    "n_eff": self._fit.n_eff,
                    "reference_version": self._fit.reference_version,
                    "analytical_method": self._fit.analytical_method,
                }
            )
        return ModelIdentity(name=NONLINEAR_MODEL_NAME, configuration=configuration)

    @property
    def fit_state(self) -> LedoitWolfNonlinearFitState:
        """Return stored nonlinear-shrinkage metadata. ``fit`` must have been called."""
        return _require_nonlinear_fit(self._fit)

    def fit(self, returns: ArrayLike) -> LedoitWolfNonlinearCovariance:
        """Wrap the pinned 2020 analytical estimator. Inputs are not mutated."""
        moments = centered_return_moments(returns)
        if moments.n_eff < NONLINEAR_MIN_N_EFF:
            raise InvalidModelInputError(
                "Ledoit-Wolf 2020 analytical shrinkage requires n_eff = T-1 "
                f">= {NONLINEAR_MIN_N_EFF} because the pinned nonlinshrink "
                f"reference rejects smaller effective samples; got T="
                f"{moments.n_observations}"
            )
        sigma = _analytical_nonlinear_shrinkage(moments.centered)
        _reject_invalid_nonlinear_forecast(sigma, moments.n_assets)
        # Read shrunk eigenvalues in the sample eigenbasis. Do not store U.
        shrunk = np.diag(
            moments.sample_eigenvectors.T @ sigma @ moments.sample_eigenvectors
        )
        self._fit = LedoitWolfNonlinearFitState(
            mean=moments.mean.copy(),
            n_observations=moments.n_observations,
            n_assets=moments.n_assets,
            n_eff=moments.n_eff,
            centering=True,
            covariance_divisor=COVARIANCE_DIVISOR,
            reference_package=NONLINEAR_REFERENCE_PACKAGE,
            reference_version=_nonlinshrink_version(),
            analytical_method=NONLINEAR_ESTIMATOR,
            sample_eigenvalues=moments.sample_eigenvalues.copy(),
            shrunk_eigenvalues=np.asarray(shrunk, dtype=float),
        )
        self._state = np.asarray(sigma, dtype=float)
        return self

    def forecast(self) -> CovarianceForecast:
        """Return the window estimate as ``H_{t+1|t}``."""
        state = require_fitted_state(self._state, NONLINEAR_MODEL_NAME)
        return pack_forecast(state, self.identity)


def centered_return_moments(returns: ArrayLike) -> CenteredReturnMoments:
    """Demean a return window and form ``S = Y'Y / (T-1)``.

    The caller is not assumed to have demeaned the returns. The input is
    copied. There is no annualization, scaling, or missing-value deletion.
    """
    history = as_daily_return_history(returns)
    # Subtract the in-window column mean.
    mean = history.mean(axis=0)
    centered = history - mean
    n_observations, n_assets = history.shape
    n_eff = n_observations - 1
    sample_covariance = centered.T @ centered / n_eff
    # Eigendecompose S for later eigenvector-retention checks.
    sample_eigenvalues, sample_eigenvectors = np.linalg.eigh(sample_covariance)
    return CenteredReturnMoments(
        returns=history,
        centered=centered,
        mean=mean,
        n_observations=n_observations,
        n_assets=n_assets,
        n_eff=n_eff,
        sample_covariance=sample_covariance,
        sample_eigenvalues=sample_eigenvalues,
        sample_eigenvectors=sample_eigenvectors,
    )


def ledoit_wolf_2004b_shrinkage_coefficient(
    scaled_centered: NDArray[np.floating],
) -> float:
    """Estimate the 2004b intensity on observations whose 1/T Gram equals S.

    ``scaled_centered`` has shape ``(T, N)`` and must already be centered.
    The rows are ``sqrt(T/(T-1)) Y``, so the MLE Gram
    ``(1/T) Y_ref' Y_ref`` equals the unbiased ``S = Y'Y/(T-1)``.
    For ``N=1`` the coefficient is 0, matching sklearn.
    """
    n_samples, n_features = scaled_centered.shape
    if n_features == 1:
        return 0.0
    squares = scaled_centered * scaled_centered
    emp_cov_trace = np.sum(squares, axis=0) / n_samples
    mu = float(np.sum(emp_cov_trace) / n_features)
    # Frobenius inner products used by the 2004b coefficient.
    gram = scaled_centered.T @ scaled_centered
    delta_ = float(np.sum(gram * gram) / (n_samples * n_samples))
    beta_ = float(np.sum(squares.T @ squares))
    beta = (1.0 / (n_features * n_samples)) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * float(np.sum(emp_cov_trace)) + n_features * mu * mu
    delta /= n_features
    # Do not shrink past the identity target.
    beta = min(beta, delta)
    if beta == 0.0:
        return 0.0
    return float(beta / delta)


def apply_linear_identity_shrinkage(
    sample_covariance: NDArray[np.floating],
    *,
    rho: float,
    mu: float,
) -> NDArray[np.floating]:
    """Return ``(1-rho) S + rho mu I``. Used by production and helper tests."""
    n_assets = sample_covariance.shape[0]
    identity = np.eye(n_assets, dtype=float)
    return (1.0 - rho) * sample_covariance + rho * mu * identity


def _analytical_nonlinear_shrinkage(
    centered: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Call the pinned 2020 analytical reference on already-centered returns.

    ``k=1`` tells the reference that the mean has already been removed and
    that the effective sample size is ``T-1``. The resulting sample
    covariance inside the reference is therefore ``Y'Y/(T-1)``.
    """
    import nonlinshrink as nls

    try:
        # Pass an independent copy so the reference cannot touch Y.
        return np.asarray(nls.shrink_cov(np.array(centered, copy=True), k=1), dtype=float)
    except Exception as exc:
        raise InvalidModelForecastError(
            "Ledoit-Wolf 2020 analytical reference returned an invalid "
            f"estimate rather than a usable covariance; {exc}"
        ) from exc


def _reject_invalid_nonlinear_forecast(
    matrix: NDArray[np.floating],
    n_assets: int,
) -> None:
    """Reject a nonfinite, misshapen, asymmetric, or non-strictly-PD matrix."""
    if matrix.shape != (n_assets, n_assets):
        raise InvalidModelForecastError(
            "Ledoit-Wolf 2020 analytical reference returned shape "
            f"{matrix.shape}, expected {(n_assets, n_assets)}"
        )
    if not np.isfinite(matrix).all():
        raise InvalidModelForecastError(
            "Ledoit-Wolf 2020 analytical reference returned a nonfinite matrix"
        )
    try:
        require_symmetric(matrix, "lw_nl forecast", atol=SYMMETRY_ATOL)
    except InvalidCovarianceMatrixError as exc:
        raise InvalidModelForecastError(str(exc)) from exc
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise InvalidModelForecastError(
            "Ledoit-Wolf 2020 analytical reference returned a matrix that "
            "is not strictly positive definite"
        ) from exc


def _nonlinshrink_version() -> str:
    """Read the installed pinned reference version at fit time."""
    return metadata.version(NONLINEAR_REFERENCE_PACKAGE)


def _require_linear_fit(
    state: LedoitWolfLinearFitState | None,
) -> LedoitWolfLinearFitState:
    """Reject a metadata read issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError(
            f"{LINEAR_MODEL_NAME} has no fitted origin-window state. "
            "Call fit before reading fit_state."
        )
    return state


def _require_nonlinear_fit(
    state: LedoitWolfNonlinearFitState | None,
) -> LedoitWolfNonlinearFitState:
    """Reject a metadata read issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError(
            f"{NONLINEAR_MODEL_NAME} has no fitted origin-window state. "
            "Call fit before reading fit_state."
        )
    return state
