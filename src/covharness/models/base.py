"""Common one-day-ahead covariance-model contract.

Models consume a caller-supplied estimation window that ends at the forecast
origin. They do not inspect VALIDATION, SCREEN, or CONFIRM labels. The
protocol or runner is responsible for constructing that window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.diagnostics.epps import matrix_eigen_diagnostics
from covharness.losses.contracts import (
    PSD_ATOL,
    SYMMETRY_ATOL,
    InvalidCovarianceMatrixError,
    require_psd,
    require_symmetric,
)
from covharness.models.exceptions import InvalidModelInputError


@dataclass(frozen=True)
class ModelIdentity:
    """Name and configuration recorded with a forecast.

    ``configuration`` is an auditable mapping. It is not a tuning engine.
    """

    name: str
    configuration: Mapping[str, object]


@dataclass(frozen=True)
class ForecastDiagnostics:
    """Numerical inspection of one forecast matrix. No repair is applied.

    ``positive_definite`` uses a Cholesky test, matching evaluation losses.
    ``condition_number`` is defined only when the smallest eigenvalue is
    strictly larger than ``PSD_ATOL``.
    """

    symmetry_error: float
    min_eigenvalue: float
    positive_semidefinite: bool
    positive_definite: bool
    condition_number: float | None
    numerical_rank: int


@dataclass(frozen=True)
class CovarianceForecast:
    """One-step covariance forecast ``H`` of shape ``(N, N)``.

    ``matrix`` is an independent copy. It does not alias model or caller
    memory. Asset order matches the fitted window.
    """

    matrix: NDArray[np.floating]
    diagnostics: ForecastDiagnostics
    identity: ModelIdentity


class CovarianceModel(ABC):
    """One-day-ahead covariance forecast issued at a supplied origin.

    Later daily-return models (DCC, LSTM-BEKK) inherit this forecast and
    identity contract and supply their own ``fit`` signature. Realized-
    covariance models use :class:`RealizedCovarianceModel`.
    """

    @property
    @abstractmethod
    def identity(self) -> ModelIdentity:
        """Return the frozen name and configuration for this instance."""

    @abstractmethod
    def forecast(self) -> CovarianceForecast:
        """Return ``H_{t+1|t}`` from the fitted origin-window state."""


class RealizedCovarianceModel(CovarianceModel):
    """Model whose origin-window input is a realized-covariance cube.

    ``realized_covariances`` has shape ``(T, N, N)``. Time index ``T-1`` is
    the origin. The target day is not part of this array.
    """

    @abstractmethod
    def fit(self, realized_covariances: ArrayLike) -> RealizedCovarianceModel:
        """Construct origin-window state from ``(T, N, N)`` realized covariances."""


def as_realized_covariance_history(
    matrices: ArrayLike,
    name: str = "realized_covariances",
) -> NDArray[np.floating]:
    """Copy a finite PSD ``(T, N, N)`` history. Inputs are not mutated.

    Each slice must be square, finite, symmetric within ``SYMMETRY_ATOL``,
    and PSD within ``PSD_ATOL``. Strict PD is not required. The array is
    not symmetrized, shrunk, or eigenvalue-repaired.
    """
    array = np.array(matrices, dtype=float, copy=True)
    if array.ndim != 3:
        raise InvalidModelInputError(
            f"{name} must have shape (T, N, N); got {array.shape}"
        )
    n_times, n_rows, n_cols = array.shape
    if n_times < 1:
        raise InvalidModelInputError(f"{name} must contain at least one matrix")
    if n_rows != n_cols or n_rows < 1:
        raise InvalidModelInputError(
            f"{name} slices must be square and non-empty; got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise InvalidModelInputError(
            f"{name} must be finite (NaN and inf are rejected)"
        )
    # Check each matrix. Do not repair.
    for time_index in range(n_times):
        slice_name = f"{name}[{time_index}]"
        try:
            require_symmetric(array[time_index], slice_name, atol=SYMMETRY_ATOL)
            require_psd(array[time_index], slice_name, atol=PSD_ATOL)
        except InvalidCovarianceMatrixError as exc:
            raise InvalidModelInputError(str(exc)) from exc
    return array


def forecast_diagnostics(matrix: NDArray[np.floating]) -> ForecastDiagnostics:
    """Inspect one forecast copy. The matrix is not repaired."""
    eigvals, rank, min_eig, psd, cond, symmetry_error, _m_over_n = (
        matrix_eigen_diagnostics(matrix)
    )
    del eigvals
    # Cholesky PD test.
    positive_definite = _cholesky_pd(matrix)
    return ForecastDiagnostics(
        symmetry_error=float(symmetry_error),
        min_eigenvalue=float(min_eig),
        positive_semidefinite=bool(psd),
        positive_definite=bool(positive_definite),
        condition_number=cond,
        numerical_rank=int(rank),
    )


def pack_forecast(
    matrix: NDArray[np.floating],
    identity: ModelIdentity,
) -> CovarianceForecast:
    """Copy ``matrix`` and attach diagnostics. Caller memory is not aliased."""
    copied = np.array(matrix, dtype=float, copy=True)
    return CovarianceForecast(
        matrix=copied,
        diagnostics=forecast_diagnostics(copied),
        identity=identity,
    )


def require_fitted_state(state: NDArray[np.floating] | None, name: str) -> NDArray[np.floating]:
    """Reject a forecast call issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError(
            f"{name} has no fitted origin-window state. Call fit before forecast."
        )
    return state


def _cholesky_pd(matrix: NDArray[np.floating]) -> bool:
    """True iff lower Cholesky succeeds. Failure is not repaired."""
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True
