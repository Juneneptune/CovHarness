"""Numerical contracts for covariance-space evaluation.

Loss functions copy inputs, check the documented conditions, and never repair
a matrix. A failed Cholesky of a forecast is a positive-definiteness failure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Absolute tolerance on max |A - A.T| for a covariance argument.
SYMMETRY_ATOL = 1e-10
# Absolute tolerance on the smallest eigenvalue for a PSD proxy.
PSD_ATOL = 1e-12


class InvalidCovarianceMatrixError(ValueError):
    """A covariance argument failed a documented numerical contract."""


class ForecastNotPositiveDefiniteError(InvalidCovarianceMatrixError):
    """Forecast ``H`` is not strictly PD. The loss does not repair it."""


class ProxyNotPositiveDefiniteError(InvalidCovarianceMatrixError):
    """Proxy ``S`` is not strictly PD. Full Stein requires SPD ``S``."""


def as_square_finite(matrix: ArrayLike, name: str) -> NDArray[np.floating]:
    """Copy ``matrix`` to a finite square 2-d array. The input is not mutated."""
    array = np.array(matrix, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise InvalidCovarianceMatrixError(
            f"{name} must be a square 2-d matrix; got shape {array.shape}"
        )
    if array.size == 0:
        raise InvalidCovarianceMatrixError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise InvalidCovarianceMatrixError(
            f"{name} must be finite (NaN and inf are rejected)"
        )
    return array


def require_matching_square(
    proxy: NDArray[np.floating],
    forecast: NDArray[np.floating],
) -> None:
    """Reject unequal dimensions after both arguments are known to be square."""
    if proxy.shape != forecast.shape:
        raise InvalidCovarianceMatrixError(
            f"proxy S and forecast H must have the same shape; "
            f"got {proxy.shape} and {forecast.shape}"
        )


def require_symmetric(
    matrix: NDArray[np.floating],
    name: str,
    *,
    atol: float = SYMMETRY_ATOL,
) -> None:
    """Reject a matrix whose departure from symmetry exceeds ``atol``."""
    # Measure the largest absolute off-symmetry residual. Do not symmetrize.
    error = float(np.max(np.abs(matrix - matrix.T)))
    if error > atol:
        raise InvalidCovarianceMatrixError(
            f"{name} must be symmetric within atol={atol}; "
            f"max |A-A.T| is {error}"
        )


def require_psd(
    matrix: NDArray[np.floating],
    name: str,
    *,
    atol: float = PSD_ATOL,
) -> None:
    """Reject a matrix with a smallest eigenvalue strictly below ``-atol``."""
    # Eigenvalues of the symmetric part. The matrix itself is not repaired.
    eigvals = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    min_eig = float(eigvals.min())
    if min_eig < -atol:
        raise InvalidCovarianceMatrixError(
            f"{name} must be positive semidefinite within atol={atol}; "
            f"min eigenvalue is {min_eig}"
        )


def cholesky_factor(
    matrix: NDArray[np.floating],
    name: str,
) -> NDArray[np.floating]:
    """Strict PD test. ``H = L @ L.T``. Failure is not repaired."""
    try:
        # Lower-triangular Cholesky. This is the positive-definiteness test.
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        if name == "forecast H":
            raise ForecastNotPositiveDefiniteError(
                "forecast H is not strictly positive definite. "
                "Cholesky failed. The loss does not repair the forecast."
            ) from exc
        raise ProxyNotPositiveDefiniteError(
            "proxy S is not strictly positive definite. "
            "Full Stein requires SPD S because of logdet(S). "
            "Use reduced QLIKE for a singular PSD proxy."
        ) from exc


def logdet_from_cholesky(chol: NDArray[np.floating]) -> float:
    """``logdet(A) = 2 * sum(log(diag(L)))`` for ``A = L @ L.T``."""
    diagonal = np.diag(chol)
    if np.any(diagonal <= 0.0):
        raise InvalidCovarianceMatrixError(
            "Cholesky diagonal must be strictly positive for logdet"
        )
    return float(2.0 * np.sum(np.log(diagonal)))


def trace_solve_from_cholesky(
    chol: NDArray[np.floating],
    proxy: NDArray[np.floating],
) -> float:
    """``trace(H^{-1} S)`` from the same ``L`` without forming ``inv(H)``.

    Solve ``L Y = S`` then ``L.T X = Y``, so ``X = H^{-1} S``.
    """
    from scipy.linalg import solve_triangular

    # Forward substitution against the Cholesky factor of H.
    lower = solve_triangular(chol, proxy, lower=True, check_finite=False)
    # Back substitution. X is H^{-1} S, not H^{-1}.
    solved = solve_triangular(chol.T, lower, lower=False, check_finite=False)
    return float(np.trace(solved))
