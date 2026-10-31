"""Global-minimum-variance weights, drifted turnover, and return costs.

The forecast is taken as given. Nothing here refits a covariance model.
Costs are subtracted from portfolio returns and are not subtracted from
variance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

BPS_TO_RETURN = 1e-4


class LongOnlySolverError(RuntimeError):
    """The long-only quadratic program did not return a feasible weight."""


def unconstrained_gmv_weights(covariance: ArrayLike) -> NDArray[np.floating]:
    """Return w = H^{-1} 1 / (1' H^{-1} 1) without forming the inverse.

    ``covariance`` has shape ``(N, N)`` and must be strictly positive definite.
    """
    matrix = np.asarray(covariance, dtype=float)
    # Symmetrize the floating-point residual, then factor.
    matrix = 0.5 * (matrix + matrix.T)
    ones = np.ones(matrix.shape[0], dtype=float)
    factor = np.linalg.cholesky(matrix)
    solved = np.linalg.solve(factor.T, np.linalg.solve(factor, ones))
    return solved / solved.sum()


def long_only_gmv_weights(covariance: ArrayLike) -> NDArray[np.floating]:
    """Minimize w' H w subject to sum(w) = 1 and w >= 0.

    The solver is SciPy SLSQP. A failed or infeasible solve is raised.
    The weight vector is not clipped and is not replaced by a projection.
    """
    matrix = np.asarray(covariance, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    n_assets = matrix.shape[0]
    start = np.full(n_assets, 1.0 / n_assets)

    def objective(weights: NDArray[np.floating]) -> float:
        return float(weights @ matrix @ weights)

    # Equality keeps the book fully invested. Bounds keep it long-only.
    solved = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, None)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"ftol": 1e-14, "maxiter": 500, "disp": False},
    )
    if not solved.success:
        raise LongOnlySolverError(str(solved.message))
    weights = np.asarray(solved.x, dtype=float)
    if not np.isfinite(weights).all() or float(np.min(weights)) < -1e-8:
        raise LongOnlySolverError("long-only weights are not feasible")
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        raise LongOnlySolverError("long-only weights do not sum to one")
    return weights


def drift_weights(
    weights: ArrayLike,
    simple_returns: ArrayLike,
) -> NDArray[np.floating]:
    """Previous weights after one day of simple-return drift.

    Component i grows by (1 + R_i) and the vector is renormalized by the
    portfolio gross return. Both inputs have shape ``(N,)``.
    """
    held = np.asarray(weights, dtype=float)
    returns = np.asarray(simple_returns, dtype=float)
    grown = held * (1.0 + returns)
    gross = float(grown.sum())
    if gross <= 0.0:
        raise ValueError("drifted portfolio wealth is not positive")
    return grown / gross


def turnover(weights: ArrayLike, drifted: ArrayLike) -> float:
    """L1 distance between the new weights and the drifted previous weights."""
    return float(np.abs(np.asarray(weights) - np.asarray(drifted)).sum())


def fleming_kirby_ostdiek_fee(
    simple_returns_a: ArrayLike,
    simple_returns_b: ArrayLike,
    gamma: float,
) -> float:
    """Daily Fleming–Kirby–Ostdiek fee on gross returns.

    ``simple_returns_a`` and ``simple_returns_b`` are net-of-cost simple
    returns with shape ``(T,)``. The gross return is ``R = 1 + r``, and the
    fee is subtracted from ``R_A``. Means and variances in the root rule use
    the ``1/T`` divisor, matching the average utility. The returned root is
    the one closer to ``(mean_A - mean_B) - (gamma / 2) (var_A - var_B)``.
    """
    returns_a = np.asarray(simple_returns_a, dtype=float).reshape(-1)
    returns_b = np.asarray(simple_returns_b, dtype=float).reshape(-1)
    if returns_a.shape != returns_b.shape or returns_a.size == 0:
        raise ValueError("fee returns must be nonempty and aligned")
    if not np.isfinite(returns_a).all() or not np.isfinite(returns_b).all():
        raise ValueError("fee returns must be finite")
    risk_aversion = float(gamma)
    if not np.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise ValueError("gamma must be positive and finite")
    # Convert the net simple returns to gross wealth relatives.
    gross_a = 1.0 + returns_a
    gross_b = 1.0 + returns_b
    theta = risk_aversion / (2.0 * (1.0 + risk_aversion))
    mean_a = float(gross_a.mean())
    mean_b = float(gross_b.mean())
    second_a = float(np.mean(gross_a**2))
    second_b = float(np.mean(gross_b**2))
    # Solve the quadratic average-utility equality for the daily fee.
    quadratic = np.roots(
        [
            theta,
            1.0 - 2.0 * theta * mean_a,
            mean_b - theta * second_b - mean_a + theta * second_a,
        ]
    )
    real_roots = np.real(quadratic[np.isclose(quadratic.imag, 0.0)])
    if real_roots.size == 0:
        raise ValueError("the performance-fee quadratic has no real root")
    # Keep the root nearer the mean-variance approximation.
    variance_a = float(returns_a.var(ddof=0))
    variance_b = float(returns_b.var(ddof=0))
    anchor = (float(returns_a.mean()) - float(returns_b.mean())) - (
        risk_aversion / 2.0
    ) * (variance_a - variance_b)
    return float(real_roots[np.argmin(np.abs(real_roots - anchor))])


def subtract_turnover_cost(
    portfolio_return: ArrayLike,
    turnover_series: ArrayLike,
    bps: float,
) -> NDArray[np.floating]:
    """Subtract basis points per unit turnover from returns, not from variance."""
    gross = np.asarray(portfolio_return, dtype=float)
    traded = np.asarray(turnover_series, dtype=float)
    return gross - (float(bps) * BPS_TO_RETURN) * traded
