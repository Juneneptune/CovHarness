"""Synchronized realized covariance."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def realized_covariance(intraday_returns: ArrayLike) -> NDArray[np.floating]:
    """Synchronized realized covariance, unscaled.

    Definition
    ----------
    For a matrix ``R`` of synchronized intraday returns with shape ``(M, N)``
    (``M`` intervals, ``N`` assets),

        RCov = R' R = sum_{j=1}^{M} r_j r_j'

    Returns an ``(N, N)`` Gram matrix. There is no division by ``M``, no
    annualization, no shrinkage, and no eigenvalue repair.
    """
    returns = np.asarray(intraday_returns, dtype=float)
    # Require a complete (M, N) return matrix.
    if returns.ndim != 2:
        raise ValueError(
            f"intraday_returns must have shape (M, N); got ndim={returns.ndim}"
        )
    if not np.isfinite(returns).all():
        raise ValueError("intraday_returns must be finite (NaN and inf are rejected)")
    # Unscaled Gram matrix R'R.
    return returns.T @ returns


def realized_semivariances(
    intraday_returns: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Positive and negative realized semivariances.

    For ``R`` with shape ``(M, N)``,

        RS+_i = sum_j R_{j,i}^2 1{R_{j,i} > 0}
        RS-_i = sum_j R_{j,i}^2 1{R_{j,i} < 0}

    Zeros contribute to neither side. The two vectors sum to the diagonal
    of ``realized_covariance(R)``.
    """
    returns = np.asarray(intraday_returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError(
            f"intraday_returns must have shape (M, N); got ndim={returns.ndim}"
        )
    if not np.isfinite(returns).all():
        raise ValueError("intraday_returns must be finite (NaN and inf are rejected)")
    # Split squares by sign. A zero return is in neither semivariance.
    positive = np.where(returns > 0.0, returns * returns, 0.0)
    negative = np.where(returns < 0.0, returns * returns, 0.0)
    return positive.sum(axis=0), negative.sum(axis=0)
