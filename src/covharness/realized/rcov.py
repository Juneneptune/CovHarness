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
