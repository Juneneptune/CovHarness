"""Daily realized covariance from a synchronized intraday price grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from covharness.data.returns import synchronized_log_returns
from covharness.realized.rcov import realized_covariance


def daily_realized_covariance(synchronized_prices: pd.DataFrame) -> NDArray[np.floating]:
    """Unscaled daily realized covariance from one day's synchronized prices.

    Composition
    -----------
    synchronized prices ``(T, N)``
        -> log returns ``R`` of shape ``(M, N)`` with ``M = T - 1``
        -> ``RCov = R' R`` of shape ``(N, N)``

    Entry ``(i, j)`` is ``sum_m r_{i,m} r_{j,m}``. There is no division by ``M``,
    annualization, shrinkage, correlation conversion, or eigenvalue repair.

    The grid frequency is determined by the caller. This function does not
    hard-code 5 minutes.

    Missing data
    ------------
    Requires a complete return matrix. If any ``r_{i,m}`` is missing, raise
    rather than pairwise-delete. Pairwise deletion would mix intervals across
    entries and can destroy PSD structure.

    Opening and overnight returns are out of scope: the input must already be
    a valid synchronized price grid for the day.
    """
    if not isinstance(synchronized_prices, pd.DataFrame):
        raise TypeError("synchronized_prices must be a pandas DataFrame")
    # Require a unique, time-ordered price index.
    if not synchronized_prices.index.is_monotonic_increasing:
        raise ValueError("synchronized_prices index must be ordered (monotonic increasing)")
    if synchronized_prices.index.has_duplicates:
        raise ValueError("synchronized_prices index must have unique timestamps")

    # Log returns, then reject any missing cell (no pairwise deletion).
    returns = synchronized_log_returns(synchronized_prices)
    values = returns.to_numpy(dtype=float, na_value=np.nan)
    if values.shape[0] < 1:
        raise ValueError("need at least two synchronized prices to form a return")
    if not np.isfinite(values).all():
        raise ValueError(
            "daily realized covariance requires a complete (M, N) return matrix; "
            "missing returns are not pairwise-deleted"
        )
    return realized_covariance(values)
