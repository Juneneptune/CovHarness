"""Synchronized log returns from a wide price panel."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synchronized_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns on a synchronized price grid.

    For consecutive grid prices ``P_{j-1}`` and ``P_j``,

        r_j = log(P_j) - log(P_{j-1}) = log(P_j / P_{j-1}).

    The returned index is the interval endpoint ``j``. Missing prices are left
    missing; returns are not filled, interpolated, or computed over skipped
    gaps.

    Parameters
    ----------
    prices : DataFrame
        Wide synchronized prices, shape ``(T, N)``. Simple (percentage)
        returns are not used.

    Returns
    -------
    DataFrame
        Log returns, shape ``(T - 1, N)``, same asset columns as ``prices``.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    # Reject inf and non-positive prices. Missing stays missing.
    values = prices.to_numpy(dtype=float, na_value=np.nan, copy=False)
    if np.isinf(values).any():
        raise ValueError("prices must be finite (inf is rejected)")
    finite = np.isfinite(values)
    if np.any(values[finite] <= 0):
        raise ValueError("prices must be strictly positive")

    # Log-price difference on consecutive grid times. Drop the first NaN row.
    return np.log(prices).diff().iloc[1:]
