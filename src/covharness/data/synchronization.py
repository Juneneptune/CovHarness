"""Previous-tick synchronization of irregular trade prices onto a fixed grid."""

from __future__ import annotations

import numpy as np
import pandas as pd


def previous_tick_sync(
    data: pd.DataFrame,
    grid: pd.DatetimeIndex | pd.Series | np.ndarray | list,
    *,
    timestamp_col: str = "timestamp",
    asset_col: str = "asset",
    price_col: str = "price",
) -> pd.DataFrame:
    """Map irregular prices onto ``grid`` by previous-tick (as-of) matching.

    For each asset ``i`` and grid time ``g``,

        P[i, g] = latest observed price for i with timestamp <= g.

    Observations with timestamp ``> g`` are never used. There is no linear
    interpolation and no backward fill from future prices. If asset ``i`` has
    no observation at or before ``g``, the value is missing.

    The grid is supplied by the caller, so the same function supports 1-minute,
    5-minute, 15-minute, and offset sampling schemes.

    Parameters
    ----------
    data : DataFrame
        Long-form ticks with columns ``timestamp``, ``asset``, ``price``.
    grid : datetime-like sequence
        Target sampling times. Sorted internally; duplicates are rejected.

    Returns
    -------
    DataFrame
        Wide prices, index ``grid`` (name ``timestamp``), one column per asset.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    # Require timestamp, asset, and price.
    required = (timestamp_col, asset_col, price_col)
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"data missing required columns: {missing}")

    # Copy ticks and parse timestamp and price.
    frame = data.loc[:, list(required)].copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="raise")
    frame[price_col] = pd.to_numeric(frame[price_col], errors="raise")

    # Reject nonfinite or non-positive prices.
    prices = frame[price_col].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(prices).all():
        raise ValueError("prices must be finite (NaN and inf are rejected)")
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive")

    # Reject duplicate (timestamp, asset) rows. Resolve them before this step.
    duplicate = frame.duplicated(subset=[timestamp_col, asset_col], keep=False)
    if duplicate.any():
        raise ValueError(
            "duplicate (timestamp, asset) observations are rejected; "
            "resolve them before synchronization"
        )

    # Sort the sampling grid. Duplicates are not allowed.
    grid_index = pd.DatetimeIndex(grid)
    if grid_index.has_duplicates:
        raise ValueError("grid timestamps must be unique")
    if not grid_index.is_monotonic_increasing:
        grid_index = grid_index.sort_values()
    grid_index = grid_index.rename("timestamp")

    if frame.empty:
        return pd.DataFrame(index=grid_index)

    # Wide panel, then ffill in calendar time and restrict to the grid (no bfill).
    wide = (
        frame.pivot(index=timestamp_col, columns=asset_col, values=price_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    union = wide.index.union(grid_index)
    synced = wide.reindex(union).ffill().reindex(grid_index)
    synced.index.name = "timestamp"
    synced.columns.name = None
    return synced
