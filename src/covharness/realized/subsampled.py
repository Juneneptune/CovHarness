"""Subsampled realized covariance from a regular synchronized price grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from covharness.realized.daily import daily_realized_covariance

SUBSAMPLE_STEP = 5


def subsample_offset_prices(
    synchronized_prices: pd.DataFrame,
    offset: int,
    *,
    subsample_step: int = SUBSAMPLE_STEP,
) -> pd.DataFrame:
    """Every ``subsample_step``-th price, starting at ``offset``.

    On a 1-minute grid with ``subsample_step=5``, offset 0 is 09:30, 09:35, ...
    and offset 1 is 09:31, 09:36, ....
    """
    _validate_price_grid(synchronized_prices)
    if subsample_step < 1:
        raise ValueError(f"subsample_step must be >= 1; got {subsample_step}")
    if offset < 0 or offset >= subsample_step:
        raise ValueError(
            f"offset must be in 0..{subsample_step - 1}; got {offset}"
        )
    # Every step-th price beginning at this offset.
    return synchronized_prices.iloc[offset::subsample_step]


def subsampled_realized_covariance(
    synchronized_prices: pd.DataFrame,
    *,
    subsample_step: int = SUBSAMPLE_STEP,
) -> NDArray[np.floating]:
    """Average of unscaled RCovs on ``subsample_step`` offset subgrids.

    For offsets ``s = 0, ..., step-1``, take every ``step``-th synchronized
    price beginning at ``s``, form log returns, and compute ``R' R``. The
    subsampled proxy is the arithmetic mean of those Gram matrices.

    There is no division by the number of intervals, annualization, shrinkage,
    correlation conversion, or eigenvalue repair. A missing return on any
    offset aborts; observations are not pairwise-deleted.

    Default ``subsample_step=5`` is the 5-minute-from-1-minute proxy
    ``RCov_5min_ss``. The step is a caller argument so the same routine can
    serve other frequencies later.
    """
    _validate_price_grid(synchronized_prices)
    if subsample_step < 1:
        raise ValueError(f"subsample_step must be >= 1; got {subsample_step}")
    if len(synchronized_prices) < 2 * subsample_step:
        raise ValueError(
            "need at least two prices on every offset grid "
            f"(len >= {2 * subsample_step} for subsample_step={subsample_step})"
        )

    # One unscaled RCov per offset grid, then arithmetic mean of those matrices.
    rcovs = [
        daily_realized_covariance(
            subsample_offset_prices(
                synchronized_prices, offset, subsample_step=subsample_step
            )
        )
        for offset in range(subsample_step)
    ]
    return np.mean(np.stack(rcovs, axis=0), axis=0)


def _validate_price_grid(synchronized_prices: pd.DataFrame) -> None:
    if not isinstance(synchronized_prices, pd.DataFrame):
        raise TypeError("synchronized_prices must be a pandas DataFrame")
    # Require a unique, ordered, regularly spaced price index.
    index = synchronized_prices.index
    if not index.is_monotonic_increasing:
        raise ValueError(
            "synchronized_prices index must be ordered (monotonic increasing)"
        )
    if index.has_duplicates:
        raise ValueError("synchronized_prices index must have unique timestamps")
    if len(index) >= 2:
        deltas = index.to_series().diff().iloc[1:]
        if not bool(deltas.eq(deltas.iloc[0]).all()):
            raise ValueError("synchronized_prices index must be regularly spaced")
