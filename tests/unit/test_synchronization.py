"""Unit tests for previous-tick synchronization and synchronized log returns."""

import numpy as np
import pandas as pd
import pytest

from covharness.data.returns import synchronized_log_returns
from covharness.data.synchronization import previous_tick_sync


def _ts(hour: int, minute: int, second: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"2019-01-02 {hour:02d}:{minute:02d}:{second:02d}")


def _ticks(rows: list[tuple[pd.Timestamp, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["timestamp", "asset", "price"])


def test_previous_tick_uses_observation_before_grid_not_after() -> None:
    data = _ticks(
        [
            (_ts(9, 34, 50), "A", 100.0),
            (_ts(9, 35, 10), "A", 200.0),
        ]
    )
    grid = pd.DatetimeIndex([_ts(9, 35, 0)])

    synced = previous_tick_sync(data, grid)

    assert synced.loc[_ts(9, 35, 0), "A"] == 100.0
    assert synced.loc[_ts(9, 35, 0), "A"] != 200.0


def test_asynchronous_assets_use_own_latest_price() -> None:
    data = _ticks(
        [
            (_ts(9, 34, 50), "A", 100.0),
            (_ts(9, 33, 0), "B", 50.0),
            (_ts(9, 34, 59), "B", 51.0),
        ]
    )
    grid = pd.DatetimeIndex([_ts(9, 35, 0)])

    synced = previous_tick_sync(data, grid)

    assert synced.loc[_ts(9, 35, 0), "A"] == 100.0
    assert synced.loc[_ts(9, 35, 0), "B"] == 51.0


def test_no_prior_observation_is_missing() -> None:
    data = _ticks(
        [
            (_ts(9, 34, 50), "A", 100.0),
            (_ts(9, 35, 10), "B", 50.0),
        ]
    )
    grid = pd.DatetimeIndex([_ts(9, 35, 0)])

    synced = previous_tick_sync(data, grid)

    assert synced.loc[_ts(9, 35, 0), "A"] == 100.0
    assert pd.isna(synced.loc[_ts(9, 35, 0), "B"])


def test_unsorted_input_matches_sorted_input() -> None:
    data = _ticks(
        [
            (_ts(9, 34, 50), "A", 100.0),
            (_ts(9, 35, 10), "A", 101.0),
            (_ts(9, 33, 0), "B", 50.0),
            (_ts(9, 36, 1), "B", 52.0),
        ]
    )
    grid = pd.DatetimeIndex([_ts(9, 35, 0), _ts(9, 40, 0)])
    shuffled = data.sample(frac=1.0, random_state=0).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        previous_tick_sync(data, grid),
        previous_tick_sync(shuffled, grid),
    )


def test_log_returns_match_log_price_ratio() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 110.0], "B": [50.0, 25.0]},
        index=pd.DatetimeIndex([_ts(9, 35, 0), _ts(9, 40, 0)], name="timestamp"),
    )

    returns = synchronized_log_returns(prices)

    assert list(returns.index) == [_ts(9, 40, 0)]
    np.testing.assert_allclose(returns.loc[_ts(9, 40, 0), "A"], np.log(110.0 / 100.0))
    np.testing.assert_allclose(returns.loc[_ts(9, 40, 0), "B"], np.log(25.0 / 50.0))


def test_observation_after_grid_is_not_used() -> None:
    data = _ticks(
        [
            (_ts(9, 35, 10), "A", 200.0),
        ]
    )
    grid = pd.DatetimeIndex([_ts(9, 35, 0)])

    synced = previous_tick_sync(data, grid)

    assert pd.isna(synced.loc[_ts(9, 35, 0), "A"])


@pytest.mark.parametrize(
    "bad_price, match",
    [
        (0.0, "strictly positive"),
        (-1.0, "strictly positive"),
        (np.nan, "finite"),
        (np.inf, "finite"),
    ],
)
def test_invalid_prices_are_rejected(bad_price: float, match: str) -> None:
    data = _ticks(
        [
            (_ts(9, 34, 50), "A", 100.0),
            (_ts(9, 34, 51), "A", bad_price),
        ]
    )
    grid = pd.DatetimeIndex([_ts(9, 35, 0)])

    with pytest.raises(ValueError, match=match):
        previous_tick_sync(data, grid)


def test_duplicate_asset_timestamp_is_rejected() -> None:
    data = _ticks(
        [
            (_ts(9, 34, 50), "A", 100.0),
            (_ts(9, 34, 50), "A", 101.0),
        ]
    )
    grid = pd.DatetimeIndex([_ts(9, 35, 0)])

    with pytest.raises(ValueError, match="duplicate"):
        previous_tick_sync(data, grid)
