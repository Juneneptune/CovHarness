"""Unit tests for NBBO quote cleaning adapted from Barndorff-Nielsen et al. (2011)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covharness.data.quotes import (
    apply_q4_keep_mask,
    clean_nbbo_quotes,
    cleaned_quotes_as_ticks,
)
from covharness.data.synchronization import previous_tick_sync


DAY = pd.Timestamp("2009-02-13")
DAY2 = pd.Timestamp("2009-02-17")


def _ts(hour: int, minute: int, second: int = 0, day: pd.Timestamp = DAY) -> pd.Timestamp:
    return pd.Timestamp(
        year=day.year,
        month=day.month,
        day=day.day,
        hour=hour,
        minute=minute,
        second=second,
    )


def _quote_row(
    timestamp: pd.Timestamp,
    asset: str,
    bid: float,
    ask: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "asset": asset,
        "best_bid": bid,
        "best_ask": ask,
    }


def _quotes(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _stable_series(
    asset: str,
    n: int,
    *,
    bid: float = 24.99,
    ask: float = 25.01,
    start: pd.Timestamp | None = None,
    day: pd.Timestamp = DAY,
) -> pd.DataFrame:
    origin = start if start is not None else _ts(9, 40, 0, day=day)
    stamps = pd.date_range(origin, periods=n, freq="1s")
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "asset": asset,
            "best_bid": bid,
            "best_ask": ask,
        }
    )


def test_q1_duplicate_timestamps_collapse_to_median_bid_and_ask() -> None:
    stamp = _ts(10, 0, 0)
    data = _quotes(
        [
            _quote_row(stamp, "IBM", 100.0, 101.0),
            _quote_row(stamp, "IBM", 102.0, 103.0),
            _quote_row(stamp, "IBM", 104.0, 105.0),
        ]
    )

    result = clean_nbbo_quotes(data)
    quotes = result.quotes

    assert len(quotes) == 1
    assert quotes.loc[0, "best_bid"] == 102.0
    assert quotes.loc[0, "best_ask"] == 103.0
    assert result.diagnostics.n_input == 3
    assert result.diagnostics.n_after_q1 == 1
    assert result.diagnostics.n_removed_q1 == 2


def test_q2_negative_spread_is_removed() -> None:
    data = _quotes(
        [
            _quote_row(_ts(10, 0, 0), "IBM", 100.0, 100.02),
            _quote_row(_ts(10, 0, 1), "IBM", 100.05, 100.03),
            _quote_row(_ts(10, 0, 2), "IBM", 100.01, 100.03),
        ]
    )

    result = clean_nbbo_quotes(data)

    assert len(result.quotes) == 2
    assert result.diagnostics.n_removed_q2 == 1
    assert (result.quotes["spread"] >= 0.0).all()
    assert not (
        (result.quotes["best_bid"] == 100.05) & (result.quotes["best_ask"] == 100.03)
    ).any()


def test_locked_quote_zero_spread_is_retained() -> None:
    data = _quotes(
        [
            _quote_row(_ts(10, 0, 0), "IBM", 100.0, 100.02),
            _quote_row(_ts(10, 0, 1), "IBM", 100.01, 100.01),
            _quote_row(_ts(10, 0, 2), "IBM", 100.01, 100.03),
        ]
    )

    result = clean_nbbo_quotes(data)

    locked = result.quotes.loc[result.quotes["spread"] == 0.0]
    assert len(result.quotes) == 3
    assert len(locked) == 1
    assert locked.iloc[0]["best_bid"] == 100.01
    assert locked.iloc[0]["best_ask"] == 100.01
    assert result.diagnostics.n_removed_q2 == 0


def test_q3_spread_above_ten_times_median_is_removed() -> None:
    rows = [
        _quote_row(_ts(10, 0, second), "IBM", 100.00, 100.02) for second in range(9)
    ]
    rows.append(_quote_row(_ts(10, 1, 0), "IBM", 100.00, 100.30))
    data = _quotes(rows)

    result = clean_nbbo_quotes(data)

    assert result.diagnostics.n_after_q2 == 10
    assert result.diagnostics.n_removed_q3 == 1
    assert (result.quotes["spread"] <= 0.20 + 1e-12).all()
    assert not (result.quotes["best_ask"] == 100.30).any()


def test_q4_isolated_bad_midpoint_with_narrow_spread_is_removed() -> None:
    data = _stable_series("JPM", 51)
    data.loc[25, ["best_bid", "best_ask"]] = (22.57, 22.59)

    result = clean_nbbo_quotes(data)

    assert result.diagnostics.n_removed_q4 == 1
    assert len(result.quotes) == 50
    mids = result.quotes["midquote"].to_numpy()
    assert np.all(np.abs(mids - 25.0) < 0.02)
    assert not np.any(np.abs(result.quotes["midquote"] - 22.58) < 1e-9)


def test_q4_normal_midpoint_in_stable_neighborhood_remains() -> None:
    data = _stable_series("JPM", 51)

    result = clean_nbbo_quotes(data)

    assert len(result.quotes) == 51
    assert result.diagnostics.n_removed_q4 == 0
    np.testing.assert_allclose(result.quotes["midquote"], 25.0)


def test_q4_zero_local_dispersion_keeps_identical_center_drops_different() -> None:
    identical = np.full(51, 25.0)
    keep_same = apply_q4_keep_mask(identical)
    assert keep_same.all()

    different = identical.copy()
    different[25] = 22.6
    keep_diff = apply_q4_keep_mask(different)
    assert keep_diff.sum() == 50
    assert not keep_diff[25]

    data_same = _stable_series("JPM", 51)
    data_diff = data_same.copy()
    data_diff.loc[25, ["best_bid", "best_ask"]] = (22.59, 22.61)

    assert len(clean_nbbo_quotes(data_same).quotes) == 51
    assert len(clean_nbbo_quotes(data_diff).quotes) == 50


def test_q4_neighborhood_does_not_cross_stock_or_day_boundaries() -> None:
    asset_a = _stable_series("AAA", 51, bid=24.99, ask=25.01)
    asset_a.loc[25, ["best_bid", "best_ask"]] = (22.57, 22.59)
    asset_b = _stable_series("BBB", 51, bid=99.99, ask=100.01)
    day2 = _stable_series("AAA", 51, bid=24.99, ask=25.01, day=DAY2)

    data = pd.concat([asset_a, asset_b, day2], ignore_index=True)
    result = clean_nbbo_quotes(data)
    quotes = result.quotes

    a_day1 = quotes.loc[(quotes["asset"] == "AAA") & (quotes["date"] == DAY)]
    b_day1 = quotes.loc[(quotes["asset"] == "BBB") & (quotes["date"] == DAY)]
    a_day2 = quotes.loc[(quotes["asset"] == "AAA") & (quotes["date"] == DAY2)]

    assert len(a_day1) == 50
    assert len(b_day1) == 51
    assert len(a_day2) == 51
    assert result.diagnostics.n_removed_q4 == 1
    np.testing.assert_allclose(b_day1["midquote"], 100.0)
    np.testing.assert_allclose(a_day2["midquote"], 25.0)
    assert not np.any(np.abs(a_day1["midquote"] - 22.58) < 1e-9)


def test_cleaned_output_is_chronologically_sorted_within_stock_day() -> None:
    a = _stable_series("AAA", 10, start=_ts(10, 0, 0))
    b = _stable_series("BBB", 8, start=_ts(11, 0, 0))
    data = pd.concat([b, a], ignore_index=True)
    shuffled = data.sample(frac=1.0, random_state=1).reset_index(drop=True)

    quotes = clean_nbbo_quotes(shuffled).quotes

    for _, group in quotes.groupby(["date", "asset"], sort=False):
        assert group["timestamp"].is_monotonic_increasing
        assert not group["timestamp"].duplicated().any()


def test_cleaning_occurs_before_previous_tick_synchronization() -> None:
    data = _stable_series("JPM", 51)
    data.loc[25, ["best_bid", "best_ask"]] = (22.57, 22.59)
    grid = pd.DatetimeIndex([data.loc[25, "timestamp"]])

    raw_ticks = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "asset": data["asset"],
            "price": (data["best_bid"] + data["best_ask"]) / 2.0,
        }
    )
    raw_synced = previous_tick_sync(raw_ticks, grid)
    assert raw_synced.loc[grid[0], "JPM"] == pytest.approx(22.58)

    cleaned = clean_nbbo_quotes(data)
    cleaned_synced = previous_tick_sync(cleaned_quotes_as_ticks(cleaned), grid)
    assert cleaned_synced.loc[grid[0], "JPM"] == pytest.approx(25.0)


def test_jpm_style_anomaly_is_removed_before_previous_tick_sampling() -> None:
    """Synthetic analogue of the 2009-02-13 JPM NBBO spike near 09:45.

    Surrounding midquotes are near 25. One narrow erroneous quote near 22.6
    is the last observation before a five-minute grid time. Raw previous-tick
    sampling therefore captures 22.6. After Q4 the same grid time falls back
    to the last valid quote near 25. Later quotes exist so that the centered
    Q4 window is complete; they arrive after the grid and cannot repair the
    raw sampler.
    """
    before = _stable_series("JPM", 25, start=_ts(9, 40, 0))
    bad = _quotes([_quote_row(_ts(9, 44, 59), "JPM", 22.59, 22.61)])
    after = _stable_series("JPM", 25, start=_ts(9, 45, 1))
    data = pd.concat([before, bad, after], ignore_index=True)

    grid = pd.DatetimeIndex([_ts(9, 45, 0)])
    raw_ticks = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "asset": data["asset"],
            "price": (data["best_bid"] + data["best_ask"]) / 2.0,
        }
    )

    raw_synced = previous_tick_sync(raw_ticks, grid)
    assert raw_synced.loc[_ts(9, 45, 0), "JPM"] == pytest.approx(22.60)

    cleaned = clean_nbbo_quotes(data)
    assert cleaned.diagnostics.n_removed_q4 == 1
    cleaned_synced = previous_tick_sync(cleaned_quotes_as_ticks(cleaned), grid)
    assert cleaned_synced.loc[_ts(9, 45, 0), "JPM"] == pytest.approx(25.0)


def test_p1_drops_quotes_outside_regular_hours() -> None:
    data = _quotes(
        [
            _quote_row(_ts(9, 29, 59), "IBM", 100.0, 100.02),
            _quote_row(_ts(9, 30, 0), "IBM", 100.0, 100.02),
            _quote_row(_ts(16, 0, 0), "IBM", 100.0, 100.02),
            _quote_row(_ts(16, 0, 1), "IBM", 100.0, 100.02),
        ]
    )

    result = clean_nbbo_quotes(data)

    assert result.diagnostics.n_removed_p1 == 2
    assert result.diagnostics.n_after_p1 == 2
    assert result.quotes["timestamp"].min() == _ts(9, 30, 0)
    assert result.quotes["timestamp"].max() == _ts(16, 0, 0)


def test_p2_requires_strictly_positive_bid_and_ask() -> None:
    data = _quotes(
        [
            _quote_row(_ts(10, 0, 0), "IBM", 0.0, 100.02),
            _quote_row(_ts(10, 0, 1), "IBM", 100.0, 0.0),
            _quote_row(_ts(10, 0, 2), "IBM", 100.0, 100.02),
        ]
    )

    result = clean_nbbo_quotes(data)

    assert result.diagnostics.n_removed_p2 == 2
    assert len(result.quotes) == 1
    assert result.quotes.iloc[0]["best_bid"] == 100.0
