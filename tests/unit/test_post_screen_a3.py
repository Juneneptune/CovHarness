"""Tests for the parameterized Binance interval grid and sign summaries."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from covharness.data.binance_calendar import (
    FIVE_MINUTE_END_MINUTES,
    HALT_DATE,
    SEGMENT_BOUNDS,
    InvalidBinanceCalendarError,
    dates_through_screen,
    interval_end_minutes,
)
from covharness.realized.binance_panel import (
    N_ASSETS,
    five_minute_log_returns,
    measure_day_on_interval,
    production_rcov_through_screen,
    build_preconfirm_interval_measurements,
)
from covharness.realized.rcov import realized_covariance, realized_semivariances
from covharness.realized.sign_asymmetry import bias_corrected_skewness, tail_sign_summary

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "binance_five_asset_panel.npz"
CACHE_ROOT = REPO_ROOT / "data" / "cache" / "open_data" / "binance" / "spot" / "monthly" / "klines" / "1m"


def _positive_cube(n_days: int = 3) -> np.ndarray:
    """Strictly positive closes with shape ``(N, n_days, 1440)``."""
    rng = np.random.default_rng(20261028)
    shock = 0.001 * rng.normal(size=(N_ASSETS, n_days, 1440))
    path = np.exp(np.cumsum(shock, axis=2))
    level = np.array([100.0, 50.0, 20.0, 10.0, 2.0], dtype=float)
    return level[:, None, None] * path


def test_interval_five_is_the_frozen_tuple() -> None:
    assert interval_end_minutes(5) is FIVE_MINUTE_END_MINUTES
    assert len(interval_end_minutes(5)) == 288


def test_return_counts_match_1440_over_interval() -> None:
    start = dt.date(2024, 1, 1)
    cube = _positive_cube()
    day = dt.date(2024, 1, 2)
    for interval in (1, 5, 15, 30):
        returns, matrix, _daily = measure_day_on_interval(cube, start, day, interval)
        expected = 1440 // interval
        assert returns.shape == (expected, N_ASSETS)
        assert matrix.shape == (N_ASSETS, N_ASSETS)
        rs_pos, rs_neg = realized_semivariances(returns)
        squares = returns * returns
        # Elementwise partition is exact. The Gram diagonal uses a different sum.
        partition = np.where(returns > 0.0, squares, 0.0) + np.where(
            returns < 0.0, squares, 0.0
        )
        np.testing.assert_array_equal(partition, squares)
        np.testing.assert_allclose(
            rs_pos + rs_neg,
            np.diag(matrix),
            rtol=1e-12,
            atol=1e-15,
        )


def test_five_minute_wrapper_matches_shared_grid() -> None:
    start = dt.date(2024, 1, 1)
    cube = _positive_cube()
    day = dt.date(2024, 1, 2)
    returns, matrix, _daily = measure_day_on_interval(cube, start, day, 5)
    prior = cube[:, 0, 1439]
    endpoints = cube[:, 1, list(FIVE_MINUTE_END_MINUTES)].T
    wrapped = five_minute_log_returns(endpoints, prior)
    np.testing.assert_array_equal(returns, wrapped)
    np.testing.assert_array_equal(matrix, realized_covariance(wrapped))


def test_dates_through_screen_exclude_halt_and_confirm() -> None:
    dates = dates_through_screen()
    assert len(dates) == 1250
    assert dates[0] == dt.date(2021, 11, 9)
    assert dates[-1] == dt.date(2025, 4, 12)
    assert HALT_DATE not in dates
    assert all(day < SEGMENT_BOUNDS["CONFIRM"][0] for day in dates)
    with pytest.raises(InvalidBinanceCalendarError):
        interval_end_minutes(7)


def test_semivariances_sum_to_rv_with_a_zero_return() -> None:
    returns = np.array(
        [[0.1, -0.2], [0.0, 0.3], [-0.4, 0.0]],
        dtype=float,
    )
    rs_pos, rs_neg = realized_semivariances(returns)
    np.testing.assert_allclose(rs_pos + rs_neg, np.sum(returns**2, axis=0))
    np.testing.assert_allclose(rs_pos, np.array([0.01, 0.09]))
    np.testing.assert_allclose(rs_neg, np.array([0.16, 0.04]))


def test_tail_summary_splits_a_known_sign_pattern() -> None:
    # Nine ordinary values and one large positive. The 90th percentile of |r|
    # leaves only the large positive beyond the threshold.
    values = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, 5.0])
    summary = tail_sign_summary(values, 0.9)
    assert summary["n_up"] == 1
    assert summary["n_down"] == 0
    assert summary["mean_abs_up"] == 5.0
    assert summary["binomial_p"] == pytest.approx(1.0)
    assert bias_corrected_skewness(values) > 0.0


@pytest.mark.skipif(not PANEL_PATH.is_file() or not CACHE_ROOT.is_dir(), reason="Binance panel or cache is absent")
def test_five_minute_preconfirm_matches_production_panel() -> None:
    measurements = build_preconfirm_interval_measurements(CACHE_ROOT, intervals=(5,))
    assert len(measurements) == 1
    panel = measurements[0]
    assert panel.dates[-1] == dt.date(2025, 4, 12)
    assert HALT_DATE not in panel.dates
    stored = production_rcov_through_screen(PANEL_PATH, panel.dates)
    np.testing.assert_array_equal(panel.rcov, stored)
    assert panel.returns.shape[1] == 288
