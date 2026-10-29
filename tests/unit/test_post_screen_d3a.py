"""Tests for shifted-day training panels. CONFIRM labels are not built."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from covharness.data.binance_calendar import HALT_DATE, SEGMENT_BOUNDS, dates_through_screen
from covharness.realized.binance_panel import (
    N_ASSETS,
    N_INTERVALS,
    build_preconfirm_phase_panels,
    measure_day_on_interval,
    measure_phase_day,
    phase_ends_by_label_close,
    phase_touches_halt,
    phase_usable_at_origin,
    production_rcov_through_screen,
)
from covharness.realized.rcov import realized_covariance

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "binance_five_asset_panel.npz"
CACHE_ROOT = REPO_ROOT / "data" / "cache" / "open_data" / "binance" / "spot" / "monthly" / "klines" / "1m"


def _positive_cube(n_days: int = 4) -> np.ndarray:
    rng = np.random.default_rng(20261028)
    shock = 0.001 * rng.normal(size=(N_ASSETS, n_days, 1440))
    path = np.exp(np.cumsum(shock, axis=2))
    level = np.array([100.0, 50.0, 20.0, 10.0, 2.0], dtype=float)
    return level[:, None, None] * path


def test_halt_rule_keeps_phase_zero_and_drops_shifted_march_25() -> None:
    kept = dt.date(2023, 3, 25)
    after = dt.date(2023, 3, 26)
    before = dt.date(2023, 3, 23)
    assert phase_touches_halt(kept, 0) is False
    assert phase_touches_halt(HALT_DATE, 0) is True
    for hour in (3, 6, 9, 12, 15, 18, 21):
        assert phase_touches_halt(kept, hour) is True
        assert phase_touches_halt(after, hour) is False
        assert phase_touches_halt(before, hour) is False


def test_no_phase_day_ends_after_its_label_or_is_early() -> None:
    day = dt.date(2024, 6, 1)
    previous = day - dt.timedelta(days=1)
    for hour in (0, 3, 6, 9, 12, 15, 18, 21):
        assert phase_ends_by_label_close(day, hour)
        assert phase_usable_at_origin(day, hour, day)
        assert phase_usable_at_origin(day, hour, previous) is False


def test_phase_zero_matches_the_production_extractor() -> None:
    start = dt.date(2024, 1, 1)
    cube = _positive_cube()
    day = dt.date(2024, 1, 2)
    phase_returns, phase_matrix, _rq, phase_daily = measure_phase_day(cube, start, day, 0)
    returns, matrix, daily = measure_day_on_interval(cube, start, day, 5)
    np.testing.assert_array_equal(phase_returns, returns)
    np.testing.assert_array_equal(phase_matrix, matrix)
    np.testing.assert_array_equal(phase_daily, daily)
    assert phase_returns.shape == (N_INTERVALS, N_ASSETS)


def test_shifted_phase_has_288_returns_and_matches_the_gram_matrix() -> None:
    start = dt.date(2024, 1, 1)
    cube = _positive_cube()
    day = dt.date(2024, 1, 3)
    for hour in (3, 6, 9, 12, 15, 18, 21):
        returns, matrix, quarticity, day_return = measure_phase_day(cube, start, day, hour)
        assert returns.shape == (N_INTERVALS, N_ASSETS)
        np.testing.assert_array_equal(matrix, realized_covariance(returns))
        assert quarticity.shape == (N_ASSETS,)
        assert np.all(quarticity >= 0.0)
        boundary = hour * 60 - 1
        anchor = cube[:, 1, boundary]
        last = cube[:, 2, boundary]
        np.testing.assert_allclose(day_return, np.log(last) - np.log(anchor))


@pytest.mark.skipif(
    not PANEL_PATH.is_file() or not CACHE_ROOT.is_dir(),
    reason="Binance panel or cache is absent",
)
def test_phase_zero_preconfirm_matches_production_panel() -> None:
    panels = build_preconfirm_phase_panels(CACHE_ROOT, hours=(0,))
    assert len(panels) == 1
    panel = panels[0]
    assert panel.dates == dates_through_screen()
    assert panel.dropped_dates == ()
    assert HALT_DATE not in panel.dates
    assert panel.dates[-1] == SEGMENT_BOUNDS["SCREEN"][1]
    assert all(day < SEGMENT_BOUNDS["CONFIRM"][0] for day in panel.dates)
    stored = production_rcov_through_screen(PANEL_PATH, panel.dates)
    np.testing.assert_array_equal(panel.rcov, stored)
    halt_neighbors = (
        panel.dates.index(dt.date(2023, 3, 23)),
        panel.dates.index(dt.date(2023, 3, 25)),
    )
    assert halt_neighbors[1] == halt_neighbors[0] + 1
    assert bool(panel.contiguous_with_previous[halt_neighbors[1]]) is False
