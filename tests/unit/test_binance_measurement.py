"""Unit tests for the distinct Binance five-minute measurement path."""

from __future__ import annotations

import datetime as dt
import inspect

import numpy as np
import pytest

from covharness.data.binance_calendar import (
    BINANCE_ASSETS,
    EXPECTED_PRODUCTION_DATES,
    EXPECTED_SEGMENT_COUNTS,
    FIVE_MINUTE_END_MINUTES,
    HALT_DATE,
    production_dates,
    segment_label,
    verify_frozen_segment_counts,
)
from covharness.realized.binance_panel import (
    N_INTERVALS,
    RQ_SCALE,
    InvalidBinanceMeasurementError,
    daily_log_return,
    five_minute_log_returns,
    realized_quarticity,
)
from covharness.realized.rcov import realized_covariance


def _hand_built_endpoints(n_assets: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 288 endpoint closes and a prior 23:59 close vector."""
    prior = np.array([100.0, 50.0, 20.0, 10.0, 2.0], dtype=float)[:n_assets]
    # Distinct positive levels so outer products are nontrivial.
    growth = 1.0 + 1e-4 * np.arange(1, N_INTERVALS + 1, dtype=float)
    endpoints = prior[None, :] * growth[:, None]
    endpoints = endpoints * (1.0 + 0.01 * np.arange(n_assets)[None, :])
    return prior, endpoints


def test_frozen_calendar_counts() -> None:
    counts = verify_frozen_segment_counts()
    assert counts == EXPECTED_SEGMENT_COUNTS
    dates = production_dates()
    assert len(dates) == EXPECTED_PRODUCTION_DATES
    assert HALT_DATE not in dates
    assert dates[0] == dt.date(2021, 11, 9)
    assert dates[-1] == dt.date(2026, 8, 25)
    assert segment_label(dt.date(2023, 3, 23)) == "VALIDATION"
    assert segment_label(dt.date(2023, 3, 25)) == "HISTORY_B"
    with pytest.raises(Exception, match="halt"):
        segment_label(HALT_DATE)


def test_halt_day_is_not_a_production_date() -> None:
    assert HALT_DATE not in production_dates()
    val_end = dt.date(2023, 3, 23)
    hist_b0 = dt.date(2023, 3, 25)
    dates = production_dates()
    assert val_end in dates and hist_b0 in dates
    assert (hist_b0 - val_end).days == 2
    assert segment_label(val_end) == "VALIDATION"
    assert segment_label(hist_b0) == "HISTORY_B"


def test_five_minute_endpoint_identity() -> None:
    assert len(FIVE_MINUTE_END_MINUTES) == 288
    assert FIVE_MINUTE_END_MINUTES[0] == 4
    assert FIVE_MINUTE_END_MINUTES[-1] == 1439


def test_288_returns_from_hand_built_grid() -> None:
    prior, endpoints = _hand_built_endpoints()
    returns = five_minute_log_returns(endpoints, prior)
    assert returns.shape == (288, 5)
    expected = np.diff(np.log(np.vstack([prior, endpoints])), axis=0)
    np.testing.assert_allclose(returns, expected, rtol=0.0, atol=0.0)


def test_first_return_uses_prior_2359_anchor() -> None:
    prior, endpoints = _hand_built_endpoints()
    returns = five_minute_log_returns(endpoints, prior)
    expected = np.log(endpoints[0]) - np.log(prior)
    np.testing.assert_allclose(returns[0], expected, rtol=0.0, atol=0.0)


def test_final_return_ends_at_current_2359() -> None:
    prior, endpoints = _hand_built_endpoints()
    returns = five_minute_log_returns(endpoints, prior)
    expected = np.log(endpoints[-1]) - np.log(endpoints[-2])
    np.testing.assert_allclose(returns[-1], expected, rtol=0.0, atol=0.0)
    assert FIVE_MINUTE_END_MINUTES[-1] == 1439
    assert FIVE_MINUTE_END_MINUTES[-2] == 1434


def test_asset_order_is_preserved() -> None:
    assert BINANCE_ASSETS == (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "LTCUSDT",
        "ADAUSDT",
    )
    prior, endpoints = _hand_built_endpoints()
    returns = five_minute_log_returns(endpoints, prior)
    # Permuting columns of the input permutes columns of the output the same way.
    perm = np.array([4, 0, 3, 1, 2])
    returns_perm = five_minute_log_returns(endpoints[:, perm], prior[perm])
    np.testing.assert_allclose(returns_perm, returns[:, perm], rtol=0.0, atol=0.0)


def test_hand_built_rcov_equals_sum_of_outer_products() -> None:
    prior, endpoints = _hand_built_endpoints()
    returns = five_minute_log_returns(endpoints, prior)
    rcov = realized_covariance(returns)
    expected = np.zeros((5, 5))
    for row in returns:
        expected += np.outer(row, row)
    # BLAS Gram and a Python outer-product loop differ at float64 roundoff.
    np.testing.assert_allclose(rcov, expected, rtol=1e-15, atol=1e-18)
    np.testing.assert_allclose(rcov, returns.T @ returns, rtol=0.0, atol=0.0)


def test_hand_built_rq_matches_formula() -> None:
    prior, endpoints = _hand_built_endpoints()
    returns = five_minute_log_returns(endpoints, prior)
    rq = realized_quarticity(returns)
    expected = RQ_SCALE * np.sum(returns**4, axis=0)
    np.testing.assert_allclose(rq, expected, rtol=0.0, atol=0.0)
    assert rq.shape == (5,)
    assert np.all(rq >= 0.0)


def test_daily_return_uses_prior_and_current_2359() -> None:
    prior, endpoints = _hand_built_endpoints()
    daily = daily_log_return(endpoints[-1], prior)
    expected = np.log(endpoints[-1]) - np.log(prior)
    np.testing.assert_allclose(daily, expected, rtol=0.0, atol=0.0)


def test_missing_required_endpoint_raises_rather_than_fills() -> None:
    prior, endpoints = _hand_built_endpoints()
    endpoints[10, 2] = np.nan
    with pytest.raises(InvalidBinanceMeasurementError, match="finite"):
        five_minute_log_returns(endpoints, prior)
    prior_bad = prior.copy()
    prior_bad[0] = np.nan
    with pytest.raises(InvalidBinanceMeasurementError, match="finite"):
        five_minute_log_returns(endpoints * 0.0 + 1.0, prior_bad)
    from covharness.realized.binance_panel import N_ASSETS, _extract_day_endpoints

    start = dt.date(2023, 3, 24)
    cube = np.ones((N_ASSETS, 2, 1440), dtype=float)
    cube[0, 1, 4] = np.nan
    with pytest.raises(InvalidBinanceMeasurementError, match="missing or invalid"):
        _extract_day_endpoints(cube, start, dt.date(2023, 3, 25))


def test_march_25_uses_calendar_halt_day_2359_anchor() -> None:
    """2023-03-24 is excluded as a statistical day but remains the calendar t-1."""
    halt_close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    march25_endpoints = np.tile(np.array([1.1, 2.1, 3.1, 4.1, 5.1]), (288, 1))
    march25_endpoints[-1] = np.array([1.2, 2.2, 3.2, 4.2, 5.2])
    returns = five_minute_log_returns(march25_endpoints, halt_close)
    np.testing.assert_allclose(
        returns[0],
        np.log(march25_endpoints[0]) - np.log(halt_close),
        rtol=0.0,
        atol=0.0,
    )
    daily = daily_log_return(march25_endpoints[-1], halt_close)
    np.testing.assert_allclose(
        daily,
        np.log(march25_endpoints[-1]) - np.log(halt_close),
        rtol=0.0,
        atol=0.0,
    )
    dates = production_dates()
    assert dt.date(2023, 3, 25) in dates
    assert dt.date(2023, 3, 24) not in dates


def test_panel_module_does_not_import_forecasting_models() -> None:
    import sys

    before = {name for name in sys.modules if name.startswith("covharness.models")}
    import covharness.realized.binance_panel as panel_mod

    source = inspect.getsource(panel_mod)
    assert "covharness.models" not in source
    after = {name for name in sys.modules if name.startswith("covharness.models")}
    assert after == before
