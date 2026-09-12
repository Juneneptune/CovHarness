"""Arithmetic tests for Epps diagnostic utilities. Synthetic prices only."""

import numpy as np
import pandas as pd
import pytest

from covharness.diagnostics.epps import (
    DEFAULT_FREQUENCIES_MINUTES,
    covariance_to_correlation,
    epps_across_frequencies,
    epps_at_frequency,
    epps_curve_to_dict,
    matrix_eigen_diagnostics,
    off_diagonal_correlations,
    summarize_off_diagonal,
)
from covharness.realized.daily import daily_realized_covariance


def test_covariance_to_correlation_known_2x2() -> None:
    rcov = np.array([[4.0, 2.0], [2.0, 1.0]])
    corr = covariance_to_correlation(rcov)
    expected = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(corr, expected)


def test_covariance_to_correlation_rejects_nonsquare() -> None:
    with pytest.raises(ValueError, match="square"):
        covariance_to_correlation(np.ones((2, 3)))


def test_off_diagonal_average_excludes_diagonal() -> None:
    corr = np.array(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.6],
            [0.4, 0.6, 1.0],
        ]
    )
    values, keys = off_diagonal_correlations(corr, ("A", "B", "C"))
    np.testing.assert_allclose(values, [0.2, 0.4, 0.6])
    assert keys == ("A-B", "A-C", "B-C")
    summary = summarize_off_diagonal(values)
    assert summary.n_pairs == 3
    assert summary.n_finite == 3
    np.testing.assert_allclose(summary.mean, 0.4)
    np.testing.assert_allclose(summary.median, 0.4)
    np.testing.assert_allclose(summary.minimum, 0.2)
    np.testing.assert_allclose(summary.maximum, 0.6)
    naive_including_diagonal = float(np.mean(corr))
    assert summary.mean != pytest.approx(naive_including_diagonal)


def test_default_frequency_order() -> None:
    assert DEFAULT_FREQUENCIES_MINUTES == (1, 2, 5, 10, 15, 30)
    assert list(DEFAULT_FREQUENCIES_MINUTES) == sorted(DEFAULT_FREQUENCIES_MINUTES)


def _synchronous_minute_ticks() -> pd.DataFrame:
    """Complete one-minute prices for two assets over a short session window."""
    index = pd.date_range("2009-02-13 09:30", "2009-02-13 10:30", freq="1min")
    rows: list[tuple[pd.Timestamp, str, float]] = []
    price_a = 100.0
    price_b = 50.0
    for step, ts in enumerate(index):
        # Shared Gaussian-like move plus a small idiosyncratic term.
        common = 0.001 * np.sin(step / 3.0)
        price_a *= np.exp(common + 0.0001 * step)
        price_b *= np.exp(common - 0.00005 * step)
        rows.append((ts, "A", float(price_a)))
        rows.append((ts, "B", float(price_b)))
    return pd.DataFrame(rows, columns=["timestamp", "asset", "price"])


def test_frequency_scan_preserves_caller_order_and_structure() -> None:
    ticks = _synchronous_minute_ticks()
    frequencies = (10, 1, 5)
    curve = epps_across_frequencies(
        ticks,
        day="2009-02-13",
        assets=("A", "B"),
        frequencies_minutes=frequencies,
    )
    assert curve.interval_minutes == frequencies
    assert len(curve.by_frequency) == 3
    for item, delta in zip(curve.by_frequency, frequencies):
        assert item.interval_minutes == delta
        assert item.assets == ("A", "B")
        assert item.rcov.shape == (2, 2)
        assert item.correlation.shape == (2, 2)
        assert "A-B" in item.pairwise_correlations
        assert "A-A" not in item.pairwise_correlations
        assert item.n_returns == item.n_complete_prices - 1
        assert item.n_returns >= 1
        np.testing.assert_allclose(item.m_over_n, item.n_returns / 2.0)
    payload = epps_curve_to_dict(curve)
    assert payload["frequencies_minutes"] == list(frequencies)
    assert len(payload["by_frequency"]) == 3
    assert "m_over_n" in payload["by_frequency"][0]


def test_finer_grid_has_more_return_observations() -> None:
    ticks = _synchronous_minute_ticks()
    curve = epps_across_frequencies(
        ticks,
        day="2009-02-13",
        assets=("A", "B"),
        frequencies_minutes=(1, 5),
    )
    n_one, n_five = (item.n_returns for item in curve.by_frequency)
    assert n_one > n_five


def test_epps_uses_production_realized_covariance() -> None:
    ticks = _synchronous_minute_ticks()
    result = epps_at_frequency(
        ticks,
        5,
        day="2009-02-13",
        assets=("A", "B"),
    )
    prices = (
        ticks.pivot(index="timestamp", columns="asset", values="price")
        .sort_index()
        .reindex(columns=["A", "B"])
    )
    grid = pd.date_range("2009-02-13 09:30", "2009-02-13 16:00", freq="5min")
    synced = prices.reindex(prices.index.union(grid)).ffill().reindex(grid)
    complete = synced.dropna(how="any")
    expected = daily_realized_covariance(complete)
    np.testing.assert_allclose(result.rcov, expected)


def test_missing_open_is_dropped_not_filled() -> None:
    ticks = _synchronous_minute_ticks()
    ticks = ticks.loc[ticks["timestamp"] > pd.Timestamp("2009-02-13 09:30:00")]
    result = epps_at_frequency(
        ticks,
        1,
        day="2009-02-13",
        assets=("A", "B"),
    )
    assert result.n_complete_prices == result.n_grid_prices - 1
    assert result.n_returns == result.n_complete_prices - 1


def test_zero_variance_correlation_is_nan_and_condition_omitted() -> None:
    rcov = np.zeros((2, 2))
    corr = covariance_to_correlation(rcov)
    assert np.isnan(corr).all()
    values, _ = off_diagonal_correlations(corr, ("A", "B"))
    summary = summarize_off_diagonal(values)
    assert summary.n_pairs == 1
    assert summary.n_finite == 0
    assert np.isnan(summary.mean)
    eigvals, rank, min_eig, psd, cond, symmetry_error, m_over_n = matrix_eigen_diagnostics(rcov)
    assert rank == 0
    np.testing.assert_allclose(min_eig, 0.0)
    assert psd is True
    assert cond is None
    assert m_over_n is None
    np.testing.assert_allclose(symmetry_error, 0.0)
    np.testing.assert_allclose(eigvals, [0.0, 0.0])


def test_rank_deficient_identical_assets_remain_unrepaired() -> None:
    index = pd.date_range("2009-02-13 09:31", periods=6, freq="5min")
    rows: list[tuple[pd.Timestamp, str, float]] = []
    price = 100.0
    for step, ts in enumerate(index, start=1):
        price *= np.exp(0.01 * step)
        rows.append((ts, "A", price))
        rows.append((ts, "B", price))
    ticks = pd.DataFrame(rows, columns=["timestamp", "asset", "price"])
    result = epps_at_frequency(
        ticks,
        5,
        day="2009-02-13",
        assets=("A", "B"),
    )
    assert result.numerical_rank == 1
    assert result.psd is True
    assert result.condition_number is None
    np.testing.assert_allclose(result.correlation[0, 1], 1.0)
    np.testing.assert_allclose(result.rcov[0, 0], result.rcov[1, 1])
