"""Unit tests for daily realized covariance composed from existing pieces."""

import numpy as np
import pandas as pd
import pytest

from covharness.data.returns import synchronized_log_returns
from covharness.realized.daily import daily_realized_covariance
from covharness.realized import daily as daily_module
from covharness.realized.rcov import realized_covariance


def _five_minute_prices() -> pd.DataFrame:
    index = pd.date_range("2019-01-02 09:30", periods=4, freq="5min")
    return pd.DataFrame(
        {
            "A": [100.0, 101.0, 100.5, 102.0],
            "B": [50.0, 50.5, 49.5, 51.0],
            "C": [80.0, 80.0, 81.0, 80.5],
        },
        index=index,
    )


def test_daily_rcov_matches_composed_gram() -> None:
    prices = _five_minute_prices()
    returns = synchronized_log_returns(prices).to_numpy()
    expected = returns.T @ returns
    result = daily_realized_covariance(prices)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)


def test_daily_rcov_shape() -> None:
    rcov = daily_realized_covariance(_five_minute_prices())
    assert rcov.shape == (3, 3)


def test_daily_rcov_symmetry() -> None:
    rcov = daily_realized_covariance(_five_minute_prices())
    assert np.allclose(rcov, rcov.T)


def test_daily_rcov_positive_semidefinite() -> None:
    eigenvalues = np.linalg.eigvalsh(daily_realized_covariance(_five_minute_prices()))
    assert np.all(eigenvalues >= -1e-12)


def test_daily_rcov_diagonal_is_sum_of_squared_returns() -> None:
    prices = _five_minute_prices()
    returns = synchronized_log_returns(prices).to_numpy()
    rcov = daily_realized_covariance(prices)
    for i in range(returns.shape[1]):
        np.testing.assert_allclose(rcov[i, i], np.sum(returns[:, i] ** 2))


def test_daily_rcov_off_diagonal_is_sum_of_products() -> None:
    prices = _five_minute_prices()
    returns = synchronized_log_returns(prices).to_numpy()
    rcov = daily_realized_covariance(prices)
    n_assets = returns.shape[1]
    for i in range(n_assets):
        for j in range(n_assets):
            np.testing.assert_allclose(
                rcov[i, j], np.sum(returns[:, i] * returns[:, j])
            )


def test_missing_return_is_not_pairwise_deleted() -> None:
    prices = _five_minute_prices()
    prices.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        daily_realized_covariance(prices)


def test_reuses_existing_log_returns_and_realized_covariance() -> None:
    assert daily_module.synchronized_log_returns is synchronized_log_returns
    assert daily_module.realized_covariance is realized_covariance
