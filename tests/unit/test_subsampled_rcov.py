"""Unit tests for subsampled realized covariance."""

import numpy as np
import pandas as pd
import pytest

from covharness.realized.daily import daily_realized_covariance
from covharness.realized.subsampled import (
    subsample_offset_prices,
    subsampled_realized_covariance,
)
from covharness.realized import subsampled as subsampled_module


def _one_minute_prices(n_minutes: int = 15) -> pd.DataFrame:
    index = pd.date_range("2019-01-02 09:30", periods=n_minutes, freq="1min")
    t = np.arange(n_minutes, dtype=float)
    return pd.DataFrame(
        {
            "A": 100.0 + 0.10 * t,
            "B": 50.0 + 0.05 * t,
            "C": 80.0 - 0.02 * t,
        },
        index=index,
    )


def test_each_offset_selects_the_correct_timestamps() -> None:
    prices = _one_minute_prices(15)
    expected = {
        0: ["2019-01-02 09:30", "2019-01-02 09:35", "2019-01-02 09:40"],
        1: ["2019-01-02 09:31", "2019-01-02 09:36", "2019-01-02 09:41"],
        2: ["2019-01-02 09:32", "2019-01-02 09:37", "2019-01-02 09:42"],
        3: ["2019-01-02 09:33", "2019-01-02 09:38", "2019-01-02 09:43"],
        4: ["2019-01-02 09:34", "2019-01-02 09:39", "2019-01-02 09:44"],
    }
    for offset, stamps in expected.items():
        grid = subsample_offset_prices(prices, offset)
        assert list(grid.index) == [pd.Timestamp(s) for s in stamps]


def test_subsampled_rcov_is_mean_of_offset_rcovs() -> None:
    prices = _one_minute_prices(15)
    offset_rcovs = [
        daily_realized_covariance(subsample_offset_prices(prices, offset))
        for offset in range(5)
    ]
    expected = np.mean(np.stack(offset_rcovs, axis=0), axis=0)
    result = subsampled_realized_covariance(prices)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)
    assert subsampled_module.daily_realized_covariance is daily_realized_covariance


def test_subsampled_rcov_shape() -> None:
    assert subsampled_realized_covariance(_one_minute_prices(15)).shape == (3, 3)


def test_subsampled_rcov_symmetry() -> None:
    rcov = subsampled_realized_covariance(_one_minute_prices(15))
    assert np.allclose(rcov, rcov.T)


def test_subsampled_rcov_positive_semidefinite() -> None:
    eigenvalues = np.linalg.eigvalsh(subsampled_realized_covariance(_one_minute_prices(15)))
    assert np.all(eigenvalues >= -1e-12)


def test_constant_prices_produce_zero_matrix() -> None:
    index = pd.date_range("2019-01-02 09:30", periods=15, freq="1min")
    prices = pd.DataFrame(
        {"A": 100.0, "B": 50.0, "C": 80.0},
        index=index,
    )
    rcov = subsampled_realized_covariance(prices)
    np.testing.assert_allclose(rcov, np.zeros((3, 3)), atol=1e-15)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p.iloc[::-1], "ordered"),
        (lambda p: p.assign(A=np.inf), "finite"),
        (lambda p: p.assign(A=0.0), "strictly positive"),
        (lambda p: p.assign(A=-1.0), "strictly positive"),
    ],
)
def test_malformed_or_nonfinite_inputs_fail(mutate, match: str) -> None:
    prices = mutate(_one_minute_prices(15))
    with pytest.raises((TypeError, ValueError), match=match):
        subsampled_realized_covariance(prices)


def test_missing_values_are_not_pairwise_deleted() -> None:
    prices = _one_minute_prices(15)
    prices.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        subsampled_realized_covariance(prices)
