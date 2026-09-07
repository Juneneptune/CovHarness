"""Unit tests for synchronized realized covariance on synthetic data."""

import numpy as np

from covharness.realized.rcov import realized_covariance
from covharness.simulation.intraday import simulate_synchronized_gaussian_returns

SIGMA_DAILY = np.array(
    [
        [0.0004, 0.00012, 0.00008],
        [0.00012, 0.0003, 0.00006],
        [0.00008, 0.00006, 0.0002],
    ]
)
N_ASSETS = SIGMA_DAILY.shape[0]
N_INTERVALS = 78


def test_realized_covariance_shape() -> None:
    rng = np.random.default_rng(0)
    returns = simulate_synchronized_gaussian_returns(
        SIGMA_DAILY, N_INTERVALS, rng, n_days=1
    )
    assert returns.shape == (1, N_INTERVALS, N_ASSETS)

    rcov = realized_covariance(returns[0])
    assert rcov.shape == (N_ASSETS, N_ASSETS)


def test_realized_covariance_symmetry() -> None:
    rng = np.random.default_rng(1)
    returns = simulate_synchronized_gaussian_returns(
        SIGMA_DAILY, N_INTERVALS, rng, n_days=1
    )
    rcov = realized_covariance(returns[0])
    assert np.allclose(rcov, rcov.T)


def test_realized_covariance_positive_semidefinite() -> None:
    rng = np.random.default_rng(2)
    returns = simulate_synchronized_gaussian_returns(
        SIGMA_DAILY, N_INTERVALS, rng, n_days=1
    )
    rcov = realized_covariance(returns[0])
    eigenvalues = np.linalg.eigvalsh(rcov)
    assert np.all(eigenvalues >= -1e-12)


def test_realized_covariance_matches_manual_gram() -> None:
    returns = np.array(
        [
            [1.0, -0.5],
            [0.0, 2.0],
            [3.0, 0.25],
        ]
    )
    expected = returns.T @ returns
    result = realized_covariance(returns)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)


def test_realized_covariance_monte_carlo_expectation() -> None:
    n_days = 4000
    rng = np.random.default_rng(0)
    returns = simulate_synchronized_gaussian_returns(
        SIGMA_DAILY, N_INTERVALS, rng, n_days=n_days
    )
    daily_rcov = np.stack([realized_covariance(day) for day in returns])
    mean_rcov = daily_rcov.mean(axis=0)
    np.testing.assert_allclose(mean_rcov, SIGMA_DAILY, rtol=0.02, atol=1e-5)
