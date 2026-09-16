"""Giacomini-Rossi fluctuation tests on synthetic loss differentials."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.inference import (
    FLUCTUATION_CRITICAL_VALUE,
    FLUCTUATION_MU,
    DegenerateFluctuationVarianceError,
    giacomini_rossi_fluctuation,
    hac_long_run_variance,
)
from covharness.inference.fluctuation import (
    even_centered_window_length,
    uncentered_bartlett_long_run_variance,
)
from covharness.inference.hac import bartlett_weights, newey_west_1994_lags, sample_autocovariances


def test_fluctuation_window_length_is_even_floor_rule() -> None:
    assert even_centered_window_length(500) == 150
    assert even_centered_window_length(10) == 2
    assert even_centered_window_length(11) == 2
    window = even_centered_window_length(100)
    assert window % 2 == 0
    assert window == 2 * int(np.floor(0.30 * 100 / 2.0))


def test_fluctuation_local_window_hand_calculation() -> None:
    loss_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    loss_b = np.zeros(10)
    differential = loss_a - loss_b
    result = giacomini_rossi_fluctuation(loss_a, loss_b)
    assert result.window_length == 2
    assert result.target_mu == FLUCTUATION_MU
    assert result.realized_mu == pytest.approx(2 / 10)
    assert result.center_indices[0] == 1
    assert result.center_indices[-1] == 9
    assert result.path.shape[0] == 9
    first_sum = differential[0] + differential[1]
    long_run, _lag = uncentered_bartlett_long_run_variance(differential)
    expected = first_sum / (np.sqrt(long_run) * np.sqrt(2.0))
    assert result.path[0] == pytest.approx(expected)
    assert result.sign_convention.startswith("d_t = L_A,t - L_B,t")


def test_uncentered_lrv_matches_hand_gamma() -> None:
    series = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    lag = newey_west_1994_lags(4)
    gammas = sample_autocovariances(series, lag)
    weights = bartlett_weights(lag)
    omega_hand = float(gammas[0] + 2.0 * np.dot(weights, gammas[1:]))
    omega, applied_lag = uncentered_bartlett_long_run_variance(series)
    assert applied_lag == lag
    assert omega == pytest.approx(omega_hand)
    assert gammas[0] == pytest.approx(np.mean(series * series))


def test_uncentered_lrv_differs_from_demeaned_block3a_hac() -> None:
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    uncentered, _lag = uncentered_bartlett_long_run_variance(series)
    demeaned = hac_long_run_variance(series)
    assert uncentered != pytest.approx(demeaned.long_run_variance)
    assert series.mean() != pytest.approx(0.0)


def test_fluctuation_stable_null_does_not_reject() -> None:
    rng = np.random.default_rng(20260914)
    n_obs = 400
    differential = rng.normal(scale=0.4, size=n_obs)
    result = giacomini_rossi_fluctuation(differential, np.zeros(n_obs))
    assert result.reject is False
    assert result.max_abs_statistic < FLUCTUATION_CRITICAL_VALUE


def test_fluctuation_known_break_changes_sign() -> None:
    n_obs = 200
    differential = np.concatenate([np.full(100, -0.8), np.full(100, 0.8)])
    result = giacomini_rossi_fluctuation(differential, np.zeros(n_obs))
    early = result.path[result.center_indices < 80]
    late = result.path[result.center_indices > 120]
    assert np.mean(early) < 0.0
    assert np.mean(late) > 0.0
    assert result.reject is True


def test_fluctuation_path_length_and_endpoints() -> None:
    n_obs = 50
    differential = np.linspace(-1.0, 1.0, n_obs)
    result = giacomini_rossi_fluctuation(differential, np.zeros(n_obs))
    window = result.window_length
    half = window // 2
    assert result.path.shape[0] == n_obs - window + 1
    np.testing.assert_array_equal(
        result.center_indices, np.arange(half, n_obs - half + 1)
    )
    assert result.center_indices[0] == half
    assert result.center_indices[-1] == n_obs - half
    for index in range(half):
        assert index not in result.center_indices
    assert 0 not in result.center_indices


def test_fluctuation_between_pointwise_and_scan_critical_value_does_not_reject() -> None:
    # Amplitude cancels in F_t, so the path depends on shape. Search seeds
    # until max|F| lies strictly between the pointwise 1.96 and Table I 3.012.
    found = None
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        n_obs = 180
        differential = rng.normal(size=n_obs)
        result = giacomini_rossi_fluctuation(differential, np.zeros(n_obs))
        if 1.96 < result.max_abs_statistic < FLUCTUATION_CRITICAL_VALUE:
            found = result
            break
    assert found is not None
    assert found.reject is False


def test_fluctuation_threshold_is_exactly_3012() -> None:
    assert FLUCTUATION_CRITICAL_VALUE == 3.012
    n_obs = 40
    differential = np.linspace(-0.2, 0.2, n_obs)
    result = giacomini_rossi_fluctuation(differential, np.zeros(n_obs))
    assert result.critical_value == 3.012
    assert result.reject is (result.max_abs_statistic > 3.012)
    assert not (3.012 > 3.012)


def test_fluctuation_does_not_mutate_inputs() -> None:
    loss_a = np.linspace(1.0, 2.0, 30)
    loss_b = np.linspace(0.5, 1.5, 30)
    a_copy = loss_a.copy()
    b_copy = loss_b.copy()
    giacomini_rossi_fluctuation(loss_a, loss_b)
    np.testing.assert_array_equal(loss_a, a_copy)
    np.testing.assert_array_equal(loss_b, b_copy)


def test_zero_differential_is_rejected_as_degenerate_variance() -> None:
    with pytest.raises(DegenerateFluctuationVarianceError, match="zero"):
        giacomini_rossi_fluctuation(np.zeros(20), np.zeros(20))


def test_uncentered_lrv_bartlett_stays_nonnegative_on_adversarial_series() -> None:
    # Bartlett Newey-West is positive semi-definite, so a non-roundoff
    # negative long-run variance cannot be constructed without changing
    # the estimator. The negative-value branch remains a contract guard.
    alternating = np.array([(-1.0) ** t for t in range(40)])
    omega, _lag = uncentered_bartlett_long_run_variance(alternating)
    assert omega > 0.0
    omega_full, _lag = uncentered_bartlett_long_run_variance(
        alternating, maxlags=39
    )
    assert omega_full >= 0.0
