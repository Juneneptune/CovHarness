"""One-step Giacomini-White tests on synthetic loss series."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.inference import (
    GW_ALPHA,
    GW_BONFERRONI_CUTOFF,
    GW_FAMILY_SIZE,
    DegenerateGWCovarianceError,
    RankDeficientGWInstrumentsError,
    diebold_mariano,
    giacomini_white,
    giacomini_white_two_specifications,
    market_state_instruments,
)
from covharness.losses.contracts import InvalidCovarianceMatrixError
from covharness.losses.localization import implied_correlation


def test_gw_omega_is_uncentered_outer_product() -> None:
    loss_a = np.array([1.0, 3.0, 0.0, 2.0])
    loss_b = np.array([0.0, 1.0, 1.0, 0.0])
    instruments = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.5],
            [1.0, -1.0],
        ]
    )
    differential = loss_a - loss_b
    moments = instruments * differential[:, None]
    omega_hand = moments.T @ moments / moments.shape[0]
    result = giacomini_white(loss_a, loss_b, instruments)
    np.testing.assert_allclose(result.omega, omega_hand)
    zbar = moments.mean(axis=0)
    statistic = float(moments.shape[0] * zbar @ np.linalg.solve(omega_hand, zbar))
    assert result.statistic == pytest.approx(statistic)
    assert result.estimator == "outer_product"
    assert result.horizon == 1
    assert result.alpha == GW_ALPHA
    assert result.df == 2


def test_gw_conditional_null_does_not_reject() -> None:
    rng = np.random.default_rng(20260914)
    n_obs = 600
    instruments = np.column_stack(
        [np.ones(n_obs), rng.normal(size=n_obs), rng.normal(size=n_obs)]
    )
    differential = rng.normal(scale=0.4, size=n_obs)
    result = giacomini_white(differential, np.zeros(n_obs), instruments)
    assert result.p_value > 0.05


def test_gw_detects_state_dependent_sign_reversal_missed_by_dm() -> None:
    rng = np.random.default_rng(20260914)
    n_obs = 800
    state = np.zeros(n_obs)
    for start in range(0, n_obs, 40):
        state[start : start + 40] = (start // 40) % 2
    differential = np.where(state > 0.5, -0.55, 0.55) + 0.05 * rng.normal(size=n_obs)
    instruments = np.column_stack([np.ones(n_obs), state])
    gw = giacomini_white(differential, np.zeros(n_obs), instruments)
    dm = diebold_mariano(differential)
    assert gw.p_value < 0.01
    assert dm.p_value > 0.10


def test_gw_constant_nonzero_differential_with_varying_instruments_is_valid() -> None:
    n_obs = 40
    differential = np.full(n_obs, 0.3)
    instruments = np.column_stack([np.ones(n_obs), np.linspace(-1.0, 1.0, n_obs)])
    result = giacomini_white(differential, np.zeros(n_obs), instruments)
    assert np.isfinite(result.statistic)
    assert result.p_value < 1e-6
    assert result.mean_differential == pytest.approx(0.3)


def test_gw_all_zero_moments_fail() -> None:
    n_obs = 30
    instruments = np.column_stack([np.ones(n_obs), np.arange(n_obs, dtype=float)])
    with pytest.raises(DegenerateGWCovarianceError, match="singular"):
        giacomini_white(np.zeros(n_obs), np.zeros(n_obs), instruments)


def test_gw_duplicate_instrument_columns_fail() -> None:
    n_obs = 25
    ones = np.ones(n_obs)
    instruments = np.column_stack([ones, ones])
    differential = np.linspace(-1.0, 1.0, n_obs)
    with pytest.raises(RankDeficientGWInstrumentsError, match="full column rank"):
        giacomini_white(differential, np.zeros(n_obs), instruments)


def test_gw_family_bonferroni_is_exactly_size_two() -> None:
    rng = np.random.default_rng(3)
    n_obs = 120
    loss_a = rng.normal(size=n_obs)
    loss_b = loss_a + 0.02
    market = np.column_stack([np.ones(n_obs), rng.normal(size=n_obs)])
    measurement = np.column_stack([np.ones(n_obs), rng.normal(size=n_obs)])
    family = giacomini_white_two_specifications(loss_a, loss_b, market, measurement)
    assert family.family_size == GW_FAMILY_SIZE
    assert family.family_alpha == 0.05
    assert family.per_test_cutoff == GW_BONFERRONI_CUTOFF
    assert family.p_value_market_bonferroni == pytest.approx(
        min(1.0, 2.0 * family.p_value_market)
    )
    assert family.p_value_measurement_bonferroni == pytest.approx(
        min(1.0, 2.0 * family.p_value_measurement)
    )
    assert family.market.n_instruments == 2
    assert family.measurement.n_instruments == 2


def test_gw_does_not_mutate_inputs() -> None:
    loss_a = np.array([1.0, 2.0, 3.0, 4.0])
    loss_b = np.array([1.5, 1.5, 2.5, 3.0])
    instruments = np.column_stack([np.ones(4), np.array([0.0, 1.0, -1.0, 0.5])])
    loss_a_copy = loss_a.copy()
    loss_b_copy = loss_b.copy()
    instruments_copy = instruments.copy()
    giacomini_white(loss_a, loss_b, instruments)
    np.testing.assert_array_equal(loss_a, loss_a_copy)
    np.testing.assert_array_equal(loss_b, loss_b_copy)
    np.testing.assert_array_equal(instruments, instruments_copy)


def test_gw_instrument_length_must_match_losses() -> None:
    with pytest.raises(ValueError, match="same number of dates"):
        giacomini_white(
            np.array([1.0, 2.0, 3.0]),
            np.zeros(3),
            np.ones((4, 1)),
        )


def test_market_state_instruments_use_origin_day_only() -> None:
    origin = np.array(
        [
            [[1.0, 0.2], [0.2, 1.5]],
            [[4.0, 0.8], [0.8, 1.0]],
        ]
    )
    target = origin + 10.0
    instruments = market_state_instruments(origin)
    assert instruments.shape == (2, 3)
    np.testing.assert_allclose(instruments[:, 0], 1.0)
    assert instruments[0, 1] == pytest.approx(np.log(1.25))
    corr0 = implied_correlation(origin[0], "origin")
    assert instruments[0, 2] == pytest.approx(corr0[0, 1])
    target_instruments = market_state_instruments(target)
    assert instruments[0, 1] != pytest.approx(target_instruments[0, 1])
    origin[0, 0, 0] = 99.0
    rebuilt = market_state_instruments(
        np.array([[[1.0, 0.2], [0.2, 1.5]], [[4.0, 0.8], [0.8, 1.0]]])
    )
    assert rebuilt[0, 1] == pytest.approx(np.log(1.25))


def test_market_state_does_not_mutate_input() -> None:
    origin = np.array(
        [
            [[1.0, 0.2], [0.2, 1.5]],
            [[4.0, 0.8], [0.8, 1.0]],
        ]
    )
    origin_copy = origin.copy()
    market_state_instruments(origin)
    np.testing.assert_array_equal(origin, origin_copy)


def test_market_state_nonpositive_mean_variance_raises() -> None:
    origin = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.2], [0.2, 1.0]],
        ]
    )
    with pytest.raises(InvalidCovarianceMatrixError, match="strictly positive"):
        market_state_instruments(origin)


def test_gw_nonfinite_omega_raises() -> None:
    loss_a = np.array([1.0e200, 1.0e200, 1.0e200])
    loss_b = np.zeros(3)
    instruments = np.column_stack([np.ones(3), np.array([0.0, 1.0, -1.0])])
    with pytest.raises(DegenerateGWCovarianceError, match="not finite"):
        giacomini_white(loss_a, loss_b, instruments)
