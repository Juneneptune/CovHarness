"""Origin-day quarticity, BNS market jump, and measurement GW instruments."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.features import (
    JUMP_ALPHA,
    JUMP_CRITICAL_VALUE,
    InvalidOriginStateError,
    bns_market_jump,
    measurement_stress_instruments,
    realized_quarticity_aggregate,
)
from covharness.inference import (
    GW_FAMILY_SIZE,
    giacomini_white_two_specifications,
    market_state_instruments,
)
from covharness.realized.rcov import realized_covariance


def test_quarticity_hand_calculation() -> None:
    returns = np.array(
        [
            [1.0, 2.0],
            [0.5, -1.0],
            [0.0, 1.0],
        ]
    )
    result = realized_quarticity_aggregate(returns)
    rq0 = (3.0 / 3.0) * (1.0**4 + 0.5**4 + 0.0**4)
    rq1 = (3.0 / 3.0) * (2.0**4 + (-1.0) ** 4 + 1.0**4)
    assert result.asset_quarticity[0] == pytest.approx(rq0)
    assert result.asset_quarticity[1] == pytest.approx(rq1)
    assert result.aggregate == pytest.approx(0.5 * (rq0 + rq1))
    assert result.n_intervals == 3
    assert result.n_assets == 2


def test_quarticity_is_cross_sectional_mean() -> None:
    rng = np.random.default_rng(31)
    returns = rng.normal(scale=0.01, size=(12, 4))
    result = realized_quarticity_aggregate(returns)
    per_asset = (12.0 / 3.0) * np.sum(returns**4, axis=0)
    assert result.aggregate == pytest.approx(float(np.mean(per_asset)))


def test_quarticity_asset_permutation_invariance() -> None:
    rng = np.random.default_rng(32)
    returns = rng.normal(scale=0.01, size=(10, 3))
    permuted = returns[:, [2, 0, 1]]
    original = realized_quarticity_aggregate(returns)
    shuffled = realized_quarticity_aggregate(permuted)
    assert shuffled.aggregate == pytest.approx(original.aggregate)


def test_quarticity_all_zero_day_is_rejected() -> None:
    with pytest.raises(InvalidOriginStateError, match="strictly positive"):
        realized_quarticity_aggregate(np.zeros((8, 2)))


def test_quarticity_does_not_mutate_inputs() -> None:
    returns = np.array([[0.1, -0.2], [0.3, 0.0], [0.05, 0.1], [-0.1, 0.2]])
    copy = returns.copy()
    realized_quarticity_aggregate(returns)
    np.testing.assert_array_equal(returns, copy)


def test_bns_hand_calculation() -> None:
    market = np.array([0.2, 0.1, 0.3, 0.1, 0.2])
    returns = np.column_stack([market, market])
    result = bns_market_jump(returns)
    realized_variance = float(np.sum(market**2))
    bipower = float(
        np.abs(market[0]) * np.abs(market[1])
        + np.abs(market[1]) * np.abs(market[2])
        + np.abs(market[2]) * np.abs(market[3])
        + np.abs(market[3]) * np.abs(market[4])
    )
    delta = 1.0 / 5.0
    quad_sum = (
        np.abs(market[0]) * np.abs(market[1]) * np.abs(market[2]) * np.abs(market[3])
        + np.abs(market[1]) * np.abs(market[2]) * np.abs(market[3]) * np.abs(market[4])
    )
    quadpower = (1.0 / delta) * quad_sum
    mu1 = np.sqrt(2.0 / np.pi)
    continuous = (1.0 / mu1**2) * bipower
    adjustment = np.sqrt(max(1.0, quadpower / bipower**2))
    adjusted = (delta ** (-0.5)) / adjustment * (continuous / realized_variance - 1.0)
    vartheta = np.pi**2 / 4.0 + np.pi - 5.0
    standardized = adjusted / np.sqrt(vartheta)
    assert result.realized_variance == pytest.approx(realized_variance)
    assert result.bipower_raw == pytest.approx(bipower)
    assert result.quadpower == pytest.approx(quadpower)
    assert result.continuous_variance_estimate == pytest.approx(continuous)
    assert result.adjusted_ratio_statistic == pytest.approx(adjusted)
    assert result.standardized_statistic == pytest.approx(standardized)
    assert result.alpha == JUMP_ALPHA
    assert result.critical_value == pytest.approx(JUMP_CRITICAL_VALUE)
    assert result.jump_indicator == int(standardized < JUMP_CRITICAL_VALUE)


def test_bns_scale_invariance() -> None:
    rng = np.random.default_rng(33)
    returns = rng.normal(scale=0.01, size=(40, 3))
    scaled = bns_market_jump(3.0 * returns)
    original = bns_market_jump(returns)
    assert scaled.standardized_statistic == pytest.approx(
        original.standardized_statistic
    )
    assert scaled.jump_indicator == original.jump_indicator


def test_bns_continuous_example_below_threshold() -> None:
    rng = np.random.default_rng(20260914)
    returns = rng.normal(scale=0.002, size=(80, 4))
    result = bns_market_jump(returns)
    assert result.jump_indicator == 0
    assert result.standardized_statistic >= JUMP_CRITICAL_VALUE
    assert result.alpha == 0.01


def test_bns_injected_market_jump_is_detected_in_lower_tail() -> None:
    rng = np.random.default_rng(34)
    returns = rng.normal(scale=0.002, size=(80, 3))
    returns[40, :] += 0.2
    result = bns_market_jump(returns)
    assert result.standardized_statistic < JUMP_CRITICAL_VALUE
    assert result.standardized_statistic < 0.0
    assert result.jump_indicator == 1
    assert result.alpha == 0.01


def test_bns_zero_realized_variance_is_rejected() -> None:
    with pytest.raises(InvalidOriginStateError, match="realized variance"):
        bns_market_jump(np.zeros((8, 2)))


def test_bns_zero_bipower_is_rejected() -> None:
    returns = np.zeros((8, 2))
    returns[3, :] = 0.1
    with pytest.raises(InvalidOriginStateError, match="bipower"):
        bns_market_jump(returns)


def test_bns_does_not_mutate_inputs() -> None:
    rng = np.random.default_rng(35)
    returns = rng.normal(scale=0.01, size=(16, 2))
    copy = returns.copy()
    bns_market_jump(returns)
    np.testing.assert_array_equal(returns, copy)


def test_measurement_stress_instruments_match_origin_state() -> None:
    rng = np.random.default_rng(36)
    origin = rng.normal(scale=0.01, size=(5, 20, 3))
    origin[2, 10, :] += 0.15
    target = origin + rng.normal(scale=0.05, size=origin.shape)
    instruments = measurement_stress_instruments(origin)
    assert instruments.shape == (5, 3)
    np.testing.assert_allclose(instruments[:, 0], 1.0)
    for time in range(5):
        rq = realized_quarticity_aggregate(origin[time]).aggregate
        jump = bns_market_jump(origin[time]).jump_indicator
        assert instruments[time, 1] == pytest.approx(np.log(rq))
        assert instruments[time, 2] == pytest.approx(float(jump))
        assert instruments[time, 2] in (0.0, 1.0)
    target_instruments = measurement_stress_instruments(target)
    assert not np.allclose(instruments, target_instruments)


def test_measurement_stress_uses_origin_day_returns_only() -> None:
    rng = np.random.default_rng(37)
    origin = rng.normal(scale=0.01, size=(4, 16, 2))
    target = origin.copy()
    target[:, 8, :] += 1.0
    from_origin = measurement_stress_instruments(origin)
    origin_copy = origin.copy()
    origin_copy[:, 8, :] += 1.0
    from_mutated_origin = measurement_stress_instruments(origin_copy)
    unchanged_target = measurement_stress_instruments(origin)
    np.testing.assert_allclose(from_origin, unchanged_target)
    assert not np.allclose(from_origin, from_mutated_origin)
    # Changing a separate target panel cannot change origin instruments.
    _ = measurement_stress_instruments(target)
    np.testing.assert_allclose(from_origin, measurement_stress_instruments(origin))


def test_two_specification_gw_uses_frozen_measurement_matrix() -> None:
    rng = np.random.default_rng(38)
    origin_returns = rng.normal(scale=0.01, size=(8, 24, 3))
    origin_returns[1, 12, :] += 0.25
    origin_returns[5, 6, :] += 0.25
    origin_covariances = np.stack(
        [realized_covariance(origin_returns[t]) for t in range(8)]
    )
    market = market_state_instruments(origin_covariances)
    measurement = measurement_stress_instruments(origin_returns)
    loss_a = rng.normal(size=8)
    loss_b = loss_a + 0.01
    family = giacomini_white_two_specifications(
        loss_a, loss_b, market, measurement
    )
    assert family.family_size == GW_FAMILY_SIZE
    assert family.family_size == 2
    assert family.per_test_cutoff == pytest.approx(0.025)
    assert family.market.n_instruments == 3
    assert family.measurement.n_instruments == 3
    assert measurement.shape == (8, 3)
    np.testing.assert_allclose(measurement[:, 0], 1.0)


def test_measurement_stress_does_not_mutate_inputs() -> None:
    rng = np.random.default_rng(39)
    origin = rng.normal(scale=0.01, size=(3, 12, 2))
    copy = origin.copy()
    measurement_stress_instruments(origin)
    np.testing.assert_array_equal(origin, copy)
