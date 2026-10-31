"""Weight, turnover, and cost identities for the GMV book."""

from __future__ import annotations

import numpy as np

from covharness.portfolio.gmv import (
    drift_weights,
    fleming_kirby_ostdiek_fee,
    long_only_gmv_weights,
    subtract_turnover_cost,
    turnover,
    unconstrained_gmv_weights,
)


def test_weights_sum_to_one_and_identity_covariance_is_equal_weight():
    covariance = np.array(
        [[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.05]]
    )
    weights = unconstrained_gmv_weights(covariance)
    assert np.isclose(weights.sum(), 1.0)
    equal = unconstrained_gmv_weights(np.eye(4))
    assert np.allclose(equal, np.full(4, 0.25))


def test_turnover_is_zero_for_constant_weights_and_zero_returns():
    weights = np.full(3, 1.0 / 3.0)
    drifted = drift_weights(weights, np.zeros(3))
    assert turnover(weights, drifted) == 0.0


def test_long_only_matches_unconstrained_when_weights_are_nonnegative():
    covariance = np.diag([1.0, 2.0, 4.0])
    free = unconstrained_gmv_weights(covariance)
    assert np.min(free) > 0.0
    constrained = long_only_gmv_weights(covariance)
    assert np.allclose(constrained, free, atol=1e-6)
    assert np.isclose(constrained.sum(), 1.0)


def test_gross_performance_fee_matches_small_return_variance_gap():
    scale = 1e-3
    returns_a = scale * np.tile([1.0, -1.0], 2000)
    returns_b = 2.0 * scale * np.tile([1.0, -1.0], 2000)
    gamma = 10.0
    fee = fleming_kirby_ostdiek_fee(returns_a, returns_b, gamma)
    target = (gamma / 2.0) * (
        float(returns_b.var(ddof=0)) - float(returns_a.var(ddof=0))
    )
    assert abs(fee - target) / abs(target) < 1e-3
    # The package solves the gross-return quadratic, not the net-return one.
    theta = gamma / (2.0 * (1.0 + gamma))
    gross_a = 1.0 + returns_a
    gross_b = 1.0 + returns_b
    gross_root = _utility_root(gross_a, gross_b, theta, fee)
    net_root = _utility_root(returns_a, returns_b, theta, fee)
    assert abs(fee - gross_root) < 1e-12
    assert abs(fee - net_root) / abs(target) > 1e-3


def _utility_root(series_a, series_b, theta, anchor):
    mean_a = float(np.mean(series_a))
    mean_b = float(np.mean(series_b))
    second_a = float(np.mean(np.asarray(series_a) ** 2))
    second_b = float(np.mean(np.asarray(series_b) ** 2))
    roots = np.roots(
        [
            theta,
            1.0 - 2.0 * theta * mean_a,
            mean_b - theta * second_b - mean_a + theta * second_a,
        ]
    )
    real_roots = np.real(roots[np.isclose(roots.imag, 0.0)])
    return float(real_roots[np.argmin(np.abs(real_roots - anchor))])


def test_net_returns_fall_by_basis_points_times_turnover():
    gross = np.array([0.01, -0.02, 0.00])
    traded = np.array([0.0, 0.4, 1.5])
    for bps in (5.0, 10.0):
        net = subtract_turnover_cost(gross, traded, bps)
        assert np.allclose(gross - net, (bps * 1e-4) * traded)
