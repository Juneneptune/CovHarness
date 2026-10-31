"""Weight, turnover, and cost identities for the GMV book."""

from __future__ import annotations

import numpy as np

from covharness.portfolio.gmv import (
    drift_weights,
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


def test_net_returns_fall_by_basis_points_times_turnover():
    gross = np.array([0.01, -0.02, 0.00])
    traded = np.array([0.0, 0.4, 1.5])
    for bps in (5.0, 10.0):
        net = subtract_turnover_cost(gross, traded, bps)
        assert np.allclose(gross - net, (bps * 1e-4) * traded)
