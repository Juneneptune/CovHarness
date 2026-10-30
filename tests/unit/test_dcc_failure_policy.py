"""Carry-forward and interior-map policies for DCC stationarity failures."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.models.dcc import DCCCovariance
from covharness.models.exceptions import InvalidModelForecastError
from covharness.models.stationarity import (
    NEAR_BOUNDARY_SUM,
    STATIONARITY_EPSILON,
    softmax_persistence,
)


def _boundary_window(n_observations: int = 120) -> np.ndarray:
    """Two-regime returns whose stage-one QMLE sits on a+b = 1.

    Seed 4 is the smallest seed in a fixed search for which
    ``fit_stage_one_garch`` rejects asset 0 under the strict rule.
    """
    local = np.random.default_rng(4)
    columns = []
    for _asset in range(2):
        shock = local.normal(scale=1e-4, size=n_observations)
        shock[n_observations // 2 :] = local.normal(scale=0.05, size=n_observations // 2)
        columns.append(shock)
    return np.column_stack(columns)


def _interior_window(n_assets: int = 2, n_observations: int = 80) -> np.ndarray:
    local = np.random.default_rng(0)
    return local.normal(scale=0.01, size=(n_observations, n_assets))


def test_softmax_map_stays_below_one() -> None:
    mapped = softmax_persistence(np.array([-40.0, -40.0, 40.0]))
    assert mapped.alpha >= 0.0
    assert mapped.beta >= 0.0
    assert mapped.alpha + mapped.beta <= 1.0 - STATIONARITY_EPSILON + 1e-15
    assert mapped.alpha + mapped.beta > NEAR_BOUNDARY_SUM


def test_carry_forward_keeps_parameters_and_moves_state() -> None:
    model = DCCCovariance(failure_policy="carry_forward")
    model.fit(_interior_window())
    before = model.fit_state
    omega = before.omega.copy()
    alpha = before.garch_alpha.copy()
    beta = before.garch_beta.copy()
    target = before.target_c.copy()
    variance = before.current_variance.copy()
    model.fit(_boundary_window())
    after = model.fit_state
    assert np.array_equal(after.omega, omega)
    assert np.array_equal(after.garch_alpha, alpha)
    assert np.array_equal(after.garch_beta, beta)
    assert after.dcc_alpha == before.dcc_alpha
    assert after.dcc_beta == before.dcc_beta
    assert np.array_equal(after.target_c, target)
    assert not np.array_equal(after.current_variance, variance)
    assert model.policy_events
    assert model.policy_events[-1]["policy"] == "carry_forward"
    forecast = model.forecast()
    assert forecast.identity.configuration.get("repaired", False) is False


def test_carry_forward_first_refit_failure_raises() -> None:
    model = DCCCovariance(failure_policy="carry_forward")
    with pytest.raises(InvalidModelForecastError, match="a\\+b<1"):
        model.fit(_boundary_window())
    assert model.policy_events == []


def test_interior_map_never_reaches_the_boundary() -> None:
    model = DCCCovariance(failure_policy="interior_map")
    model.fit(_boundary_window())
    state = model.fit_state
    assert np.all(state.garch_alpha + state.garch_beta < 1.0)
    assert state.dcc_alpha + state.dcc_beta < 1.0
    assert model.near_boundary_count >= 1
    forecast = model.forecast()
    assert forecast.identity.configuration.get("repaired", False) is False


def test_reject_policy_still_fails_on_the_boundary() -> None:
    model = DCCCovariance()
    with pytest.raises(InvalidModelForecastError, match="a\\+b<1"):
        model.fit(_boundary_window())
