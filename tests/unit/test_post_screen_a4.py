"""Synthetic sign recovery for the descriptive leverage regression."""

from __future__ import annotations

import numpy as np

from covharness.inference.leverage import (
    fit_leverage_ols,
    har_leverage_design,
)


def _planted_sample(g_pos: float, g_neg: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_obs = 420
    variance = np.exp(rng.normal(scale=0.4, size=n_obs))
    daily_return = rng.normal(scale=0.03, size=n_obs)
    # Force both signs so neither signed column is identically zero.
    daily_return[: n_obs // 2] = np.abs(daily_return[: n_obs // 2]) + 0.01
    daily_return[n_obs // 2 :] = -np.abs(daily_return[n_obs // 2 :]) - 0.01
    index = np.arange(22, n_obs, dtype=np.intp)
    design, _response = har_leverage_design(variance, daily_return, index)
    true = np.array([0.05, 0.30, 0.20, 0.10, g_pos, g_neg], dtype=float)
    response = design @ true + rng.normal(scale=1e-4, size=design.shape[0])
    return design, response


def test_planted_negative_gap_is_recovered() -> None:
    design, response = _planted_sample(g_pos=0.15, g_neg=-0.80, seed=20261028)
    fit = fit_leverage_ols(design, response)
    g_pos = float(fit.coefficients[4])
    g_neg = float(fit.coefficients[5])
    assert g_neg < 0.0
    assert abs(g_neg) > g_pos
    assert fit.gap < 0.0
    assert fit.leverage_label
    assert abs(g_pos - 0.15) < 0.05
    assert abs(g_neg + 0.80) < 0.05
    assert fit.gap_ci_high < 0.0


def test_planted_positive_g_neg_is_not_flipped() -> None:
    design, response = _planted_sample(g_pos=0.10, g_neg=0.40, seed=20261029)
    fit = fit_leverage_ols(design, response)
    assert float(fit.coefficients[5]) > 0.0
    assert not fit.leverage_label


def test_design_does_not_read_the_response_date() -> None:
    variance = np.exp(np.linspace(0.0, 0.2, 40))
    returns = np.linspace(-0.02, 0.02, 40)
    index = np.array([22, 30], dtype=np.intp)
    design, response = har_leverage_design(variance, returns, index)
    # The last target is not a lag of either included response.
    changed = variance.copy()
    changed[30] = changed[30] * 10.0
    design_after, response_after = har_leverage_design(changed, returns, index)
    np.testing.assert_array_equal(design, design_after)
    assert not np.array_equal(response, response_after)
    # An earlier target is a lag of the later response, and only of that row.
    changed[22] = changed[22] * 10.0
    design_lag, _response_lag = har_leverage_design(changed, returns, index)
    np.testing.assert_array_equal(design_lag[0], design[0])
    assert not np.array_equal(design_lag[1], design[1])
