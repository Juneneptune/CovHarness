"""Published Bartlett fixed-b critical values and the new lag rules."""

from __future__ import annotations

import numpy as np

from covharness.inference.hac import (
    andrews_1991_bartlett_lag,
    bartlett_fixed_b_critical,
    ewc_long_run_variance,
    llsw_sqrt_lag,
    newey_west_1994_lags,
)


def test_table_matches_kiefer_vogelsang_2005() -> None:
    # Table I, b = 0.02 and b = 1. The b = 1 entries are analytical.
    assert bartlett_fixed_b_critical(0.02, 0.95) == 1.690
    assert bartlett_fixed_b_critical(0.02, 0.975) == 2.018
    assert bartlett_fixed_b_critical(1.0, 0.95) == 3.764
    assert bartlett_fixed_b_critical(1.0, 0.975) == 4.771
    assert bartlett_fixed_b_critical(1.0, 0.99) == 6.090
    # Linear interpolation halfway between b = 0.02 and b = 0.04.
    midpoint = 0.5 * (1.690 + 1.731)
    assert np.isclose(bartlett_fixed_b_critical(0.03, 0.95), midpoint)


def test_lag_rules_at_250_and_500() -> None:
    assert newey_west_1994_lags(250) == int(np.floor(4.0 * (250 / 100.0) ** (2.0 / 9.0)))
    assert newey_west_1994_lags(500) == int(np.floor(4.0 * (500 / 100.0) ** (2.0 / 9.0)))
    assert llsw_sqrt_lag(250) == int(np.floor(1.3 * np.sqrt(250)))
    assert llsw_sqrt_lag(500) == int(np.floor(1.3 * np.sqrt(500)))
    assert llsw_sqrt_lag(250) == 20
    assert llsw_sqrt_lag(500) == 29


def test_andrews_plug_in_matches_the_formula() -> None:
    series = np.array([0.2, -0.1, 0.4, 0.1, -0.3, 0.5, 0.0, 0.2], dtype=float)
    centered = series - series.mean()
    rho = float(np.dot(centered[1:], centered[:-1]) / np.dot(centered, centered))
    alpha = (4.0 * rho * rho) / (1.0 - rho * rho) ** 2
    expected = int(np.floor(1.1447 * (alpha * series.shape[0]) ** (1.0 / 3.0)))
    assert andrews_1991_bartlett_lag(series) == expected


def test_ewc_scale_on_white_noise() -> None:
    rng = np.random.default_rng(20260916)
    draws = []
    for _ in range(30):
        omega, _terms = ewc_long_run_variance(rng.normal(size=2000))
        draws.append(omega)
    mean_omega = float(np.mean(draws))
    assert 0.7 < mean_omega < 1.3
