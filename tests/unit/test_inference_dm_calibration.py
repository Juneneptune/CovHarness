"""Synthetic DM HAC calibration helpers. The DM statistic is not reimplemented."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from covharness.inference import (
    CALIBRATION_LAG_RULES,
    CALIBRATION_N_REPS,
    CALIBRATION_RHOS,
    CALIBRATION_SAMPLE_SIZES,
    CALIBRATION_SEED,
    ar1_true_long_run_variance,
    calibration_hac_lag,
    monte_carlo_interval,
    monte_carlo_standard_error,
    newey_west_1994_lags,
    simulate_dm_hac_calibration,
    simulate_hac_size_power,
    stationary_ar1_paths,
)
from covharness.inference import size as size_module
from covharness.inference.hac import hac_long_run_variance
from covharness.inference.size import (
    LAG_RULE_AUTO,
    LAG_RULE_FOUR_AUTO,
    LAG_RULE_L0,
    LAG_RULE_TWO_AUTO,
    SIZE_ALTERNATIVE_MEAN,
    SIZE_N_OBS,
    SIZE_N_REPS,
    SIZE_SEED,
    dm_maxlags_argument,
    score_dm_size_on_paths,
)


def test_stationary_ar1_generator_is_deterministic_under_seed() -> None:
    first = stationary_ar1_paths(n_reps=3, n_obs=12, rho=0.6, seed=17)
    second = stationary_ar1_paths(n_reps=3, n_obs=12, rho=0.6, seed=17)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 12)


def test_rho_zero_generator_matches_iid_normal_draws() -> None:
    seed = 19
    n_reps, n_obs = 5, 9
    generated = stationary_ar1_paths(n_reps=n_reps, n_obs=n_obs, rho=0.0, seed=seed)
    expected = np.random.default_rng(seed).normal(size=(n_reps, n_obs))
    np.testing.assert_array_equal(generated, expected)


def test_theoretical_ar1_lrv_matches_independent_acf_sum() -> None:
    rho = 0.6
    gamma0 = 1.0 / (1.0 - rho * rho)
    independent = gamma0
    for lag in range(1, 4001):
        independent += 2.0 * (rho**lag) * gamma0
    assert ar1_true_long_run_variance(rho) == pytest.approx(1.0 / (1.0 - rho) ** 2)
    assert ar1_true_long_run_variance(rho) == pytest.approx(independent, rel=1e-10)
    assert ar1_true_long_run_variance(0.0) == pytest.approx(1.0)


def test_automatic_lag_matches_newey_west_1994_formula() -> None:
    for n_obs in (250, 500, 1000):
        expected = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
        assert newey_west_1994_lags(n_obs) == expected
        assert calibration_hac_lag(n_obs, LAG_RULE_AUTO) == expected
    assert newey_west_1994_lags(250) == 4
    assert newey_west_1994_lags(500) == 5
    assert newey_west_1994_lags(1000) == 6


def test_lag_multipliers_produce_the_frozen_grid() -> None:
    assert CALIBRATION_LAG_RULES == (LAG_RULE_L0, LAG_RULE_AUTO, LAG_RULE_TWO_AUTO, LAG_RULE_FOUR_AUTO)
    assert calibration_hac_lag(250, LAG_RULE_L0) == 0
    assert calibration_hac_lag(250, LAG_RULE_AUTO) == 4
    assert calibration_hac_lag(250, LAG_RULE_TWO_AUTO) == 8
    assert calibration_hac_lag(250, LAG_RULE_FOUR_AUTO) == 16
    assert calibration_hac_lag(500, LAG_RULE_FOUR_AUTO) == 20
    assert calibration_hac_lag(1000, LAG_RULE_FOUR_AUTO) == 24
    assert dm_maxlags_argument(250, LAG_RULE_AUTO) is None
    assert dm_maxlags_argument(250, LAG_RULE_L0) == 0


def test_common_random_paths_are_reused_across_bandwidth_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    original = size_module._stationary_ar1

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(size_module, "_stationary_ar1", wrapped)
    table = simulate_dm_hac_calibration(
        sample_sizes=(20,),
        rhos=(0.3,),
        lag_rules=CALIBRATION_LAG_RULES,
        n_reps=6,
        seed=21,
    )
    assert calls["n"] == 1
    assert table["T"].nunique() == 1
    assert table["rho"].nunique() == 1
    assert list(table["lag_rule"]) == list(CALIBRATION_LAG_RULES)


def test_output_table_has_one_row_per_requested_cell() -> None:
    sizes = (24, 36)
    rhos = (0.0, 0.8)
    table = simulate_dm_hac_calibration(
        sample_sizes=sizes,
        rhos=rhos,
        lag_rules=CALIBRATION_LAG_RULES,
        n_reps=5,
        seed=23,
    )
    assert len(table) == len(sizes) * len(rhos) * len(CALIBRATION_LAG_RULES)
    pairs = list(zip(table["T"], table["rho"], table["lag_rule"], strict=True))
    assert len(pairs) == len(set(pairs))
    assert list(table.columns) == list(size_module.CALIBRATION_TABLE_COLUMNS)


def test_mcse_and_interval_match_hand_calculation() -> None:
    rate = 0.12
    n_reps = 5000
    expected_se = float(np.sqrt(rate * (1.0 - rate) / n_reps))
    assert monte_carlo_standard_error(rate, n_reps) == pytest.approx(expected_se)
    lower, upper = monte_carlo_interval(rate, n_reps)
    assert lower == pytest.approx(rate - 1.96 * expected_se)
    assert upper == pytest.approx(rate + 1.96 * expected_se)


def test_rejection_scoring_calls_existing_diebold_mariano(monkeypatch: pytest.MonkeyPatch) -> None:
    source = inspect.getsource(score_dm_size_on_paths)
    assert "diebold_mariano(" in source
    assert "def diebold_mariano" not in source
    calls: list[int | None] = []
    original = size_module.diebold_mariano

    def wrapped(series, *, alternative="two_sided", maxlags=None):
        calls.append(maxlags)
        return original(series, alternative=alternative, maxlags=maxlags)

    monkeypatch.setattr(size_module, "diebold_mariano", wrapped)
    paths = stationary_ar1_paths(n_reps=4, n_obs=16, rho=0.0, seed=29)
    score_dm_size_on_paths(paths, maxlags=None)
    score_dm_size_on_paths(paths, maxlags=0)
    assert None in calls
    assert 0 in calls
    assert len(calls) == 8


def test_calibration_api_does_not_consume_empirical_forecasts() -> None:
    forbidden = (
        "RollingForecastRecord",
        "wrds",
        "WRDS",
        "realized_covariance",
        "score_forecast_records",
    )
    sources = (
        inspect.getsource(simulate_dm_hac_calibration)
        + inspect.getsource(stationary_ar1_paths)
        + inspect.getsource(score_dm_size_on_paths)
    )
    for token in forbidden:
        assert token not in sources
    assert "from covharness.evaluation" not in inspect.getsource(size_module)
    assert "from covharness.models" not in inspect.getsource(size_module)


def test_same_seed_reproduces_identical_table() -> None:
    kwargs = dict(
        sample_sizes=(18,),
        rhos=(0.0, 0.6),
        lag_rules=(LAG_RULE_L0, LAG_RULE_AUTO),
        n_reps=7,
        seed=31,
    )
    first = simulate_dm_hac_calibration(**kwargs)
    second = simulate_dm_hac_calibration(**kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_monte_carlo_paths() -> None:
    first = stationary_ar1_paths(n_reps=4, n_obs=20, rho=0.8, seed=33)
    second = stationary_ar1_paths(n_reps=4, n_obs=20, rho=0.8, seed=34)
    assert not np.array_equal(first, second)


def test_original_simulate_hac_size_power_contract_is_unchanged() -> None:
    signature = inspect.signature(simulate_hac_size_power)
    assert signature.parameters["n_reps"].default == SIZE_N_REPS
    assert signature.parameters["n_obs"].default == SIZE_N_OBS
    assert signature.parameters["seed"].default == SIZE_SEED
    assert signature.parameters["alternative_mean"].default == SIZE_ALTERNATIVE_MEAN
    assert SIZE_N_REPS == 2000
    assert SIZE_SEED == 20260912
    source = inspect.getsource(simulate_hac_size_power)
    assert "simulate_dm_hac_calibration" not in source


def test_l_auto_uses_existing_hac_default_lag() -> None:
    n_obs = 40
    paths = stationary_ar1_paths(n_reps=1, n_obs=n_obs, rho=0.0, seed=37)
    default = hac_long_run_variance(paths[0], maxlags=None)
    assert default.maxlags == newey_west_1994_lags(n_obs)
    assert calibration_hac_lag(n_obs, LAG_RULE_AUTO) == default.maxlags


def test_frozen_expanded_grid_constants() -> None:
    assert CALIBRATION_SAMPLE_SIZES == (250, 500, 1000)
    assert CALIBRATION_RHOS == (0.0, 0.3, 0.6, 0.8, 0.9)
    assert CALIBRATION_N_REPS == 5000
    assert CALIBRATION_SEED == 20260916
    assert len(CALIBRATION_SAMPLE_SIZES) * len(CALIBRATION_RHOS) * len(CALIBRATION_LAG_RULES) == 60
