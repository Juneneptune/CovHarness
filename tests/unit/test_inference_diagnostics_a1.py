"""Synthetic checks for SCREEN loss-differential dependence helpers."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.inference import (
    ar1_effective_sample_size,
    equal_refit_phase_means_wald,
    hac_long_run_variance,
    refit_cycle_means,
    refit_phase_from_flags,
    refit_phase_index,
    sample_acf,
)


def test_sample_acf_recovers_ar1_rho_within_monte_carlo_tolerance() -> None:
    rng = np.random.default_rng(20261028)
    n_obs = 4000
    rho = 0.45
    innovations = rng.normal(size=n_obs)
    series = np.empty(n_obs)
    series[0] = innovations[0] / np.sqrt(1.0 - rho**2)
    for time in range(1, n_obs):
        series[time] = rho * series[time - 1] + innovations[time]
    acf = sample_acf(series, 30)
    assert acf.shape == (30,)
    assert acf[0] == pytest.approx(rho, abs=0.05)


def test_ar1_effective_sample_size_on_known_values() -> None:
    assert ar1_effective_sample_size(500, 0.0) == pytest.approx(500.0)
    assert ar1_effective_sample_size(500, 0.5) == pytest.approx(500.0 * 0.5 / 1.5)
    assert ar1_effective_sample_size(500, -0.2) == pytest.approx(500.0 * 1.2 / 0.8)
    assert ar1_effective_sample_size(250, 0.6) == pytest.approx(250.0 * 0.4 / 1.6)
    assert ar1_effective_sample_size(500, 1.0) is None
    assert ar1_effective_sample_size(500, -1.0) is None
    assert ar1_effective_sample_size(500, float("nan")) is None


def test_refit_phase_labels_include_partial_final_cycle() -> None:
    n_obs = 500
    cadence = 21
    phase = refit_phase_index(n_obs, cadence=cadence, first_is_refit=True)
    assert phase.shape == (n_obs,)
    assert phase[0] == 0
    assert phase[20] == 20
    assert phase[21] == 0
    np.testing.assert_array_equal(phase[483:], np.arange(17))
    counts = np.bincount(phase, minlength=cadence)
    np.testing.assert_array_equal(counts[:17], np.full(17, 24))
    np.testing.assert_array_equal(counts[17:], np.full(4, 23))
    assert int(n_obs // cadence) == 23
    assert int(n_obs % cadence) == 17

    flags = np.zeros(n_obs, dtype=bool)
    flags[::cadence] = True
    from_flags = refit_phase_from_flags(flags)
    np.testing.assert_array_equal(from_flags, phase)


def test_refit_cycle_means_match_group_averages() -> None:
    phase = refit_phase_index(42, cadence=21, first_is_refit=True)
    series = np.arange(42, dtype=float)
    result = refit_cycle_means(series, phase, cadence=21)
    assert result.n_complete_cycles == 2
    assert result.final_cycle_length == 0
    for k, count, mean in zip(result.occupied_phases, result.counts, result.means):
        mask = phase == k
        assert int(count) == int(mask.sum())
        assert mean == pytest.approx(series[mask].mean())


def test_equal_phase_mean_wald_rejects_a_known_cycle_shift() -> None:
    rng = np.random.default_rng(20261028)
    n_obs = 500
    phase = refit_phase_index(n_obs, cadence=21, first_is_refit=True)
    series = rng.normal(size=n_obs)
    series = series + 1.5 * (phase == 0).astype(float)
    result = equal_refit_phase_means_wald(series, phase, maxlags=29)
    assert result.df == 20
    assert result.p_value is not None
    assert result.p_value < 1e-6
    assert result.statistic is not None
    assert result.statistic > 0.0


def test_one_column_dummy_scores_match_scalar_hac() -> None:
    rng = np.random.default_rng(11)
    series = rng.normal(size=200)
    residuals = series - series.mean()
    scores = residuals.reshape(-1, 1)
    from covharness.inference.diagnostics import _bartlett_score_long_run_covariance

    omega = _bartlett_score_long_run_covariance(scores, maxlags=5)
    scalar = hac_long_run_variance(series, maxlags=5)
    assert omega.shape == (1, 1)
    assert omega[0, 0] == pytest.approx(scalar.long_run_variance, rel=1e-12)
