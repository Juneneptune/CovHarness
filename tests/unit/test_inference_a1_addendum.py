"""Synthetic checks for A1-addendum refit-Wald placebos and one-df contrasts."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.inference import (
    equal_refit_phase_means_wald,
    intercept_slope_wald,
    largest_absolute_indices,
    refit_phase_index,
    roll_refit_phase,
)


def test_modular_relabel_leaves_equal_mean_wald_unchanged() -> None:
    rng = np.random.default_rng(20261028)
    n_obs = 500
    cadence = 21
    phase = refit_phase_index(n_obs, cadence=cadence, first_is_refit=True)
    series = rng.normal(size=n_obs) + 0.8 * (phase == 0).astype(float)
    original = equal_refit_phase_means_wald(series, phase, maxlags=29)
    relabeled = (phase + 7) % cadence
    shifted_names = equal_refit_phase_means_wald(series, relabeled, maxlags=29)
    assert original.statistic is not None
    assert shifted_names.statistic == pytest.approx(original.statistic, rel=1e-10)


def test_time_roll_changes_wald_when_mean_shift_is_locked_to_k0() -> None:
    rng = np.random.default_rng(20261028)
    n_obs = 500
    phase = refit_phase_index(n_obs, cadence=21, first_is_refit=True)
    series = rng.normal(size=n_obs) + 1.5 * (phase == 0).astype(float)
    true = equal_refit_phase_means_wald(series, phase, maxlags=29)
    rolled = equal_refit_phase_means_wald(
        series, roll_refit_phase(phase, 11), maxlags=29
    )
    assert true.statistic is not None
    assert rolled.statistic is not None
    assert true.statistic != pytest.approx(rolled.statistic, rel=1e-8)
    np.testing.assert_array_equal(roll_refit_phase(phase, 0), phase)


def test_intercept_slope_wald_rejects_fresh_refit_and_linear_k() -> None:
    rng = np.random.default_rng(20261028)
    n_obs = 500
    phase = refit_phase_index(n_obs, cadence=21, first_is_refit=True)
    dummy = (phase == 0).astype(float)
    dummy_series = rng.normal(size=n_obs) + 1.2 * dummy
    dummy_wald = intercept_slope_wald(dummy_series, dummy, maxlags=5)
    assert dummy_wald.p_value is not None
    assert dummy_wald.p_value < 1e-6
    assert dummy_wald.df == 1

    linear_series = rng.normal(size=n_obs) + 0.08 * phase.astype(float)
    linear_wald = intercept_slope_wald(linear_series, phase.astype(float), maxlags=29)
    assert linear_wald.p_value is not None
    assert linear_wald.p_value < 1e-6
    assert linear_wald.slope == pytest.approx(0.08, abs=0.02)


def test_dropping_planted_outliers_changes_equal_mean_wald() -> None:
    rng = np.random.default_rng(20261028)
    n_obs = 500
    phase = refit_phase_index(n_obs, cadence=21, first_is_refit=True)
    series = rng.normal(size=n_obs, scale=0.2)
    planted = np.array([3, 40, 80, 120, 160, 200, 240, 280, 320, 360], dtype=int)
    series[planted] = 8.0
    full = equal_refit_phase_means_wald(series, phase, maxlags=29)
    drop = largest_absolute_indices(series, 10)
    np.testing.assert_array_equal(np.sort(drop), np.sort(planted))
    keep = np.ones(n_obs, dtype=bool)
    keep[drop] = False
    reduced = equal_refit_phase_means_wald(series[keep], phase[keep], maxlags=29)
    assert full.statistic is not None
    assert reduced.statistic is not None
    assert reduced.statistic != pytest.approx(full.statistic, rel=1e-8)
