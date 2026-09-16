"""Unit tests for the recentered stationary-bootstrap pairwise mean test."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from covharness.inference import (
    DegenerateLossDifferentialError,
    diebold_mariano,
    pairwise_block_length,
    stationary_bootstrap_mean_test,
)
from covharness.inference import bootstrap_mean as mean_module
from covharness.inference.bootstrap import default_block_length, stationary_bootstrap_indices
from covharness.inference.bootstrap_mean import (
    PAIRWISE_BOOTSTRAP_SEED,
    PAIRWISE_N_BOOT,
    bootstrap_mean_statistics,
    finite_b_pvalue,
    null_centered_differential,
)
from covharness.inference.differentials import ALTERNATIVE_A_BETTER, ALTERNATIVE_B_BETTER


def test_null_centering_is_exact() -> None:
    series = np.array([1.0, -0.5, 2.0, 0.25], dtype=float)
    centered = null_centered_differential(series)
    np.testing.assert_allclose(centered, series - series.mean())
    assert float(centered.mean()) == pytest.approx(0.0, abs=1e-15)


def test_reuses_existing_stationary_bootstrap_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    source = inspect.getsource(mean_module.stationary_bootstrap_mean_test)
    assert "stationary_bootstrap_indices(" in source
    assert "def stationary_bootstrap_indices" not in source
    calls: list[tuple[int, int, int, int]] = []
    original = mean_module.stationary_bootstrap_indices

    def wrapped(n_observations, n_boot=5000, *, block_length=None, seed=0):
        calls.append((n_observations, n_boot, int(block_length), int(seed)))
        return original(n_observations, n_boot, block_length=block_length, seed=seed)

    monkeypatch.setattr(mean_module, "stationary_bootstrap_indices", wrapped)
    series = np.linspace(-0.4, 0.6, 12)
    stationary_bootstrap_mean_test(series, n_boot=7, seed=11)
    assert calls == [(12, 7, pairwise_block_length(12), 11)]


def test_frozen_block_length_rule() -> None:
    for n_obs in (250, 500, 1000, 40, 8):
        expected = max(2, int(np.floor(np.cbrt(n_obs))))
        assert pairwise_block_length(n_obs) == expected
        assert pairwise_block_length(n_obs) == default_block_length(n_obs)
    assert pairwise_block_length(250) == 6
    assert pairwise_block_length(500) == 7
    assert pairwise_block_length(1000) == 10


def test_observed_statistic_is_sqrt_t_times_mean() -> None:
    series = np.array([0.2, -0.1, 0.4, -0.3, 0.5], dtype=float)
    result = stationary_bootstrap_mean_test(series, n_boot=5, seed=3)
    assert result.statistic == pytest.approx(np.sqrt(series.size) * float(series.mean()))
    assert result.mean_differential == pytest.approx(float(series.mean()))
    assert result.method == "stationary_bootstrap_mean"


def test_bootstrap_statistic_is_sqrt_t_times_recentered_mean() -> None:
    series = np.array([1.0, 0.0, -0.5, 0.25, 0.75, -0.2], dtype=float)
    centered = null_centered_differential(series)
    indices = np.array([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]], dtype=int)
    stars = bootstrap_mean_statistics(centered, indices)
    expected = np.sqrt(series.size) * centered[indices].mean(axis=1)
    np.testing.assert_allclose(stars, expected)


def test_two_sided_pvalue_matches_hand_count() -> None:
    observed = 1.2
    stars = np.array([0.5, -1.4, 1.2, 2.0, -0.1], dtype=float)
    count = int(np.sum(np.abs(stars) >= abs(observed)))
    expected = (1 + count) / (stars.size + 1)
    assert finite_b_pvalue(observed, stars, "two_sided", stars.size) == pytest.approx(expected)
    assert count == 3


def test_one_sided_tails_match_hand_counts() -> None:
    observed = -0.4
    stars = np.array([-0.8, -0.4, 0.1, 0.7, -0.2], dtype=float)
    a_count = int(np.sum(stars <= observed))
    b_count = int(np.sum(stars >= observed))
    assert finite_b_pvalue(observed, stars, ALTERNATIVE_A_BETTER, stars.size) == pytest.approx(
        (1 + a_count) / (stars.size + 1)
    )
    assert finite_b_pvalue(observed, stars, ALTERNATIVE_B_BETTER, stars.size) == pytest.approx(
        (1 + b_count) / (stars.size + 1)
    )
    assert a_count == 2
    assert b_count == 4


def test_weak_inequalities_count_ties() -> None:
    observed = 1.0
    stars = np.array([1.0, -1.0, 0.5], dtype=float)
    two = finite_b_pvalue(observed, stars, "two_sided", stars.size)
    assert two == pytest.approx((1 + 2) / 4)
    source = inspect.getsource(finite_b_pvalue)
    assert ">=" in source
    assert "<=" in source
    assert "T* > T" not in source


def test_finite_b_correction_and_open_unit_interval() -> None:
    stars = np.array([0.1, 0.2, 0.3], dtype=float)
    p_value = finite_b_pvalue(10.0, stars, "two_sided", 3)
    assert p_value == pytest.approx(1.0 / 4.0)
    series = np.linspace(-1.0, 1.5, 20)
    result = stationary_bootstrap_mean_test(series, n_boot=9, seed=5)
    assert 0.0 < result.p_value <= 1.0
    assert any(result.p_value == pytest.approx(k / 10.0) for k in range(1, 11))


def test_same_seed_is_deterministic() -> None:
    series = np.linspace(-0.3, 0.8, 16)
    first = stationary_bootstrap_mean_test(series, n_boot=21, seed=17)
    second = stationary_bootstrap_mean_test(series, n_boot=21, seed=17)
    assert first.p_value == second.p_value
    assert first.statistic == second.statistic
    assert first.block_length == second.block_length


def test_different_seed_changes_resamples() -> None:
    series = np.linspace(-0.3, 0.8, 16)
    first = stationary_bootstrap_indices(16, 21, block_length=pairwise_block_length(16), seed=17)
    second = stationary_bootstrap_indices(16, 21, block_length=pairwise_block_length(16), seed=18)
    assert not np.array_equal(first, second)
    p_one = stationary_bootstrap_mean_test(series, n_boot=51, seed=17).p_value
    p_two = stationary_bootstrap_mean_test(series, n_boot=51, seed=18).p_value
    assert p_one != p_two


def test_sign_reversal_swaps_one_sided_tails() -> None:
    series = np.linspace(-0.8, 0.1, 18)
    positive = -series
    two = stationary_bootstrap_mean_test(series, alternative="two_sided", n_boot=31, seed=19)
    two_flip = stationary_bootstrap_mean_test(positive, alternative="two_sided", n_boot=31, seed=19)
    a_orig = stationary_bootstrap_mean_test(series, alternative="a_better", n_boot=31, seed=19)
    b_flip = stationary_bootstrap_mean_test(positive, alternative="b_better", n_boot=31, seed=19)
    b_orig = stationary_bootstrap_mean_test(series, alternative="b_better", n_boot=31, seed=19)
    a_flip = stationary_bootstrap_mean_test(positive, alternative="a_better", n_boot=31, seed=19)
    assert two.p_value == pytest.approx(two_flip.p_value)
    assert a_orig.p_value == pytest.approx(b_flip.p_value)
    assert b_orig.p_value == pytest.approx(a_flip.p_value)


def test_constant_series_are_degenerate() -> None:
    with pytest.raises(DegenerateLossDifferentialError, match="equal"):
        stationary_bootstrap_mean_test(np.zeros(12))
    with pytest.raises(DegenerateLossDifferentialError, match="equal"):
        stationary_bootstrap_mean_test(np.full(12, 0.4))


def test_input_is_not_mutated() -> None:
    series = np.array([0.2, -0.1, 0.3, 0.0, -0.4], dtype=float)
    original = series.copy()
    stationary_bootstrap_mean_test(series, n_boot=6, seed=7)
    np.testing.assert_array_equal(series, original)


def test_existing_diebold_mariano_result_is_unchanged() -> None:
    series = np.linspace(-0.25, 0.4, 30)
    before = diebold_mariano(series, alternative="two_sided")
    stationary_bootstrap_mean_test(series, n_boot=11, seed=13)
    after = diebold_mariano(series, alternative="two_sided")
    assert after.statistic == before.statistic
    assert after.p_value == before.p_value
    assert after.hac_maxlags == before.hac_maxlags
    assert after.hln_small_sample_correction is False


def test_companion_does_not_use_hac_or_lrv() -> None:
    source = inspect.getsource(mean_module)
    assert "hac_long_run_variance" not in source
    assert "diebold_mariano" not in source
    assert "from covharness.inference.hac" not in source
    assert "from covharness.inference.dm" not in source
    assert PAIRWISE_N_BOOT == 5000
    assert PAIRWISE_BOOTSTRAP_SEED == 20260917
    assert PAIRWISE_BOOTSTRAP_SEED != 20260913
