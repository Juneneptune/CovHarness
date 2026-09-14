"""SPA, MCS, and stationary-bootstrap tests on synthetic loss matrices."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.inference import (
    BOOTSTRAP_SEED,
    DegenerateLossDifferentialError,
    DegenerateMCSDifferentialError,
    default_block_length,
    hansen_lil_threshold,
    model_confidence_set,
    stationary_bootstrap_indices,
    superior_predictive_ability,
)
from covharness.inference.bootstrap import restart_probability
from covharness.inference.mcs import PROCEDURE_MAX, PROCEDURE_RANGE
from covharness.inference.spa import stationary_bootstrap_long_run_variance

TEST_N_BOOT = 400
TEST_SEED = 20260913


def _iid_equal_losses(
    rng: np.random.Generator, n_obs: int, n_models: int
) -> np.ndarray:
    return rng.normal(loc=1.0, scale=0.25, size=(n_obs, n_models))


def test_default_block_length_is_cube_root_of_t() -> None:
    assert default_block_length(8) == 2
    assert default_block_length(27) == 3
    assert default_block_length(1000) == 10
    assert default_block_length(500) == int(np.floor(np.cbrt(500)))
    assert default_block_length(500) != int(np.floor(np.sqrt(500)))


def test_stationary_bootstrap_indices_shape_and_range() -> None:
    n_obs = 40
    n_boot = 25
    indices = stationary_bootstrap_indices(n_obs, n_boot, seed=TEST_SEED)
    assert indices.shape == (n_boot, n_obs)
    assert indices.min() >= 0
    assert indices.max() < n_obs
    assert indices.dtype == np.int64


def test_stationary_bootstrap_wraparound_and_restarts() -> None:
    n_obs = 30
    indices = stationary_bootstrap_indices(n_obs, 80, seed=TEST_SEED)
    continued = (indices[:, :-1] + 1) % n_obs
    is_continuation = indices[:, 1:] == continued
    # A non-continuation must still be a valid origin, which the range test covers.
    assert is_continuation.any()
    assert (~is_continuation).any()
    wrapped = (indices[:, :-1] == n_obs - 1) & is_continuation
    assert wrapped.any()
    assert np.all(indices[:, 1:][wrapped] == 0)


def test_stationary_bootstrap_restart_probability() -> None:
    n_obs = 200
    n_boot = 2000
    length = default_block_length(n_obs)
    q = restart_probability(n_obs, length)
    indices = stationary_bootstrap_indices(n_obs, n_boot, seed=TEST_SEED)
    continued = (indices[:, :-1] + 1) % n_obs
    restart_rate = float(np.mean(indices[:, 1:] != continued))
    assert restart_rate == pytest.approx(q, abs=0.02)


def test_stationary_bootstrap_seed_reproducibility() -> None:
    first = stationary_bootstrap_indices(50, 30, seed=TEST_SEED)
    second = stationary_bootstrap_indices(50, 30, seed=TEST_SEED)
    other = stationary_bootstrap_indices(50, 30, seed=TEST_SEED + 1)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_stationary_bootstrap_is_joint_across_columns() -> None:
    n_obs = 20
    indices = stationary_bootstrap_indices(n_obs, 5, seed=7)
    losses = np.arange(n_obs * 3, dtype=float).reshape(n_obs, 3)
    resampled = losses[indices]
    # The same time index is applied to every model column.
    np.testing.assert_array_equal(resampled[..., 0], losses[:, 0][indices])
    np.testing.assert_array_equal(resampled[..., 1], losses[:, 1][indices])
    np.testing.assert_array_equal(resampled[..., 2], losses[:, 2][indices])


def test_hansen_lil_threshold_formula() -> None:
    n_obs = 100
    expected = -np.sqrt(2.0 * np.log(np.log(n_obs)))
    assert hansen_lil_threshold(n_obs) == pytest.approx(expected)
    wrong_factor = -np.sqrt(2.0) * np.log(np.log(n_obs))
    wrong_double = -2.0 * np.log(np.log(n_obs))
    assert expected != pytest.approx(wrong_factor)
    assert expected != pytest.approx(wrong_double)


def test_spa_sign_convention_benchmark_minus_alternative() -> None:
    losses = np.column_stack(
        [
            np.full(40, 2.0) + 0.05 * np.arange(40),
            np.full(40, 1.0) + 0.05 * np.arange(40),
        ]
    )
    result = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    # Alternative column 1 has lower loss, so d = L0 - L1 is positive.
    assert result.mean_differentials[0] > 0.0
    assert result.statistic > 0.0


def test_spa_statistic_zero_when_all_sample_means_nonpositive() -> None:
    losses = np.column_stack(
        [
            np.linspace(0.0, 1.0, 50),
            np.linspace(0.0, 1.0, 50) + 0.4,
            np.linspace(0.0, 1.0, 50) + 0.8,
        ]
    )
    result = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert np.all(result.mean_differentials <= 0.0)
    assert result.statistic == 0.0


def test_spa_p_value_ordering() -> None:
    rng = np.random.default_rng(3)
    losses = _iid_equal_losses(rng, 80, 5)
    losses[:, 1] -= 0.08
    result = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert result.p_value_lower <= result.p_value_consistent + 1e-15
    assert result.p_value_consistent <= result.p_value_upper + 1e-15


def test_spa_equal_accuracy_null_does_not_reject() -> None:
    rng = np.random.default_rng(11)
    losses = _iid_equal_losses(rng, 120, 4)
    result = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert result.p_value_consistent > result.alpha


def test_spa_rejects_clearly_superior_alternative() -> None:
    rng = np.random.default_rng(12)
    losses = _iid_equal_losses(rng, 120, 4)
    losses[:, 2] -= 0.8
    result = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert result.p_value_consistent < result.alpha
    assert int(np.argmax(result.studentized_statistics)) == 1


def test_spa_poor_alternatives_do_not_inflate_consistent_pvalue() -> None:
    rng = np.random.default_rng(13)
    n_obs = 120
    benchmark = rng.normal(loc=1.0, scale=0.2, size=n_obs)
    rival = benchmark - 0.45 + 0.05 * rng.normal(size=n_obs)
    poor = np.column_stack(
        [benchmark + 1.5 + 0.2 * rng.normal(size=n_obs) for _ in range(8)]
    )
    compact = np.column_stack([benchmark, rival])
    crowded = np.column_stack([benchmark, rival, poor])
    compact_spa = superior_predictive_ability(
        compact, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    crowded_spa = superior_predictive_ability(
        crowded, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert compact_spa.p_value_consistent < compact_spa.alpha
    assert crowded_spa.p_value_consistent < crowded_spa.alpha
    assert crowded_spa.p_value_upper >= crowded_spa.p_value_consistent
    # The consistent p-value must not explode toward 1 merely from poor models.
    assert crowded_spa.p_value_consistent < 0.20


def test_spa_column_permutation_invariance() -> None:
    rng = np.random.default_rng(14)
    losses = _iid_equal_losses(rng, 90, 4)
    losses[:, 1] -= 0.3
    base = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    permuted = losses[:, [0, 3, 1, 2]]
    shuffled = superior_predictive_ability(
        permuted, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert base.statistic == pytest.approx(shuffled.statistic)
    assert base.p_value_consistent == pytest.approx(shuffled.p_value_consistent)


def test_spa_seed_reproducibility() -> None:
    rng = np.random.default_rng(15)
    losses = _iid_equal_losses(rng, 70, 3)
    first = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    second = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    third = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED + 7
    )
    assert first.statistic == second.statistic
    assert first.p_value_consistent == second.p_value_consistent
    assert first.p_value_consistent != third.p_value_consistent or first.statistic == 0.0


def test_spa_serially_dependent_differentials() -> None:
    rng = np.random.default_rng(16)
    n_obs = 150
    noise = rng.normal(size=n_obs)
    ar = np.empty(n_obs)
    ar[0] = noise[0]
    for t in range(1, n_obs):
        ar[t] = 0.6 * ar[t - 1] + noise[t]
    losses = np.column_stack([ar, ar + 0.15 * rng.normal(size=n_obs)])
    result = superior_predictive_ability(
        losses, benchmark_index=0, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert result.block_length == default_block_length(n_obs)
    assert 0.0 <= result.p_value_consistent <= 1.0
    q = 1.0 / result.block_length
    assert n_obs * q**2 > 1.0


def test_spa_rejects_exact_constant_differential() -> None:
    losses = np.column_stack([np.ones(40), np.ones(40)])
    with pytest.raises(DegenerateLossDifferentialError, match="equal"):
        superior_predictive_ability(losses, benchmark_index=0, n_boot=20, seed=TEST_SEED)
    shifted = np.column_stack([np.full(40, 1.5), np.full(40, 0.5)])
    with pytest.raises(DegenerateLossDifferentialError, match="equal"):
        superior_predictive_ability(shifted, benchmark_index=0, n_boot=20, seed=TEST_SEED)


def test_spa_does_not_mutate_inputs() -> None:
    losses = np.array([[1.0, 0.8], [1.1, 0.9], [0.9, 0.7], [1.2, 1.0]])
    original = losses.copy()
    superior_predictive_ability(losses, benchmark_index=0, n_boot=30, seed=TEST_SEED)
    np.testing.assert_array_equal(losses, original)


def test_spa_long_run_variance_is_not_bartlett_hac() -> None:
    rng = np.random.default_rng(17)
    series = rng.normal(size=(80, 1))
    series[1:] += 0.4 * series[:-1]
    q = restart_probability(80)
    omega = stationary_bootstrap_long_run_variance(series, q)
    from covharness.inference.hac import hac_long_run_variance

    bartlett = hac_long_run_variance(series[:, 0]).long_run_variance
    assert omega[0] != pytest.approx(bartlett)


def test_mcs_all_models_equivalent() -> None:
    rng = np.random.default_rng(21)
    losses = _iid_equal_losses(rng, 100, 4)
    for procedure in (PROCEDURE_RANGE, PROCEDURE_MAX):
        result = model_confidence_set(
            losses, procedure=procedure, n_boot=TEST_N_BOOT, seed=TEST_SEED
        )
        assert result.mcs_pvalues[result.elimination_order[-1]] == 1.0
        assert np.all(result.in_mcs)
        assert np.all(np.diff(result.mcs_pvalues[result.elimination_order]) >= -1e-15)


def test_mcs_removes_clearly_inferior_model() -> None:
    rng = np.random.default_rng(22)
    losses = _iid_equal_losses(rng, 120, 4)
    losses[:, 3] += 1.2
    for procedure in (PROCEDURE_RANGE, PROCEDURE_MAX):
        result = model_confidence_set(
            losses, procedure=procedure, n_boot=TEST_N_BOOT, seed=TEST_SEED
        )
        assert result.elimination_order[0] == 3
        assert result.mcs_pvalues[3] < result.alpha
        assert not result.in_mcs[3]
        assert np.all(result.in_mcs[[0, 1, 2]])


def test_mcs_retains_clearly_superior_model() -> None:
    rng = np.random.default_rng(23)
    losses = _iid_equal_losses(rng, 120, 4)
    losses[:, 1] -= 1.0
    for procedure in (PROCEDURE_RANGE, PROCEDURE_MAX):
        result = model_confidence_set(
            losses, procedure=procedure, n_boot=TEST_N_BOOT, seed=TEST_SEED
        )
        assert result.elimination_order[-1] == 1
        assert result.mcs_pvalues[1] == 1.0
        assert result.in_mcs[1]


def test_mcs_multiple_equivalent_best_models() -> None:
    rng = np.random.default_rng(24)
    n_obs = 130
    shared = rng.normal(loc=0.5, scale=0.2, size=n_obs)
    best_a = shared + 0.05 * rng.normal(size=n_obs)
    best_b = shared + 0.05 * rng.normal(size=n_obs)
    worse = shared + 1.0 + 0.05 * rng.normal(size=n_obs)
    losses = np.column_stack([best_a, worse, best_b])
    for procedure in (PROCEDURE_RANGE, PROCEDURE_MAX):
        result = model_confidence_set(
            losses, procedure=procedure, n_boot=TEST_N_BOOT, seed=TEST_SEED
        )
        assert result.elimination_order[0] == 1
        assert result.in_mcs[0] and result.in_mcs[2]
        assert not result.in_mcs[1]


def test_mcs_permutation_invariance_by_identity() -> None:
    rng = np.random.default_rng(25)
    losses = _iid_equal_losses(rng, 100, 3)
    losses[:, 2] += 1.1
    base = model_confidence_set(
        losses, procedure=PROCEDURE_RANGE, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    permuted = losses[:, [2, 0, 1]]
    shuffled = model_confidence_set(
        permuted, procedure=PROCEDURE_RANGE, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    # Column 2 in the original matrix is column 0 after permutation.
    assert base.in_mcs[2] == shuffled.in_mcs[0]
    assert base.in_mcs[0] == shuffled.in_mcs[1]
    assert base.in_mcs[1] == shuffled.in_mcs[2]


def test_mcs_seed_reproducibility() -> None:
    rng = np.random.default_rng(26)
    losses = _iid_equal_losses(rng, 80, 3)
    first = model_confidence_set(
        losses, procedure=PROCEDURE_RANGE, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    second = model_confidence_set(
        losses, procedure=PROCEDURE_RANGE, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    np.testing.assert_array_equal(first.elimination_order, second.elimination_order)
    np.testing.assert_allclose(first.mcs_pvalues, second.mcs_pvalues)


def test_mcs_monotone_pvalues_and_membership() -> None:
    rng = np.random.default_rng(27)
    losses = _iid_equal_losses(rng, 90, 5)
    losses[:, 4] += 0.9
    result = model_confidence_set(
        losses, procedure=PROCEDURE_RANGE, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    ordered = result.mcs_pvalues[result.elimination_order]
    assert np.all(np.diff(ordered) >= -1e-15)
    assert result.mcs_pvalues[result.elimination_order[-1]] == 1.0
    np.testing.assert_array_equal(result.in_mcs, result.mcs_pvalues >= result.alpha)
    at_05 = result.mcs_pvalues >= 0.05
    at_25 = result.mcs_pvalues >= 0.25
    assert np.all(at_25 <= at_05)


def test_mcs_identical_columns_are_ties() -> None:
    rng = np.random.default_rng(28)
    base = rng.normal(size=(60, 1))
    losses = np.column_stack([base[:, 0], base[:, 0], base[:, 0] + 0.02 * rng.normal(size=60)])
    result = model_confidence_set(
        losses, procedure=PROCEDURE_RANGE, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert result.n_models == 3
    assert np.isfinite(result.mcs_pvalues).all()


def test_mcs_nonzero_constant_differential_is_rejected() -> None:
    base = np.arange(40, dtype=float)
    losses = np.column_stack([base, base + 1.0])
    with pytest.raises(DegenerateMCSDifferentialError, match="nonzero constant"):
        model_confidence_set(losses, n_boot=20, seed=TEST_SEED)


def test_mcs_range_and_max_are_separate_procedures() -> None:
    rng = np.random.default_rng(29)
    losses = _iid_equal_losses(rng, 80, 4)
    losses[:, 3] += 0.7
    range_result = model_confidence_set(
        losses, procedure=PROCEDURE_RANGE, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    max_result = model_confidence_set(
        losses, procedure=PROCEDURE_MAX, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert range_result.procedure == PROCEDURE_RANGE
    assert max_result.procedure == PROCEDURE_MAX
    assert range_result.statistics_along_path[0] != pytest.approx(
        max_result.statistics_along_path[0]
    )


def test_mcs_reuses_one_bootstrap_index_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(30)
    losses = _iid_equal_losses(rng, 70, 4)
    calls = {"n": 0}
    real = stationary_bootstrap_indices

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "covharness.inference.mcs.stationary_bootstrap_indices", wrapped
    )
    model_confidence_set(
        losses, procedure=PROCEDURE_RANGE, n_boot=TEST_N_BOOT, seed=TEST_SEED
    )
    assert calls["n"] == 1


def test_mcs_does_not_mutate_inputs() -> None:
    losses = np.array(
        [[1.0, 1.1, 0.9], [0.8, 0.7, 1.2], [1.3, 1.0, 0.8], [0.9, 1.4, 1.1]],
        dtype=float,
    )
    original = losses.copy()
    model_confidence_set(losses, n_boot=40, seed=TEST_SEED)
    np.testing.assert_array_equal(losses, original)


def test_spa_and_mcs_production_defaults_remain_locked() -> None:
    from covharness.inference.bootstrap import DEFAULT_N_BOOT
    from covharness.inference.mcs import MCS_ALPHA
    from covharness.inference.spa import SPA_ALPHA

    assert DEFAULT_N_BOOT == 5000
    assert BOOTSTRAP_SEED == 20260913
    assert SPA_ALPHA == 0.05
    assert MCS_ALPHA == 0.10
    assert default_block_length(1000) == 10
