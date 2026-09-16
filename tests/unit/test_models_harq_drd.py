"""Synthetic tests for HARQ-DRD realized-covariance forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covharness.models import (
    HARDRDRealizedCovariance,
    HARQDRDRealizedCovariance,
    InvalidModelForecastError,
    InvalidModelInputError,
)
from covharness.models.har_drd import (
    LAG_CONVENTION,
    MIN_WINDOW_LENGTH,
    PAIR_ORDERING,
    REPAIR_METHOD,
    RESPONSE_START,
    correlation_from_pair_vector,
    covariance_from_drd,
    dummy_fixed_effects_shared_slopes,
    fixed_effects_shared_slopes,
    har_predictors,
    har_regression_indices,
    unique_pair_indices,
)
from covharness.models.harq_drd import (
    HARQ_VARIANCE_SLOPES,
    QUARTICITY_DEFINITION,
    VARIANCE_SLOPE_ORDER,
    as_realized_quarticity_history,
    harq_forecast_predictors,
    harq_variance_design,
    harq_variance_predictors,
)
from covharness.protocol import (
    BlockName,
    ConfirmLockedError,
    build_temporal_protocol,
)

SEED = 20260916


def _spd(variances: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    return covariance_from_drd(variances, correlation)


def _constant_rho(n_assets: int, rho: float) -> np.ndarray:
    corr = np.full((n_assets, n_assets), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return corr


def _cube_from_variances(
    variances: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    return np.stack([_spd(row, correlation) for row in variances], axis=0)


def _simulate_har(
    n_times: int,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    start: float,
) -> np.ndarray:
    n_groups = int(intercepts.shape[0])
    panel = np.full((n_times, n_groups), float(start))
    for time_index in range(RESPONSE_START, n_times):
        predictors = har_predictors(panel, time_index)
        panel[time_index] = intercepts + predictors.T @ slopes
    return panel


def _simulate_harq(
    n_times: int,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    quarticity: np.ndarray,
    start: float,
) -> np.ndarray:
    n_assets = int(intercepts.shape[0])
    panel = np.full((n_times, n_assets), float(start))
    for time_index in range(RESPONSE_START, n_times):
        predictors = harq_variance_predictors(panel, quarticity, time_index)
        panel[time_index] = intercepts + predictors.T @ slopes
    return panel


def _varying_rq(n_times: int, n_assets: int) -> np.ndarray:
    time = np.arange(n_times, dtype=float)
    panel = np.empty((n_times, n_assets), dtype=float)
    for asset in range(n_assets):
        panel[:, asset] = 0.35 + 0.4 * asset + 0.25 * (
            1.0 + np.sin((time + 2.0 * asset) / 3.0)
        )
    return panel


def test_rq_must_have_shape_tn() -> None:
    cube = _cube_from_variances(np.ones((MIN_WINDOW_LENGTH, 2)), _constant_rho(2, 0.2))
    with pytest.raises(InvalidModelInputError, match="shape \\(T, N\\)"):
        HARQDRDRealizedCovariance().fit(cube, np.ones(MIN_WINDOW_LENGTH))


def test_mismatched_rq_t_fails() -> None:
    cube = _cube_from_variances(np.ones((MIN_WINDOW_LENGTH, 2)), _constant_rho(2, 0.2))
    with pytest.raises(InvalidModelInputError, match="T must match"):
        HARQDRDRealizedCovariance().fit(cube, np.ones((MIN_WINDOW_LENGTH - 1, 2)))


def test_mismatched_rq_n_fails() -> None:
    cube = _cube_from_variances(np.ones((MIN_WINDOW_LENGTH, 2)), _constant_rho(2, 0.2))
    with pytest.raises(InvalidModelInputError, match="N must match"):
        HARQDRDRealizedCovariance().fit(cube, np.ones((MIN_WINDOW_LENGTH, 3)))


def test_nonfinite_rq_fails() -> None:
    cube = _cube_from_variances(np.ones((MIN_WINDOW_LENGTH, 2)), _constant_rho(2, 0.2))
    rq = np.ones((MIN_WINDOW_LENGTH, 2))
    rq[4, 0] = np.nan
    with pytest.raises(InvalidModelInputError, match="finite"):
        HARQDRDRealizedCovariance().fit(cube, rq)


def test_negative_rq_fails() -> None:
    cube = _cube_from_variances(np.ones((MIN_WINDOW_LENGTH, 2)), _constant_rho(2, 0.2))
    rq = np.ones((MIN_WINDOW_LENGTH, 2))
    rq[3, 1] = -0.1
    with pytest.raises(InvalidModelInputError, match="nonnegative"):
        HARQDRDRealizedCovariance().fit(cube, rq)


def test_zero_rq_is_accepted() -> None:
    n_times = MIN_WINDOW_LENGTH
    cube = _cube_from_variances(np.full((n_times, 2), 1.2), _constant_rho(2, 0.2))
    rq = np.zeros((n_times, 2))
    model = HARQDRDRealizedCovariance().fit(cube, rq)
    assert model.fit_state.window_length == n_times
    copied = as_realized_quarticity_history(rq, n_times=n_times, n_assets=2)
    np.testing.assert_array_equal(copied, rq)


def test_harq_interaction_is_sqrt_rq_times_daily_variance() -> None:
    n_times = 30
    variances = np.column_stack(
        [np.arange(1, n_times + 1, dtype=float), 2.0 * np.arange(1, n_times + 1)]
    )
    quarticity = _varying_rq(n_times, 2)
    index = 22
    predictors = harq_variance_predictors(variances, quarticity, index)
    for asset in range(2):
        expected = np.sqrt(quarticity[index - 1, asset]) * variances[index - 1, asset]
        np.testing.assert_allclose(predictors[1, asset], expected)


def test_interaction_uses_per_asset_rq_not_aggregate() -> None:
    n_times = 30
    variances = np.full((n_times, 2), 1.5)
    quarticity = np.ones((n_times, 2))
    quarticity[:, 0] = 0.16
    quarticity[:, 1] = 4.00
    index = 22
    predictors = harq_variance_predictors(variances, quarticity, index)
    np.testing.assert_allclose(predictors[1, 0], np.sqrt(0.16) * 1.5)
    np.testing.assert_allclose(predictors[1, 1], np.sqrt(4.00) * 1.5)
    aggregate = np.sqrt(np.mean(quarticity[index - 1])) * 1.5
    assert not np.allclose(predictors[1], aggregate)


def test_t250_still_has_exactly_228_regression_dates() -> None:
    index = har_regression_indices(250)
    n_times, n_assets = 250, 2
    y, design = harq_variance_design(
        np.ones((n_times, n_assets)), np.ones((n_times, n_assets)), index
    )
    assert index.size == 228
    assert y.shape == (n_assets, 228)
    assert design.shape == (n_assets, 228, HARQ_VARIANCE_SLOPES)


def test_within_four_column_variance_matches_dummy_ols() -> None:
    rng = np.random.default_rng(SEED)
    n_times, n_assets = 40, 3
    variances = 0.4 + rng.random((n_times, n_assets))
    quarticity = 0.2 + rng.random((n_times, n_assets))
    y, x = harq_variance_design(variances, quarticity, har_regression_indices(n_times))
    alpha, beta, rank, n_columns = fixed_effects_shared_slopes(y, x)
    dummy_alpha, dummy_beta = dummy_fixed_effects_shared_slopes(y, x)
    assert n_columns == 4
    assert rank <= 4
    np.testing.assert_allclose(alpha, dummy_alpha, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(beta, dummy_beta, atol=1e-10, rtol=0.0)


def test_correlation_matches_har_drd_on_same_covariance_history() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 2), 1.3), _constant_rho(2, 0.25))
    rq = _varying_rq(n_times, 2)
    har = HARDRDRealizedCovariance().fit(cube)
    harq = HARQDRDRealizedCovariance().fit(cube, rq)
    np.testing.assert_allclose(harq.fit_state.alpha_R, har.fit_state.alpha_R)
    np.testing.assert_allclose(harq.fit_state.beta_R, har.fit_state.beta_R)
    assert harq.fit_state.pair_ordering == har.fit_state.pair_ordering == PAIR_ORDERING


def test_noiseless_harq_recovers_known_coefficients_with_varying_rq() -> None:
    n_times = 80
    alpha_q = np.array([0.05, 0.08])
    beta_q = np.array([0.35, 0.08, 0.20, 0.10])
    alpha_r = np.array([0.02])
    beta_r = np.array([0.35, 0.20, 0.10])
    quarticity = _varying_rq(n_times, 2)
    variances = _simulate_harq(n_times, alpha_q, beta_q, quarticity, start=1.0)
    pairs = _simulate_har(n_times, alpha_r, beta_r, start=0.15)
    assert np.all(variances > 0.0)
    assert np.all(np.abs(pairs) < 1.0)
    daily = variances[21]
    interaction = np.sqrt(quarticity[21]) * daily
    assert not np.allclose(interaction, daily)
    cube = np.stack(
        [
            _spd(variances[time], correlation_from_pair_vector(pairs[time], 2))
            for time in range(n_times)
        ],
        axis=0,
    )
    model = HARQDRDRealizedCovariance().fit(cube, quarticity)
    state = model.fit_state
    np.testing.assert_allclose(state.beta_Q, beta_q, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(state.alpha_Q, alpha_q, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(state.beta_R, beta_r, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(state.alpha_R, alpha_r, atol=1e-8, rtol=0.0)
    assert state.beta_Q[1] > 0.0
    assert state.quarticity_definition == QUARTICITY_DEFINITION
    assert state.beta_Q.shape == (4,)
    assert VARIANCE_SLOPE_ORDER[1] == "quarticity_interaction"
    forecast = model.forecast()
    assert forecast.identity.name == "harq_drd"
    assert forecast.identity.configuration["repaired"] is False
    assert forecast.identity.configuration["lag_convention"] == LAG_CONVENTION


def test_positive_phi_is_not_sign_constrained() -> None:
    n_times = 80
    alpha_q = np.array([0.04, 0.06])
    beta_q = np.array([0.30, 0.12, 0.18, 0.09])
    quarticity = _varying_rq(n_times, 2)
    variances = _simulate_harq(n_times, alpha_q, beta_q, quarticity, start=1.1)
    cube = _cube_from_variances(variances, _constant_rho(2, 0.2))
    model = HARQDRDRealizedCovariance().fit(cube, quarticity)
    np.testing.assert_allclose(model.fit_state.beta_Q[1], 0.12, atol=1e-8, rtol=0.0)
    assert model.fit_state.beta_Q[1] > 0.0


def test_forecast_uses_origin_rq_only() -> None:
    n_times = 80
    alpha_q = np.array([0.05, 0.08])
    beta_q = np.array([0.35, 0.08, 0.20, 0.10])
    quarticity = _varying_rq(n_times, 2)
    variances = _simulate_harq(n_times, alpha_q, beta_q, quarticity, start=1.0)
    cube = _cube_from_variances(variances, _constant_rho(2, 0.2))
    origin = harq_forecast_predictors(variances, quarticity)
    np.testing.assert_allclose(
        origin[1], np.sqrt(quarticity[n_times - 1]) * variances[n_times - 1]
    )
    original = HARQDRDRealizedCovariance().fit(cube, quarticity).forecast().matrix
    rq_origin_changed = quarticity.copy()
    rq_origin_changed[-1] = quarticity[-1] + 3.0
    changed = (
        HARQDRDRealizedCovariance().fit(cube, rq_origin_changed).forecast().matrix
    )
    assert not np.allclose(changed, original)
    rq_unused = quarticity.copy()
    rq_unused[0] = quarticity[0] + 50.0
    unused = HARQDRDRealizedCovariance().fit(cube, rq_unused).forecast().matrix
    np.testing.assert_allclose(unused, original)


def test_target_day_rq_does_not_affect_origin_forecast() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 2), 1.2), _constant_rho(2, 0.2))
    rq = _varying_rq(n_times, 2)
    first = HARQDRDRealizedCovariance().fit(cube, rq).forecast().matrix
    future = _spd(np.array([9.0, 8.0]), _constant_rho(2, 0.8))
    future_rq = np.array([1.0e6, 1.0e6])
    rq_extended = np.vstack([rq, future_rq])
    cube_extended = np.concatenate([cube, future[None]], axis=0)
    rq_extended[-1] += 50.0
    second = HARQDRDRealizedCovariance().fit(cube, rq).forecast().matrix
    np.testing.assert_array_equal(first, second)
    longer = HARQDRDRealizedCovariance().fit(cube_extended, rq_extended)
    assert longer.fit_state.window_length == n_times + 1
    assert not np.allclose(longer.forecast().matrix, first)


def test_window_does_not_read_before_start() -> None:
    n_times = 250
    variances = np.arange(n_times, dtype=float).reshape(n_times, 1)
    quarticity = np.ones((n_times, 1))
    first = har_regression_indices(n_times)[0]
    assert first - 22 == 0
    predictors = harq_variance_predictors(variances, quarticity, int(first))
    np.testing.assert_allclose(predictors[3], variances[0:17].mean())
    with pytest.raises(InvalidModelInputError):
        harq_variance_predictors(variances, quarticity, 21)


def test_asset_permutation_equivariance_joint_with_rq() -> None:
    n_times = 50
    rng = np.random.default_rng(SEED + 2)
    cube = np.empty((n_times, 3, 3))
    for time in range(n_times):
        factor = rng.normal(size=(3, 3))
        matrix = factor @ factor.T / 3.0 + np.eye(3)
        cube[time] = 0.5 * (matrix + matrix.T)
    rq = 0.3 + rng.random((n_times, 3))
    order = np.array([2, 0, 1])
    permute = np.eye(3)[order]
    permuted_cube = np.einsum("ij,tjk,lk->til", permute, cube, permute)
    permuted_rq = rq[:, order]
    original = HARQDRDRealizedCovariance().fit(cube, rq).forecast().matrix
    shuffled = HARQDRDRealizedCovariance().fit(permuted_cube, permuted_rq).forecast().matrix
    restored = permute.T @ shuffled @ permute
    np.testing.assert_allclose(restored, original, atol=1e-10, rtol=0.0)


def test_pair_ordering_unchanged_from_har_drd() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 3), 1.1), _constant_rho(3, 0.2))
    rq = _varying_rq(n_times, 3)
    model = HARQDRDRealizedCovariance().fit(cube, rq)
    model.forecast()
    assert model.fit_state.pair_ordering == PAIR_ORDERING
    rows, cols = unique_pair_indices(3)
    np.testing.assert_array_equal(rows, [0, 0, 1])
    np.testing.assert_array_equal(cols, [1, 2, 2])
    assert model._x_raw is not None
    rebuilt = correlation_from_pair_vector(model._x_raw, 3)
    np.testing.assert_allclose(rebuilt, model._R_raw)
    np.testing.assert_allclose(rebuilt[rows, cols], model._x_raw)


def test_zero_phi_dgp_matches_har_variance_equation() -> None:
    n_times = 60
    intercepts = np.array([0.05, 0.08])
    har_slopes = np.array([0.45, 0.25, 0.15])
    harq_slopes = np.array([0.45, 0.0, 0.25, 0.15])
    quarticity = _varying_rq(n_times, 2)
    from_har = _simulate_har(n_times, intercepts, har_slopes, start=1.0)
    from_harq = _simulate_harq(n_times, intercepts, harq_slopes, quarticity, start=1.0)
    np.testing.assert_allclose(from_harq, from_har, atol=1e-12, rtol=0.0)


def test_valid_raw_forecast_is_not_repaired() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 2), 1.3), _constant_rho(2, 0.2))
    rq = np.ones((n_times, 2))
    model = HARQDRDRealizedCovariance().fit(cube, rq)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert validity.repaired is False
    assert validity.repair_method is None
    assert forecast.diagnostics.positive_definite is True


def test_raw_nonpositive_variance_triggers_window_mean_repair() -> None:
    n_times = 40
    variances = np.ones((n_times, 2))
    variances[:, 0] = np.where(np.arange(n_times) % 2 == 0, 2.0, 6.0)
    variances[-1, 0] = 20.0
    cube = _cube_from_variances(variances, _constant_rho(2, 0.2))
    rq = np.ones((n_times, 2))
    rq[:, 0] = np.where(np.arange(n_times) % 2 == 0, 0.4, 1.6)
    original = cube.copy()
    model = HARQDRDRealizedCovariance().fit(cube, rq)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert validity.repaired is True
    assert validity.repair_method == REPAIR_METHOD
    assert validity.raw_nonpositive_variance is True
    np.testing.assert_allclose(forecast.matrix, original.mean(axis=0))
    np.testing.assert_array_equal(cube, original)


def test_raw_correlation_out_of_bounds_triggers_same_fallback() -> None:
    n_times = 30
    pairs = _simulate_har(n_times, np.array([0.02]), np.array([1.25, 0.0, 0.0]), 0.08)
    variances = np.ones((n_times, 2))
    cube = np.stack(
        [
            _spd(variances[time], correlation_from_pair_vector(pairs[time], 2))
            for time in range(n_times)
        ],
        axis=0,
    )
    rq = _varying_rq(n_times, 2)
    model = HARQDRDRealizedCovariance().fit(cube, rq)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert model._x_raw is not None
    assert np.any(np.abs(model._x_raw) > 1.0)
    assert validity.repaired is True
    assert validity.raw_correlation_out_of_bounds is True
    assert validity.repair_method == REPAIR_METHOD
    np.testing.assert_allclose(forecast.matrix, cube.mean(axis=0))


def test_assembled_correlation_not_pd_triggers_fallback() -> None:
    n_times = 26
    positive = _simulate_har(
        n_times, np.array([0.01, 0.01]), np.array([1.3, 0.0, 0.0]), 0.15
    )
    negative = _simulate_har(
        n_times, np.array([-0.02]), np.array([1.3, 0.0, 0.0]), -0.08
    )
    pairs = np.column_stack([positive[:, 0], positive[:, 1], negative[:, 0]])
    variances = np.ones((n_times, 3))
    cube = np.stack(
        [
            _spd(variances[time], correlation_from_pair_vector(pairs[time], 3))
            for time in range(n_times)
        ],
        axis=0,
    )
    rq = _varying_rq(n_times, 3)
    model = HARQDRDRealizedCovariance().fit(cube, rq)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert model._x_raw is not None
    assert np.all(np.abs(model._x_raw) <= 1.0)
    assert validity.raw_correlation_out_of_bounds is False
    assert validity.repaired is True
    assert validity.raw_correlation_not_pd is True
    assert validity.repair_method == REPAIR_METHOD
    np.testing.assert_allclose(forecast.matrix, cube.mean(axis=0))


def test_non_pd_fallback_raises_without_second_repair() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    history = np.stack([singular] * MIN_WINDOW_LENGTH, axis=0)
    rq = np.ones((MIN_WINDOW_LENGTH, 2))
    model = HARQDRDRealizedCovariance().fit(history, rq)
    with pytest.raises(InvalidModelForecastError, match="No second repair"):
        model.forecast()


def test_outputs_do_not_alias_inputs() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 2), 1.4), _constant_rho(2, 0.1))
    rq = _varying_rq(n_times, 2)
    original_cube = cube.copy()
    original_rq = rq.copy()
    model = HARQDRDRealizedCovariance().fit(cube, rq)
    np.testing.assert_array_equal(cube, original_cube)
    np.testing.assert_array_equal(rq, original_rq)
    forecast = model.forecast()
    assert not np.shares_memory(forecast.matrix, cube)
    forecast.matrix[0, 0] = -99.0
    cube[0, 0, 0] += 1.0
    rq[0, 0] += 1.0
    second = model.forecast().matrix
    assert second[0, 0] != -99.0
    third = HARQDRDRealizedCovariance().fit(original_cube, original_rq).forecast().matrix
    np.testing.assert_array_equal(second, third)


def test_deterministic_repeatability() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 2), 1.1), _constant_rho(2, 0.3))
    rq = _varying_rq(n_times, 2)
    first = HARQDRDRealizedCovariance().fit(cube, rq).forecast().matrix
    second = HARQDRDRealizedCovariance().fit(cube, rq).forecast().matrix
    np.testing.assert_array_equal(first, second)


def test_moderate_n_keeps_four_and_three_within_columns() -> None:
    rng = np.random.default_rng(SEED + 3)
    n_times, n_assets = 40, 50
    cube = np.empty((n_times, n_assets, n_assets))
    for time in range(n_times):
        factor = rng.normal(size=(n_assets, 3))
        matrix = factor @ factor.T / 3.0 + np.eye(n_assets)
        cube[time] = 0.5 * (matrix + matrix.T)
    rq = 0.2 + rng.random((n_times, n_assets))
    model = HARQDRDRealizedCovariance().fit(cube, rq)
    state = model.fit_state
    n_pairs = n_assets * (n_assets - 1) // 2
    assert state.n_pairs == n_pairs
    assert state.beta_Q.shape == (4,)
    assert state.beta_R.shape == (3,)
    assert state.alpha_Q.shape == (n_assets,)
    assert state.alpha_R.shape == (n_pairs,)
    assert state.variance_within_rank <= 4
    assert state.correlation_within_rank <= 3
    forecast = model.forecast()
    assert forecast.matrix.shape == (n_assets, n_assets)


def test_synthetic_protocol_window_confirm_locked() -> None:
    calendar = pd.bdate_range("1990-01-01", periods=1500)
    protocol = build_temporal_protocol(calendar)
    target = protocol.validation_targets()[0]
    step = protocol.forecast_step(protocol.origin_for_target(target))
    cube = np.stack(
        [(1.0 + 0.01 * float(i)) * np.eye(2) + 0.1 * np.ones((2, 2)) for i in range(len(calendar))],
        axis=0,
    )
    for time in range(cube.shape[0]):
        np.fill_diagonal(cube[time], np.diag(cube[time]) + 1.0)
    rq = _varying_rq(len(calendar), 2)
    window = cube[step.window_start_index : step.window_end_index]
    rq_window = rq[step.window_start_index : step.window_end_index]
    assert window.shape[0] == protocol.m
    cube[step.target_index] = np.array([[50.0, 0.2], [0.2, 40.0]])
    rq[step.target_index] = np.array([80.0, 90.0])
    forecast = HARQDRDRealizedCovariance().fit(window, rq_window).forecast()
    assert forecast.matrix.shape == (2, 2)
    assert not np.allclose(forecast.matrix, cube[step.target_index])
    with pytest.raises(ConfirmLockedError):
        protocol.confirm_targets()
    with pytest.raises(ConfirmLockedError):
        protocol.forecast_schedule(BlockName.CONFIRM)
    assert protocol.confirm_locked is True
