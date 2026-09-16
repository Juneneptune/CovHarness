"""Synthetic tests for HAR-DRD realized-covariance forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covharness.models import (
    HARDRDRealizedCovariance,
    InvalidModelForecastError,
    InvalidModelInputError,
)
from covharness.models.har_drd import (
    LAG_CONVENTION,
    MIN_WINDOW_LENGTH,
    MONTHLY_WIDTH,
    PAIR_ORDERING,
    REPAIR_METHOD,
    RESPONSE_START,
    WEEKLY_WIDTH,
    correlation_from_pair_vector,
    covariance_from_drd,
    drd_components,
    dummy_fixed_effects_shared_slopes,
    forecast_predictors,
    har_predictors,
    har_regression_indices,
    har_response_design,
    pair_vector_from_correlation,
    unique_pair_indices,
    fixed_effects_shared_slopes,
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


def test_drd_roundtrip_hand_spd() -> None:
    matrix = np.array([[4.0, 1.2, -0.4], [1.2, 9.0, 0.6], [-0.4, 0.6, 1.0]])
    variances, correlation = drd_components(matrix)
    rebuilt = covariance_from_drd(variances, correlation)
    np.testing.assert_allclose(rebuilt, matrix)
    np.testing.assert_allclose(np.diag(correlation), 1.0)


def test_drd_normalization_uses_sqrt_variance_scale() -> None:
    # Hand-built SPD with unequal variances. diag(S) is not sqrt(diag(S)).
    matrix = np.array(
        [[16.0, 6.0, -2.0], [6.0, 25.0, 4.0], [-2.0, 4.0, 4.0]],
        dtype=float,
    )
    variances = np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2]], dtype=float)
    scales = np.sqrt(variances)
    assert not np.allclose(variances, scales)
    np.testing.assert_array_equal(variances, [16.0, 25.0, 4.0])
    np.testing.assert_array_equal(scales, [4.0, 5.0, 2.0])

    # Independent correlation formula. Do not call production DRD here.
    expected_corr = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            expected_corr[i, j] = matrix[i, j] / (scales[i] * scales[j])
    np.testing.assert_allclose(np.diag(expected_corr), 1.0)
    np.testing.assert_allclose(expected_corr[0, 1], 6.0 / (4.0 * 5.0))
    np.testing.assert_allclose(expected_corr[0, 2], -2.0 / (4.0 * 2.0))
    np.testing.assert_allclose(expected_corr[1, 2], 4.0 / (5.0 * 2.0))

    rebuilt = np.diag(scales) @ expected_corr @ np.diag(scales)
    np.testing.assert_allclose(rebuilt, matrix, atol=1e-12, rtol=0.0)

    produced_v, produced_r = drd_components(matrix)
    np.testing.assert_allclose(produced_v, variances, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(produced_r, expected_corr, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        covariance_from_drd(produced_v, produced_r),
        matrix,
        atol=1e-12,
        rtol=0.0,
    )


def test_pair_vector_roundtrip_frozen_order() -> None:
    correlation = np.array(
        [[1.0, 0.2, -0.3], [0.2, 1.0, 0.4], [-0.3, 0.4, 1.0]]
    )
    rows, cols = unique_pair_indices(3)
    np.testing.assert_array_equal(rows, [0, 0, 1])
    np.testing.assert_array_equal(cols, [1, 2, 2])
    vector = pair_vector_from_correlation(correlation)
    np.testing.assert_allclose(vector, [0.2, -0.3, 0.4])
    rebuilt = correlation_from_pair_vector(vector, 3)
    np.testing.assert_allclose(rebuilt, correlation)


def test_nonoverlapping_lag_widths_and_no_overlap() -> None:
    values = np.arange(30, dtype=float)
    index = 22
    daily = values[index - 1]
    weekly = values[index - 5 : index - 1]
    monthly = values[index - 22 : index - 5]
    assert weekly.size == WEEKLY_WIDTH
    assert monthly.size == MONTHLY_WIDTH
    daily_set = {int(daily)}
    weekly_set = set(weekly.astype(int))
    monthly_set = set(monthly.astype(int))
    assert daily_set.isdisjoint(weekly_set)
    assert daily_set.isdisjoint(monthly_set)
    assert weekly_set.isdisjoint(monthly_set)
    predictors = har_predictors(values, index)
    np.testing.assert_allclose(predictors[0], daily)
    np.testing.assert_allclose(predictors[1], weekly.mean())
    np.testing.assert_allclose(predictors[2], monthly.mean())


def test_t250_has_exactly_228_regression_dates() -> None:
    index = har_regression_indices(250)
    np.testing.assert_array_equal(index, np.arange(22, 250))
    assert index.size == 228


def test_forecast_slices_match_authorized_python_bounds() -> None:
    values = np.arange(250, dtype=float)
    predictors = forecast_predictors(values)
    np.testing.assert_allclose(predictors[0], values[249])
    np.testing.assert_allclose(predictors[1], values[245:249].mean())
    np.testing.assert_allclose(predictors[2], values[228:245].mean())
    assert values[245:249].size == 4
    assert values[228:245].size == 17


def test_within_variance_matches_dummy_ols() -> None:
    rng = np.random.default_rng(SEED)
    n_times, n_assets = 40, 3
    panel = 0.4 + rng.random((n_times, n_assets))
    y, x = har_response_design(panel, har_regression_indices(n_times))
    alpha, beta, rank, n_columns = fixed_effects_shared_slopes(y, x)
    dummy_alpha, dummy_beta = dummy_fixed_effects_shared_slopes(y, x)
    assert n_columns == 3
    assert rank <= 3
    np.testing.assert_allclose(alpha, dummy_alpha, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(beta, dummy_beta, atol=1e-10, rtol=0.0)


def test_within_correlation_matches_dummy_ols() -> None:
    rng = np.random.default_rng(SEED + 1)
    n_times, n_pairs = 40, 3
    panel = 0.05 + 0.2 * rng.random((n_times, n_pairs))
    y, x = har_response_design(panel, har_regression_indices(n_times))
    alpha, beta, rank, n_columns = fixed_effects_shared_slopes(y, x)
    dummy_alpha, dummy_beta = dummy_fixed_effects_shared_slopes(y, x)
    assert n_columns == 3
    assert rank <= 3
    np.testing.assert_allclose(alpha, dummy_alpha, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(beta, dummy_beta, atol=1e-10, rtol=0.0)


def test_noiseless_har_recovers_known_coefficients() -> None:
    n_times = 80
    alpha_d = np.array([0.05, 0.08])
    beta_d = np.array([0.45, 0.25, 0.15])
    alpha_r = np.array([0.02])
    beta_r = np.array([0.35, 0.20, 0.10])
    variances = _simulate_har(n_times, alpha_d, beta_d, start=1.0)
    pairs = _simulate_har(n_times, alpha_r, beta_r, start=0.15)
    assert np.all(variances > 0.0)
    assert np.all(np.abs(pairs) < 1.0)
    cube = np.stack(
        [
            _spd(variances[time], correlation_from_pair_vector(pairs[time], 2))
            for time in range(n_times)
        ],
        axis=0,
    )
    model = HARDRDRealizedCovariance().fit(cube)
    state = model.fit_state
    np.testing.assert_allclose(state.beta_D, beta_d, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(state.alpha_D, alpha_d, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(state.beta_R, beta_r, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(state.alpha_R, alpha_r, atol=1e-8, rtol=0.0)
    forecast = model.forecast()
    assert forecast.identity.name == "har_drd"
    assert forecast.identity.configuration["repaired"] is False
    assert forecast.identity.configuration["lag_convention"] == LAG_CONVENTION
    assert forecast.identity.configuration["pair_ordering"] == PAIR_ORDERING


def test_forecast_ignores_synthetic_target_day() -> None:
    n_times = 40
    variances = np.full((n_times, 2), 1.2)
    cube = _cube_from_variances(variances, _constant_rho(2, 0.25))
    future = _spd(np.array([9.0, 8.0]), _constant_rho(2, 0.8))
    first = HARDRDRealizedCovariance().fit(cube).forecast().matrix
    future[0, 0] += 5.0
    second = HARDRDRealizedCovariance().fit(cube).forecast().matrix
    np.testing.assert_array_equal(first, second)
    with_future = HARDRDRealizedCovariance().fit(
        np.concatenate([cube, future[None]], axis=0)
    )
    assert with_future.fit_state.window_length == n_times + 1
    assert not np.allclose(with_future.forecast().matrix, first)


def test_window_does_not_read_before_start() -> None:
    values = np.arange(250, dtype=float)
    first = har_regression_indices(250)[0]
    assert first - 22 == 0
    monthly = values[first - 22 : first - 5]
    assert monthly[0] == values[0]
    with pytest.raises(InvalidModelInputError):
        har_predictors(values, 21)


def test_asset_permutation_equivariance() -> None:
    n_times = 50
    rng = np.random.default_rng(SEED + 2)
    cube = np.empty((n_times, 3, 3))
    for time in range(n_times):
        factor = rng.normal(size=(3, 3))
        matrix = factor @ factor.T / 3.0 + np.eye(3)
        cube[time] = 0.5 * (matrix + matrix.T)
    order = np.array([2, 0, 1])
    permute = np.eye(3)[order]
    permuted = np.einsum("ij,tjk,lk->til", permute, cube, permute)
    original = HARDRDRealizedCovariance().fit(cube).forecast().matrix
    shuffled = HARDRDRealizedCovariance().fit(permuted).forecast().matrix
    restored = permute.T @ shuffled @ permute
    np.testing.assert_allclose(restored, original, atol=1e-10, rtol=0.0)


def test_pair_order_consistent_under_reconstruction() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 3), 1.1), _constant_rho(3, 0.2))
    model = HARDRDRealizedCovariance().fit(cube)
    model.forecast()
    assert model._x_raw is not None
    rebuilt = correlation_from_pair_vector(model._x_raw, 3)
    np.testing.assert_allclose(rebuilt, model._R_raw)
    rows, cols = unique_pair_indices(3)
    np.testing.assert_allclose(rebuilt[rows, cols], model._x_raw)


def test_zero_diagonal_psd_is_rejected() -> None:
    valid = np.array([[2.0, 0.3], [0.3, 1.5]])
    zero_diag = np.array([[1.0, 0.0], [0.0, 0.0]])
    history = np.stack([valid] * (MIN_WINDOW_LENGTH - 1) + [zero_diag], axis=0)
    with pytest.raises(InvalidModelInputError, match="strictly positive"):
        HARDRDRealizedCovariance().fit(history)


def test_malformed_inputs_use_existing_contract() -> None:
    valid = np.array([[2.0, 0.2], [0.2, 1.5]])
    cube = np.stack([valid] * MIN_WINDOW_LENGTH, axis=0)
    with pytest.raises(InvalidModelInputError):
        HARDRDRealizedCovariance().fit(cube[:2])
    with pytest.raises(InvalidModelInputError):
        HARDRDRealizedCovariance().fit(np.ones((MIN_WINDOW_LENGTH, 2, 3)))
    bad = cube.copy()
    bad[3, 0, 1] = 10.0
    with pytest.raises(InvalidModelInputError):
        HARDRDRealizedCovariance().fit(bad)
    nan = cube.copy()
    nan[5, 0, 0] = np.nan
    with pytest.raises(InvalidModelInputError):
        HARDRDRealizedCovariance().fit(nan)
    with pytest.raises(InvalidModelInputError, match="fit before forecast"):
        HARDRDRealizedCovariance().forecast()


def test_raw_nonpositive_variance_triggers_window_mean_repair() -> None:
    n_times = 40
    variances = np.ones((n_times, 2))
    variances[:, 0] = np.where(np.arange(n_times) % 2 == 0, 2.0, 6.0)
    variances[-1, 0] = 20.0
    cube = _cube_from_variances(variances, _constant_rho(2, 0.2))
    original = cube.copy()
    model = HARDRDRealizedCovariance().fit(cube)
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
    # Noiseless explosive HAR. In-sample |rho|<1, one-step raw |rho|>1.
    pairs = _simulate_har(n_times, np.array([0.02]), np.array([1.25, 0.0, 0.0]), 0.08)
    variances = np.ones((n_times, 2))
    cube = np.stack(
        [
            _spd(variances[time], correlation_from_pair_vector(pairs[time], 2))
            for time in range(n_times)
        ],
        axis=0,
    )
    model = HARDRDRealizedCovariance().fit(cube)
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
    # Same daily slope, opposite intercepts. Pairwise |x_raw|<=1, R_raw not PD.
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
    model = HARDRDRealizedCovariance().fit(cube)
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


def test_valid_raw_forecast_is_not_repaired() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 2), 1.3), _constant_rho(2, 0.2))
    model = HARDRDRealizedCovariance().fit(cube)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert validity.repaired is False
    assert validity.repair_method is None
    assert forecast.diagnostics.positive_definite is True
    np.testing.assert_allclose(forecast.matrix, cube[0], atol=1e-10)


def test_non_pd_fallback_raises_without_second_repair() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    history = np.stack([singular] * MIN_WINDOW_LENGTH, axis=0)
    model = HARDRDRealizedCovariance().fit(history)
    with pytest.raises(InvalidModelForecastError, match="No second repair"):
        model.forecast()


def test_outputs_do_not_alias_inputs() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 2), 1.4), _constant_rho(2, 0.1))
    original = cube.copy()
    model = HARDRDRealizedCovariance().fit(cube)
    forecast = model.forecast()
    assert not np.shares_memory(forecast.matrix, cube)
    forecast.matrix[0, 0] = -99.0
    np.testing.assert_array_equal(cube, original)
    second = model.forecast().matrix
    assert second[0, 0] != -99.0
    assert model._H_raw is None or not np.shares_memory(forecast.matrix, model._H_raw)


def test_deterministic_repeatability() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 2), 1.1), _constant_rho(2, 0.3))
    first = HARDRDRealizedCovariance().fit(cube).forecast().matrix
    second = HARDRDRealizedCovariance().fit(cube).forecast().matrix
    np.testing.assert_array_equal(first, second)


def test_moderate_n_does_not_build_pair_dummy_design() -> None:
    rng = np.random.default_rng(SEED + 3)
    n_times, n_assets = 40, 50
    cube = np.empty((n_times, n_assets, n_assets))
    for time in range(n_times):
        factor = rng.normal(size=(n_assets, 3))
        matrix = factor @ factor.T / 3.0 + np.eye(n_assets)
        cube[time] = 0.5 * (matrix + matrix.T)
    model = HARDRDRealizedCovariance().fit(cube)
    state = model.fit_state
    n_pairs = n_assets * (n_assets - 1) // 2
    n_dates = state.n_regression_dates
    assert state.n_pairs == n_pairs
    assert state.beta_D.shape == (3,)
    assert state.beta_R.shape == (3,)
    assert state.alpha_R.shape == (n_pairs,)
    dummy_columns = n_pairs + 3
    dummy_rows = n_dates * n_pairs
    assert dummy_columns > 1000
    assert dummy_rows > 10_000
    assert state.correlation_within_rank <= 3
    forecast = model.forecast()
    assert forecast.matrix.shape == (n_assets, n_assets)
    assert forecast.diagnostics.positive_definite is True


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
    window = cube[step.window_start_index : step.window_end_index]
    assert window.shape[0] == protocol.m
    cube[step.target_index] = np.array([[50.0, 0.2], [0.2, 40.0]])
    forecast = HARDRDRealizedCovariance().fit(window).forecast()
    assert forecast.matrix.shape == (2, 2)
    assert not np.allclose(forecast.matrix, cube[step.target_index])
    with pytest.raises(ConfirmLockedError):
        protocol.confirm_targets()
    with pytest.raises(ConfirmLockedError):
        protocol.forecast_schedule(BlockName.CONFIRM)
    assert protocol.confirm_locked is True
