"""Synthetic tests for Ridge-DRD as the regularized HAR-DRD control."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from covharness.models import (
    HARDRDRealizedCovariance,
    InvalidModelConfigurationError,
    InvalidModelForecastError,
    InvalidModelInputError,
    RidgeDRDRealizedCovariance,
)
from covharness.models.har_drd import (
    LAG_CONVENTION,
    MIN_WINDOW_LENGTH,
    PAIR_ORDERING,
    REPAIR_METHOD,
    RESPONSE_START,
    WEEKLY_WIDTH,
    MONTHLY_WIDTH,
    correlation_from_pair_vector,
    covariance_from_drd,
    forecast_predictors,
    har_predictors,
    har_regression_indices,
    har_response_design,
    unique_pair_indices,
)
from covharness.models.ridge_drd import (
    HAR_SLOPES,
    PENALTY_CONVENTION,
    SCALING_CONVENTION,
    ridge_fixed_effects_shared_slopes,
    scale_within_predictors,
    within_predictor_scales,
    within_transformed_arrays,
)

SEED = 20260916


def _spd(variances: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    return covariance_from_drd(variances, correlation)


def _constant_rho(n_assets: int, rho: float) -> np.ndarray:
    corr = np.full((n_assets, n_assets), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return corr


def _cube_from_variances(variances: np.ndarray, correlation: np.ndarray) -> np.ndarray:
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


def _random_spd_cube(n_times: int, n_assets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cube = np.empty((n_times, n_assets, n_assets))
    for time in range(n_times):
        factor = rng.normal(size=(n_assets, max(3, n_assets // 8)))
        matrix = factor @ factor.T / factor.shape[1] + np.eye(n_assets)
        cube[time] = 0.5 * (matrix + matrix.T)
    return cube


def test_constructor_requires_nonnegative_lambda() -> None:
    with pytest.raises(TypeError):
        RidgeDRDRealizedCovariance()  # type: ignore[call-arg]
    with pytest.raises(InvalidModelConfigurationError, match="lambda"):
        RidgeDRDRealizedCovariance(lambda_=-0.1)
    with pytest.raises(InvalidModelConfigurationError, match="lambda"):
        RidgeDRDRealizedCovariance(lambda_=np.inf)
    model = RidgeDRDRealizedCovariance(lambda_=0.0)
    assert model.identity.configuration["lambda"] == 0.0


def test_input_contract_matches_har_drd() -> None:
    valid = np.array([[2.0, 0.2], [0.2, 1.5]])
    cube = np.stack([valid] * MIN_WINDOW_LENGTH, axis=0)
    with pytest.raises(InvalidModelInputError):
        RidgeDRDRealizedCovariance(lambda_=0.0).fit(cube[:2])
    with pytest.raises(InvalidModelInputError):
        RidgeDRDRealizedCovariance(lambda_=0.0).fit(np.ones((MIN_WINDOW_LENGTH, 2, 3)))
    bad = cube.copy()
    bad[3, 0, 1] = 10.0
    with pytest.raises(InvalidModelInputError):
        RidgeDRDRealizedCovariance(lambda_=0.0).fit(bad)
    nan = cube.copy()
    nan[5, 0, 0] = np.nan
    with pytest.raises(InvalidModelInputError):
        RidgeDRDRealizedCovariance(lambda_=0.0).fit(nan)
    zero_diag = np.array([[1.0, 0.0], [0.0, 0.0]])
    history = np.stack([valid] * (MIN_WINDOW_LENGTH - 1) + [zero_diag], axis=0)
    with pytest.raises(InvalidModelInputError, match="strictly positive"):
        RidgeDRDRealizedCovariance(lambda_=0.0).fit(history)
    with pytest.raises(InvalidModelInputError, match="fit before forecast"):
        RidgeDRDRealizedCovariance(lambda_=0.0).forecast()


def test_nonoverlapping_feature_construction_matches_har() -> None:
    values = np.arange(30, dtype=float)
    index = 22
    daily = values[index - 1]
    weekly = values[index - 5 : index - 1]
    monthly = values[index - 22 : index - 5]
    assert weekly.size == WEEKLY_WIDTH
    assert monthly.size == MONTHLY_WIDTH
    predictors = har_predictors(values, index)
    np.testing.assert_allclose(predictors[0], daily)
    np.testing.assert_allclose(predictors[1], weekly.mean())
    np.testing.assert_allclose(predictors[2], monthly.mean())


def test_t250_has_exactly_228_regression_dates() -> None:
    index = har_regression_indices(250)
    np.testing.assert_array_equal(index, np.arange(22, 250))
    assert index.size == 228
    cube = _cube_from_variances(np.full((250, 2), 1.2), _constant_rho(2, 0.2))
    state = RidgeDRDRealizedCovariance(lambda_=1.0).fit(cube).fit_state
    assert state.n_regression_dates == 228


def test_within_arrays_match_har_construction() -> None:
    rng = np.random.default_rng(SEED)
    panel = 0.4 + rng.random((40, 3))
    y, x = har_response_design(panel, har_regression_indices(40))
    y_bar, x_bar, y_tilde, x_tilde = within_transformed_arrays(y, x)
    np.testing.assert_allclose(y_bar, y.mean(axis=1))
    np.testing.assert_allclose(x_bar, x.mean(axis=1))
    np.testing.assert_allclose(y_tilde, (y - y.mean(axis=1, keepdims=True)).reshape(-1))
    np.testing.assert_allclose(
        x_tilde, (x - x.mean(axis=1, keepdims=True)).reshape(y_tilde.size, 3)
    )
    assert x_tilde.shape[1] == HAR_SLOPES


def test_predictor_scales_match_rms_and_response_is_not_scaled() -> None:
    rng = np.random.default_rng(SEED + 1)
    panel = 0.5 + rng.random((40, 3))
    y, x = har_response_design(panel, har_regression_indices(40))
    _y_bar, _x_bar, y_tilde, x_tilde = within_transformed_arrays(y, x)
    scales = within_predictor_scales(x_tilde)
    expected = np.sqrt(np.mean(x_tilde * x_tilde, axis=0))
    np.testing.assert_allclose(scales, expected)
    x_scaled = scale_within_predictors(x_tilde, scales)
    np.testing.assert_allclose(x_scaled, x_tilde / expected)
    assert not np.allclose(y_tilde, y_tilde / np.sqrt(np.mean(y_tilde * y_tilde)))


def test_hand_constructed_scale_is_rms_not_mean_square() -> None:
    # Already within-demeaned. Expected RMS is computed by hand, not by the helper.
    x_tilde = np.array(
        [
            [-2.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    expected_first = float(np.sqrt((4.0 + 0.0 + 4.0) / 3.0))
    mean_square_first = (4.0 + 0.0 + 4.0) / 3.0
    assert expected_first == pytest.approx(np.sqrt(8.0 / 3.0))
    assert expected_first != pytest.approx(mean_square_first)
    scales = within_predictor_scales(x_tilde)
    assert scales[0] == pytest.approx(expected_first)
    assert scales[0] != pytest.approx(mean_square_first)
    assert scales[1] == 1.0


def test_exact_zero_within_column_keeps_scale_one() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.zeros((2, 2, 3))
    x[:, :, 0] = [[1.0, 3.0], [2.0, 4.0]]
    x[:, :, 1] = 0.0
    x[:, :, 2] = [[0.5, 1.5], [1.0, 2.0]]
    _y_bar, _x_bar, y_tilde, x_tilde = within_transformed_arrays(y, x)
    scales = within_predictor_scales(x_tilde)
    assert scales[1] == 1.0
    x_scaled = scale_within_predictors(x_tilde, scales)
    np.testing.assert_array_equal(x_scaled[:, 1], 0.0)
    np.testing.assert_array_equal(x_tilde[:, 1], 0.0)
    del y_tilde


def test_production_gamma_matches_sklearn_cholesky_for_positive_lambda() -> None:
    cube = _random_spd_cube(50, 3, SEED + 4)
    penalty = 2.5
    model = RidgeDRDRealizedCovariance(lambda_=penalty).fit(cube)
    variances = model._variance_panel
    assert variances is not None
    y, x = har_response_design(variances, har_regression_indices(50))
    _y_bar, _x_bar, y_tilde, x_tilde = within_transformed_arrays(y, x)
    scales = within_predictor_scales(x_tilde)
    x_scaled = scale_within_predictors(x_tilde, scales)
    reference = Ridge(alpha=penalty, fit_intercept=False, solver="cholesky", copy_X=True)
    reference.fit(x_scaled, y_tilde)
    np.testing.assert_allclose(model.fit_state.gamma_D, reference.coef_, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        model.fit_state.beta_D, reference.coef_ / scales, atol=1e-10, rtol=0.0
    )


def test_lambda_zero_nests_har_coefficients_and_forecast() -> None:
    cube = _random_spd_cube(60, 3, SEED + 5)
    har = HARDRDRealizedCovariance().fit(cube)
    ridge = RidgeDRDRealizedCovariance(lambda_=0.0).fit(cube)
    np.testing.assert_allclose(ridge.fit_state.beta_D, har.fit_state.beta_D, atol=1e-12)
    np.testing.assert_allclose(ridge.fit_state.alpha_D, har.fit_state.alpha_D, atol=1e-12)
    np.testing.assert_allclose(ridge.fit_state.beta_R, har.fit_state.beta_R, atol=1e-12)
    np.testing.assert_allclose(ridge.fit_state.alpha_R, har.fit_state.alpha_R, atol=1e-12)
    har_forecast = har.forecast()
    ridge_forecast = ridge.forecast()
    np.testing.assert_allclose(ridge_forecast.matrix, har_forecast.matrix, atol=1e-12)
    assert ridge_forecast.identity.configuration["repaired"] == (
        har_forecast.identity.configuration["repaired"]
    )


def test_intercepts_are_not_penalized() -> None:
    rng = np.random.default_rng(SEED + 6)
    panel = 1.0 + rng.random((40, 4))
    y, x = har_response_design(panel, har_regression_indices(40))
    alpha, gamma, beta, scales, _rank, width = ridge_fixed_effects_shared_slopes(
        y, x, penalty=8.0
    )
    y_bar, x_bar, _y_tilde, _x_tilde = within_transformed_arrays(y, x)
    np.testing.assert_allclose(alpha, y_bar - x_bar @ beta, atol=1e-12)
    assert gamma.shape == (3,)
    assert beta.shape == (3,)
    assert width == 3
    assert scales.shape == (3,)


def test_larger_lambda_shrinks_scaled_slope_norm() -> None:
    rng = np.random.default_rng(SEED + 7)
    panel = 0.8 + rng.random((50, 3))
    y, x = har_response_design(panel, har_regression_indices(50))
    _a_small, gamma_small, _b_small, _s, _r, _w = ridge_fixed_effects_shared_slopes(
        y, x, penalty=0.5
    )
    _a_large, gamma_large, _b_large, _s2, _r2, _w2 = ridge_fixed_effects_shared_slopes(
        y, x, penalty=25.0
    )
    assert float(np.linalg.norm(gamma_large)) < float(np.linalg.norm(gamma_small))


def test_one_lambda_shared_by_variance_and_correlation() -> None:
    cube = _random_spd_cube(40, 3, SEED + 8)
    penalty = 3.0
    state = RidgeDRDRealizedCovariance(lambda_=penalty).fit(cube).fit_state
    assert state.ridge_lambda == penalty
    assert state.beta_D.shape == (3,)
    assert state.beta_R.shape == (3,)
    assert state.gamma_D.shape == (3,)
    assert state.gamma_R.shape == (3,)


def test_no_quarticity_cross_section_log_or_fisher() -> None:
    cube = _random_spd_cube(40, 3, SEED + 9)
    model = RidgeDRDRealizedCovariance(lambda_=1.0).fit(cube)
    configuration = model.identity.configuration
    assert configuration["n_slope_columns"] == 3
    assert configuration["slope_names"] == ("daily", "weekly", "monthly")
    assert configuration["responses_standardized"] is False
    assert "quarticity" not in configuration
    assert model._variance_panel is not None
    np.testing.assert_allclose(
        model._variance_panel[-1], np.diag(cube[-1]), atol=1e-12
    )
    y, x = har_response_design(model._variance_panel, har_regression_indices(40))
    np.testing.assert_allclose(y[:, -1], np.diag(cube[39]))
    assert np.all(y > 0.0)
    assert model._pair_panel is not None
    assert np.all(np.abs(model._pair_panel) <= 1.0 + 1e-12)
    del x


def test_asset_permutation_equivariance() -> None:
    cube = _random_spd_cube(50, 3, SEED + 10)
    order = np.array([2, 0, 1])
    permute = np.eye(3)[order]
    permuted = np.einsum("ij,tjk,lk->til", permute, cube, permute)
    original = RidgeDRDRealizedCovariance(lambda_=1.5).fit(cube).forecast().matrix
    shuffled = RidgeDRDRealizedCovariance(lambda_=1.5).fit(permuted).forecast().matrix
    restored = permute.T @ shuffled @ permute
    np.testing.assert_allclose(restored, original, atol=1e-10, rtol=0.0)


def test_pair_order_is_strict_upper_triangle() -> None:
    cube = _cube_from_variances(np.full((40, 3), 1.1), _constant_rho(3, 0.2))
    model = RidgeDRDRealizedCovariance(lambda_=0.5).fit(cube)
    model.forecast()
    assert model.fit_state.pair_ordering == PAIR_ORDERING
    rows, cols = unique_pair_indices(3)
    np.testing.assert_array_equal(rows, [0, 0, 1])
    np.testing.assert_array_equal(cols, [1, 2, 2])
    assert model._x_raw is not None
    rebuilt = correlation_from_pair_vector(model._x_raw, 3)
    np.testing.assert_allclose(rebuilt, model._R_raw)
    np.testing.assert_allclose(rebuilt[rows, cols], model._x_raw)


def test_no_lookahead_and_no_read_before_window() -> None:
    n_times = 40
    cube = _cube_from_variances(np.full((n_times, 2), 1.2), _constant_rho(2, 0.25))
    future = _spd(np.array([9.0, 8.0]), _constant_rho(2, 0.8))
    first = RidgeDRDRealizedCovariance(lambda_=1.0).fit(cube).forecast().matrix
    future[0, 0] += 5.0
    second = RidgeDRDRealizedCovariance(lambda_=1.0).fit(cube).forecast().matrix
    np.testing.assert_array_equal(first, second)
    values = np.arange(250, dtype=float)
    first_index = har_regression_indices(250)[0]
    assert first_index - 22 == 0
    monthly = values[first_index - 22 : first_index - 5]
    assert monthly[0] == values[0]
    with pytest.raises(InvalidModelInputError):
        har_predictors(values, 21)
    origin = forecast_predictors(values)
    np.testing.assert_allclose(origin[0], values[249])


def test_valid_raw_forecast_is_not_repaired() -> None:
    cube = _cube_from_variances(np.full((40, 2), 1.3), _constant_rho(2, 0.2))
    model = RidgeDRDRealizedCovariance(lambda_=0.0).fit(cube)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert validity.repaired is False
    assert validity.repair_method is None
    assert forecast.diagnostics.positive_definite is True


def test_raw_nonpositive_variance_uses_window_mean_fallback() -> None:
    n_times = 40
    variances = np.ones((n_times, 2))
    variances[:, 0] = np.where(np.arange(n_times) % 2 == 0, 2.0, 6.0)
    variances[-1, 0] = 20.0
    cube = _cube_from_variances(variances, _constant_rho(2, 0.2))
    original = cube.copy()
    model = RidgeDRDRealizedCovariance(lambda_=0.0).fit(cube)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert validity.repaired is True
    assert validity.repair_method == REPAIR_METHOD
    assert validity.raw_nonpositive_variance is True
    np.testing.assert_allclose(forecast.matrix, original.mean(axis=0))
    np.testing.assert_array_equal(cube, original)


def test_raw_out_of_bounds_correlation_uses_same_fallback() -> None:
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
    model = RidgeDRDRealizedCovariance(lambda_=0.0).fit(cube)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert model._x_raw is not None
    assert np.any(np.abs(model._x_raw) > 1.0)
    assert validity.repaired is True
    assert validity.raw_correlation_out_of_bounds is True
    np.testing.assert_allclose(forecast.matrix, cube.mean(axis=0))


def test_in_bounds_non_pd_r_uses_same_fallback() -> None:
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
    model = RidgeDRDRealizedCovariance(lambda_=0.0).fit(cube)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert model._x_raw is not None
    assert np.all(np.abs(model._x_raw) <= 1.0)
    assert validity.raw_correlation_out_of_bounds is False
    assert validity.repaired is True
    assert validity.raw_correlation_not_pd is True
    np.testing.assert_allclose(forecast.matrix, cube.mean(axis=0))


def test_non_pd_fallback_raises_without_second_repair() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    history = np.stack([singular] * MIN_WINDOW_LENGTH, axis=0)
    model = RidgeDRDRealizedCovariance(lambda_=0.0).fit(history)
    with pytest.raises(InvalidModelForecastError, match="No second repair"):
        model.forecast()


def test_inputs_not_mutated_and_forecast_does_not_alias() -> None:
    cube = _cube_from_variances(np.full((40, 2), 1.4), _constant_rho(2, 0.1))
    original = cube.copy()
    model = RidgeDRDRealizedCovariance(lambda_=1.0).fit(cube)
    forecast = model.forecast()
    assert not np.shares_memory(forecast.matrix, cube)
    forecast.matrix[0, 0] = -99.0
    np.testing.assert_array_equal(cube, original)
    second = model.forecast().matrix
    assert second[0, 0] != -99.0
    assert model._H_raw is None or not np.shares_memory(forecast.matrix, model._H_raw)


def test_deterministic_repeatability() -> None:
    cube = _random_spd_cube(40, 3, SEED + 11)
    first = RidgeDRDRealizedCovariance(lambda_=2.0).fit(cube).forecast().matrix
    second = RidgeDRDRealizedCovariance(lambda_=2.0).fit(cube).forecast().matrix
    np.testing.assert_array_equal(first, second)


def test_moderate_n_has_three_slope_columns_and_no_dummy_matrix() -> None:
    cube = _random_spd_cube(40, 50, SEED + 12)
    model = RidgeDRDRealizedCovariance(lambda_=1.0).fit(cube)
    state = model.fit_state
    n_pairs = 50 * 49 // 2
    assert state.n_pairs == n_pairs
    assert state.beta_D.shape == (3,)
    assert state.beta_R.shape == (3,)
    assert state.gamma_D.shape == (3,)
    assert state.gamma_R.shape == (3,)
    assert state.scale_D.shape == (3,)
    assert state.scale_R.shape == (3,)
    assert state.alpha_D.shape == (50,)
    assert state.alpha_R.shape == (n_pairs,)
    dummy_columns = n_pairs + 3
    dummy_rows = state.n_regression_dates * n_pairs
    assert dummy_columns > 1000
    assert dummy_rows > 10_000
    assert state.correlation_within_rank <= 3
    forecast = model.forecast()
    assert forecast.matrix.shape == (50, 50)


def test_architecture_isolation_same_information_only_penalty_differs() -> None:
    cube = _random_spd_cube(55, 3, SEED + 13)
    har = HARDRDRealizedCovariance().fit(cube)
    zero = RidgeDRDRealizedCovariance(lambda_=0.0).fit(cube)
    penalized = RidgeDRDRealizedCovariance(lambda_=4.0).fit(cube)
    np.testing.assert_allclose(zero.forecast().matrix, har.forecast().matrix, atol=1e-12)
    assert har._variance_panel is not None and penalized._variance_panel is not None
    np.testing.assert_array_equal(har._variance_panel, penalized._variance_panel)
    assert har._pair_panel is not None and penalized._pair_panel is not None
    np.testing.assert_array_equal(har._pair_panel, penalized._pair_panel)
    har_y, har_x = har_response_design(har._variance_panel, har_regression_indices(55))
    ridge_y, ridge_x = har_response_design(
        penalized._variance_panel, har_regression_indices(55)
    )
    np.testing.assert_array_equal(har_y, ridge_y)
    np.testing.assert_array_equal(har_x, ridge_x)
    assert not np.allclose(penalized.fit_state.beta_D, har.fit_state.beta_D)
    assert penalized.fit_state.ridge_lambda == 4.0
    assert zero.fit_state.ridge_lambda == 0.0
    assert penalized.fit_state.lag_convention == LAG_CONVENTION
    assert penalized.fit_state.scaling_convention == SCALING_CONVENTION
    assert penalized.fit_state.penalty_convention == PENALTY_CONVENTION
    assert penalized.forecast().identity.name == "ridge_drd"
