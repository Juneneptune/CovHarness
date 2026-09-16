"""Focused tests for daily origin-state APIs on implemented models."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.models import (
    EWMARealizedCovariance,
    HARDRDRealizedCovariance,
    HARQDRDRealizedCovariance,
    InvalidModelInputError,
    LedoitWolfLinearCovariance,
    LedoitWolfNonlinearCovariance,
    RandomWalkRealizedCovariance,
    RidgeDRDRealizedCovariance,
    RollingCadence,
    XGBoostDRDRealizedCovariance,
    capabilities_of,
)
from covharness.models.xgboost_drd import booster_raw_bytes
from covharness.models.har_drd import (
    MIN_WINDOW_LENGTH,
    _drd_panel,
    forecast_predictors,
    pair_panel_from_correlations,
)
from covharness.models.harq_drd import harq_forecast_predictors

SEED = 20260916
SPD_A = np.array([[2.0, 0.4], [0.4, 1.0]])
SPD_B = np.array([[1.5, 0.2], [0.2, 1.2]])
SPD_C = np.array([[3.0, -0.1], [-0.1, 2.5]])
SPD_D = np.array([[2.2, 0.3], [0.3, 1.8]])


def _history(*matrices: np.ndarray) -> np.ndarray:
    return np.stack(matrices, axis=0)


def _random_spd_cube(n_times: int, n_assets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cube = np.empty((n_times, n_assets, n_assets))
    for time in range(n_times):
        factor = rng.normal(size=(n_assets, max(3, n_assets)))
        matrix = factor @ factor.T / factor.shape[1] + np.eye(n_assets)
        cube[time] = 0.5 * (matrix + matrix.T)
    return cube


def _rq_panel(n_times: int, n_assets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 1.8, size=(n_times, n_assets))


def test_ewma_one_update_matches_hand_recursion() -> None:
    decay = 0.6
    fitted = decay * SPD_A + (1.0 - decay) * SPD_B
    expected = decay * fitted + (1.0 - decay) * SPD_C
    model = EWMARealizedCovariance(decay=decay).fit(_history(SPD_A, SPD_B))
    model.update(SPD_C)
    np.testing.assert_allclose(model.forecast().matrix, expected)


def test_ewma_lambda_unchanged_by_update() -> None:
    model = EWMARealizedCovariance(decay=0.94).fit(_history(SPD_A, SPD_B))
    assert model.identity.configuration["decay"] == 0.94
    model.update(SPD_C)
    assert model.identity.configuration["decay"] == 0.94
    assert model._decay == 0.94


def test_ewma_two_updates_equal_two_hand_steps() -> None:
    decay = 0.7
    one_minus = 1.0 - decay
    state = decay * SPD_A + one_minus * SPD_B
    state = decay * state + one_minus * SPD_C
    state = decay * state + one_minus * SPD_D
    model = EWMARealizedCovariance(decay=decay).fit(_history(SPD_A, SPD_B))
    model.update(SPD_C)
    model.update(SPD_D)
    np.testing.assert_allclose(model.forecast().matrix, state)


def test_ewma_update_input_validation() -> None:
    model = EWMARealizedCovariance(decay=0.5).fit(_history(SPD_A, SPD_B))
    with pytest.raises(InvalidModelInputError):
        model.update(np.array([[1.0, 0.0]]))
    with pytest.raises(InvalidModelInputError):
        model.update(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(InvalidModelInputError):
        model.update(np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(InvalidModelInputError):
        model.update(np.array([[1.0, 0.0], [0.0, -0.2]]))
    with pytest.raises(InvalidModelInputError):
        model.update(np.eye(3))


def test_ewma_forecast_does_not_mutate_state() -> None:
    model = EWMARealizedCovariance(decay=0.5).fit(_history(SPD_A, SPD_B))
    first = model.forecast().matrix
    stored = first.copy()
    first[0, 0] = -99.0
    second = model.forecast().matrix
    np.testing.assert_allclose(second, stored)
    model.update(SPD_C)
    after = model.forecast().matrix
    after_copy = after.copy()
    after[1, 1] = -50.0
    np.testing.assert_allclose(model.forecast().matrix, after_copy)


def test_rw_successive_origins_use_successive_s() -> None:
    model = RandomWalkRealizedCovariance().fit(_history(SPD_A, SPD_B))
    np.testing.assert_allclose(model.forecast().matrix, SPD_B)
    model.update(SPD_C)
    np.testing.assert_allclose(model.forecast().matrix, SPD_C)
    model.update(SPD_D)
    np.testing.assert_allclose(model.forecast().matrix, SPD_D)
    assert capabilities_of(model).rolling_cadence is RollingCadence.ORIGIN_MAP


def test_har_coefficients_frozen_and_origin_features_exact() -> None:
    cube = _random_spd_cube(30, 2, SEED)
    model = HARDRDRealizedCovariance().fit(cube[:MIN_WINDOW_LENGTH])
    alpha_d = model.fit_state.alpha_D.copy()
    beta_d = model.fit_state.beta_D.copy()
    alpha_r = model.fit_state.alpha_R.copy()
    beta_r = model.fit_state.beta_R.copy()
    n_dates = model.fit_state.n_regression_dates
    first_mean = model.fit_state.window_mean.copy()
    new_window = cube[1 : 1 + MIN_WINDOW_LENGTH]
    model.update_window(new_window)
    np.testing.assert_array_equal(model.fit_state.alpha_D, alpha_d)
    np.testing.assert_array_equal(model.fit_state.beta_D, beta_d)
    np.testing.assert_array_equal(model.fit_state.alpha_R, alpha_r)
    np.testing.assert_array_equal(model.fit_state.beta_R, beta_r)
    assert model.fit_state.n_regression_dates == n_dates
    variances, correlations = _drd_panel(new_window)
    expected_v = forecast_predictors(variances).T
    expected_x = forecast_predictors(pair_panel_from_correlations(correlations)).T
    got_v, got_x = model._origin_predictors()
    np.testing.assert_allclose(got_v, expected_v)
    np.testing.assert_allclose(got_x, expected_x)
    np.testing.assert_allclose(model.fit_state.window_mean, new_window.mean(axis=0))
    assert not np.allclose(model.fit_state.window_mean, first_mean)


def test_har_update_window_does_not_refit() -> None:
    cube = _random_spd_cube(28, 2, SEED + 1)
    model = HARDRDRealizedCovariance().fit(cube[:MIN_WINDOW_LENGTH])
    fit_calls = {"n": 0}
    original = model.fit

    def counted(history):
        fit_calls["n"] += 1
        return original(history)

    model.fit = counted  # type: ignore[method-assign]
    model.update_window(cube[2 : 2 + MIN_WINDOW_LENGTH])
    model.forecast()
    assert fit_calls["n"] == 0


def test_harq_coefficients_rq_and_fallback_current() -> None:
    cube = _random_spd_cube(30, 2, SEED + 2)
    rq = _rq_panel(30, 2, SEED + 3)
    rq[1 + MIN_WINDOW_LENGTH] = rq[MIN_WINDOW_LENGTH] + 1.7
    model = HARQDRDRealizedCovariance().fit(cube[:MIN_WINDOW_LENGTH], rq[:MIN_WINDOW_LENGTH])
    alpha_q = model.fit_state.alpha_Q.copy()
    beta_q = model.fit_state.beta_Q.copy()
    alpha_r = model.fit_state.alpha_R.copy()
    beta_r = model.fit_state.beta_R.copy()
    new_window = cube[1 : 1 + MIN_WINDOW_LENGTH]
    new_rq = rq[1 : 1 + MIN_WINDOW_LENGTH]
    target_rq = rq[1 + MIN_WINDOW_LENGTH]
    model.update_window(new_window, new_rq)
    np.testing.assert_array_equal(model.fit_state.alpha_Q, alpha_q)
    np.testing.assert_array_equal(model.fit_state.beta_Q, beta_q)
    np.testing.assert_array_equal(model.fit_state.alpha_R, alpha_r)
    np.testing.assert_array_equal(model.fit_state.beta_R, beta_r)
    variances, _correlations = _drd_panel(new_window)
    expected_v = harq_forecast_predictors(variances, new_rq).T
    got_v, _got_x = model._origin_predictors()
    np.testing.assert_allclose(got_v, expected_v)
    leaked = np.array(new_rq, copy=True)
    leaked[-1] = target_rq
    leaked_v = harq_forecast_predictors(variances, leaked).T
    assert not np.allclose(got_v, leaked_v)
    np.testing.assert_allclose(model.fit_state.window_mean, new_window.mean(axis=0))


def test_ridge_scales_coefs_features_and_fallback() -> None:
    cube = _random_spd_cube(30, 2, SEED + 4)
    model = RidgeDRDRealizedCovariance(lambda_=0.75).fit(cube[:MIN_WINDOW_LENGTH])
    ridge_lambda = model.fit_state.ridge_lambda
    scale_d = model.fit_state.scale_D.copy()
    scale_r = model.fit_state.scale_R.copy()
    gamma_d = model.fit_state.gamma_D.copy()
    beta_d = model.fit_state.beta_D.copy()
    first_features = model._origin_predictors()
    first_mean = model.fit_state.window_mean.copy()
    new_window = cube[3 : 3 + MIN_WINDOW_LENGTH]
    model.update_window(new_window)
    assert model.fit_state.ridge_lambda == ridge_lambda
    np.testing.assert_array_equal(model.fit_state.scale_D, scale_d)
    np.testing.assert_array_equal(model.fit_state.scale_R, scale_r)
    np.testing.assert_array_equal(model.fit_state.gamma_D, gamma_d)
    np.testing.assert_array_equal(model.fit_state.beta_D, beta_d)
    new_features = model._origin_predictors()
    assert not np.allclose(new_features[0], first_features[0])
    variances, correlations = _drd_panel(new_window)
    expected_v = forecast_predictors(variances).T
    expected_x = forecast_predictors(pair_panel_from_correlations(correlations)).T
    np.testing.assert_allclose(new_features[0], expected_v)
    np.testing.assert_allclose(new_features[1], expected_x)
    np.testing.assert_allclose(model.fit_state.window_mean, new_window.mean(axis=0))
    assert not np.allclose(model.fit_state.window_mean, first_mean)


def test_ridge_lambda_zero_tracks_har_across_refreshes() -> None:
    cube = _random_spd_cube(32, 2, SEED + 5)
    har = HARDRDRealizedCovariance().fit(cube[:MIN_WINDOW_LENGTH])
    ridge = RidgeDRDRealizedCovariance(lambda_=0.0).fit(cube[:MIN_WINDOW_LENGTH])
    np.testing.assert_allclose(har.forecast().matrix, ridge.forecast().matrix, atol=1e-10)
    for shift in range(1, 6):
        window = cube[shift : shift + MIN_WINDOW_LENGTH]
        har.update_window(window)
        ridge.update_window(window)
        np.testing.assert_array_equal(ridge.fit_state.scale_D, ridge.fit_state.scale_D)
        np.testing.assert_allclose(har.fit_state.window_mean, window.mean(axis=0))
        np.testing.assert_allclose(ridge.fit_state.window_mean, window.mean(axis=0))
        np.testing.assert_allclose(
            har.forecast().matrix, ridge.forecast().matrix, atol=1e-10
        )


def test_xgboost_maps_frozen_and_fallback_current() -> None:
    cube = _random_spd_cube(32, 2, SEED + 8)
    model = XGBoostDRDRealizedCovariance(
        n_estimators=2,
        max_depth=1,
        learning_rate=0.3,
        min_child_weight=0.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        gamma=0.0,
    ).fit(cube[:MIN_WINDOW_LENGTH])
    ybar_d = model.fit_state.ybar_D.copy()
    xbar_d = model.fit_state.Xbar_D.copy()
    scale_d = model.fit_state.scale_D.copy()
    origin_d = model.fit_state.origin_X_D.copy()
    mean0 = model.fit_state.window_mean.copy()
    bytes_d = booster_raw_bytes(model.fit_state.variance_booster)
    new_window = cube[2 : 2 + MIN_WINDOW_LENGTH]
    model.update_window(new_window)
    np.testing.assert_array_equal(model.fit_state.ybar_D, ybar_d)
    np.testing.assert_array_equal(model.fit_state.Xbar_D, xbar_d)
    np.testing.assert_array_equal(model.fit_state.scale_D, scale_d)
    assert booster_raw_bytes(model.fit_state.variance_booster) == bytes_d
    assert not np.allclose(model.fit_state.origin_X_D, origin_d)
    np.testing.assert_allclose(model.fit_state.window_mean, new_window.mean(axis=0))
    assert not np.allclose(model.fit_state.window_mean, mean0)
    variances, correlations = _drd_panel(new_window)
    expected_v = forecast_predictors(variances).T
    expected_x = forecast_predictors(pair_panel_from_correlations(correlations)).T
    np.testing.assert_allclose(model.fit_state.origin_X_D, expected_v)
    np.testing.assert_allclose(model.fit_state.origin_X_R, expected_x)


def test_lw_non_refit_forecast_unchanged_until_refit() -> None:
    rng = np.random.default_rng(SEED + 6)
    returns = rng.normal(size=(40, 3))
    linear = LedoitWolfLinearCovariance().fit(returns[:20])
    nonlinear = LedoitWolfNonlinearCovariance().fit(returns[:20])
    h_lin = linear.forecast().matrix.copy()
    h_nl = nonlinear.forecast().matrix.copy()
    np.testing.assert_array_equal(linear.forecast().matrix, h_lin)
    np.testing.assert_array_equal(nonlinear.forecast().matrix, h_nl)
    assert not hasattr(linear, "update")
    assert not hasattr(nonlinear, "update")
    assert capabilities_of(linear).rolling_cadence is RollingCadence.REFIT_HOLD
    unused = returns[20:40]
    assert unused.shape[0] == 20
    np.testing.assert_array_equal(linear.forecast().matrix, h_lin)
    np.testing.assert_array_equal(nonlinear.forecast().matrix, h_nl)
    linear.fit(returns[10:30])
    nonlinear.fit(returns[10:30])
    assert not np.allclose(linear.forecast().matrix, h_lin)
    assert not np.allclose(nonlinear.forecast().matrix, h_nl)
