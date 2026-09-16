"""Synthetic tests for XGBoost-DRD as the nonlinear HAR/Ridge architecture control."""

from __future__ import annotations

import inspect
from importlib import metadata

import numpy as np
import pytest
import xgboost
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

from covharness.models import (
    HARDRDRealizedCovariance,
    InvalidModelConfigurationError,
    InvalidModelForecastError,
    InvalidModelInputError,
    RidgeDRDRealizedCovariance,
    RollingCadence,
    XGBoostDRDRealizedCovariance,
    capabilities_of,
)
from covharness.models.har_drd import (
    HAR_SLOPES,
    LAG_CONVENTION,
    MIN_WINDOW_LENGTH,
    PAIR_ORDERING,
    REPAIR_METHOD,
    RESPONSE_START,
    WEEKLY_WIDTH,
    MONTHLY_WIDTH,
    _drd_panel,
    covariance_from_drd,
    forecast_predictors,
    har_predictors,
    har_regression_indices,
    har_response_design,
    pair_panel_from_correlations,
    unique_pair_indices,
)
from covharness.models.ridge_drd import (
    SCALING_CONVENTION,
    scale_within_predictors,
    within_predictor_scales,
    within_transformed_arrays,
)
from covharness.models.xgboost_drd import (
    BASE_SCORE,
    BOOSTER_TYPE,
    COLSAMPLE_BYLEVEL,
    COLSAMPLE_BYNODE,
    COLSAMPLE_BYTREE,
    DEVICE,
    EARLY_STOPPING,
    GROW_POLICY,
    MAX_BIN,
    MODEL_NAME,
    N_BOOSTERS,
    N_JOBS,
    OBJECTIVE,
    PINNED_XGBOOST_VERSION,
    RANDOM_STATE,
    SUBSAMPLE,
    TREE_METHOD,
    WITHIN_CONVENTION,
    booster_raw_bytes,
    fit_squared_error_booster,
    origin_scaled_predictors,
    scaled_within_design,
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


def _random_spd_cube(n_times: int, n_assets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cube = np.empty((n_times, n_assets, n_assets))
    for time in range(n_times):
        factor = rng.normal(size=(n_assets, max(3, n_assets // 8)))
        matrix = factor @ factor.T / factor.shape[1] + np.eye(n_assets)
        cube[time] = 0.5 * (matrix + matrix.T)
    return cube


def _small_model() -> XGBoostDRDRealizedCovariance:
    return XGBoostDRDRealizedCovariance(
        n_estimators=8,
        max_depth=2,
        learning_rate=0.2,
        min_child_weight=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
    )


def _tiny_model() -> XGBoostDRDRealizedCovariance:
    return XGBoostDRDRealizedCovariance(
        n_estimators=2,
        max_depth=1,
        learning_rate=0.3,
        min_child_weight=0.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        gamma=0.0,
    )


class _ConstantPredictor:
    """Stub booster used only to force a known raw forecast."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=float)

    def predict(self, data: np.ndarray) -> np.ndarray:
        n_rows = int(np.asarray(data).shape[0])
        if self._values.ndim == 0 or self._values.size == 1:
            return np.full(n_rows, float(np.reshape(self._values, -1)[0]))
        return np.asarray(self._values, dtype=float)


def test_constructor_requires_explicit_hyperparameters() -> None:
    with pytest.raises(TypeError):
        XGBoostDRDRealizedCovariance()  # type: ignore[call-arg]
    with pytest.raises(InvalidModelConfigurationError, match="n_estimators"):
        XGBoostDRDRealizedCovariance(
            n_estimators=0,
            max_depth=1,
            learning_rate=0.1,
            min_child_weight=0.0,
            reg_lambda=0.0,
            reg_alpha=0.0,
            gamma=0.0,
        )
    with pytest.raises(InvalidModelConfigurationError, match="max_depth"):
        XGBoostDRDRealizedCovariance(
            n_estimators=1,
            max_depth=0,
            learning_rate=0.1,
            min_child_weight=0.0,
            reg_lambda=0.0,
            reg_alpha=0.0,
            gamma=0.0,
        )
    with pytest.raises(InvalidModelConfigurationError, match="learning_rate"):
        XGBoostDRDRealizedCovariance(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.0,
            min_child_weight=0.0,
            reg_lambda=0.0,
            reg_alpha=0.0,
            gamma=0.0,
        )
    with pytest.raises(InvalidModelConfigurationError, match="min_child_weight"):
        XGBoostDRDRealizedCovariance(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            min_child_weight=-0.1,
            reg_lambda=0.0,
            reg_alpha=0.0,
            gamma=0.0,
        )
    with pytest.raises(InvalidModelConfigurationError, match="reg_lambda"):
        XGBoostDRDRealizedCovariance(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            min_child_weight=0.0,
            reg_lambda=-1.0,
            reg_alpha=0.0,
            gamma=0.0,
        )
    with pytest.raises(InvalidModelConfigurationError, match="reg_alpha"):
        XGBoostDRDRealizedCovariance(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            min_child_weight=0.0,
            reg_lambda=0.0,
            reg_alpha=np.inf,
            gamma=0.0,
        )
    with pytest.raises(InvalidModelConfigurationError, match="gamma"):
        XGBoostDRDRealizedCovariance(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            min_child_weight=0.0,
            reg_lambda=0.0,
            reg_alpha=0.0,
            gamma=-0.2,
        )
    model = _small_model()
    assert model.identity.name == MODEL_NAME
    assert model.identity.configuration["n_estimators"] == 8


def test_input_contract_matches_har_ridge() -> None:
    valid = np.array([[2.0, 0.2], [0.2, 1.5]])
    cube = np.stack([valid] * MIN_WINDOW_LENGTH, axis=0)
    with pytest.raises(InvalidModelInputError):
        _small_model().fit(cube[:2])
    with pytest.raises(InvalidModelInputError):
        _small_model().fit(np.ones((MIN_WINDOW_LENGTH, 2, 3)))
    bad = cube.copy()
    bad[3, 0, 1] = 10.0
    with pytest.raises(InvalidModelInputError):
        _small_model().fit(bad)
    nan = cube.copy()
    nan[5, 0, 0] = np.nan
    with pytest.raises(InvalidModelInputError):
        _small_model().fit(nan)
    zero_diag = np.array([[1.0, 0.0], [0.0, 0.0]])
    history = np.stack([valid] * (MIN_WINDOW_LENGTH - 1) + [zero_diag], axis=0)
    with pytest.raises(InvalidModelInputError, match="strictly positive"):
        _small_model().fit(history)
    with pytest.raises(InvalidModelInputError, match="fit before forecast"):
        _small_model().forecast()


def test_same_targets_features_dates_and_pair_order() -> None:
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
    index250 = har_regression_indices(250)
    np.testing.assert_array_equal(index250, np.arange(22, 250))
    assert index250.size == 228
    cube = _cube_from_variances(np.full((250, 2), 1.2), _constant_rho(2, 0.2))
    state = _tiny_model().fit(cube).fit_state
    assert state.n_regression_dates == 228
    assert state.lag_convention == LAG_CONVENTION
    assert state.pair_ordering == PAIR_ORDERING
    rows, cols = unique_pair_indices(2)
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(cols, [1])


def test_no_quarticity_cross_section_identity_log_or_fisher() -> None:
    cube = _random_spd_cube(40, 3, SEED + 1)
    model = _small_model().fit(cube)
    configuration = model.identity.configuration
    assert configuration["n_slope_columns"] == HAR_SLOPES
    assert configuration["slope_names"] == ("daily", "weekly", "monthly")
    assert configuration["responses_standardized"] is False
    assert configuration["within_convention"] == WITHIN_CONVENTION
    joined = " ".join(str(key).lower() for key in configuration)
    for token in (
        "quarticity",
        "cross_section",
        "group_id",
        "asset_id",
        "pair_id",
        "log_variance",
        "fisher",
        "shap",
        "ale",
    ):
        assert token not in joined
    source = inspect.getsource(XGBoostDRDRealizedCovariance.fit)
    assert "fisher" not in source.lower()
    assert "quarticity" not in source.lower()
    assert "group_id" not in source.lower()
    y, x = har_response_design(
        np.stack([np.diag(slice_) for slice_ in cube], axis=0),
        har_regression_indices(40),
    )
    assert np.all(y > 0.0)
    assert x.shape[-1] == 3


def test_within_means_and_residuals_match_independent_construction() -> None:
    cube = _random_spd_cube(40, 3, SEED + 2)
    model = _small_model().fit(cube)
    variances, correlations = _drd_panel(cube)
    pair_panel = pair_panel_from_correlations(correlations)
    response_index = har_regression_indices(40)
    y_d, x_d = har_response_design(variances, response_index)
    y_r, x_r = har_response_design(pair_panel, response_index)
    ybar_d, xbar_d, ytilde_d, x_tilde_d = within_transformed_arrays(y_d, x_d)
    ybar_r, xbar_r, ytilde_r, x_tilde_r = within_transformed_arrays(y_r, x_r)
    np.testing.assert_allclose(model.fit_state.ybar_D, y_d.mean(axis=1))
    np.testing.assert_allclose(model.fit_state.Xbar_D, x_d.mean(axis=1))
    np.testing.assert_allclose(model.fit_state.ybar_R, y_r.mean(axis=1))
    np.testing.assert_allclose(model.fit_state.Xbar_R, x_r.mean(axis=1))
    np.testing.assert_allclose(model.fit_state.ybar_D, ybar_d)
    np.testing.assert_allclose(model.fit_state.Xbar_D, xbar_d)
    np.testing.assert_allclose(ytilde_d, (y_d - ybar_d[:, None]).reshape(-1))
    np.testing.assert_allclose(
        x_tilde_d, (x_d - xbar_d[:, None, :]).reshape(ytilde_d.size, 3)
    )
    np.testing.assert_allclose(ytilde_r, (y_r - ybar_r[:, None]).reshape(-1))
    np.testing.assert_allclose(
        x_tilde_r, (x_r - xbar_r[:, None, :]).reshape(ytilde_r.size, 3)
    )


def test_scales_match_rms_zero_column_and_unscaled_response() -> None:
    cube = _random_spd_cube(40, 3, SEED + 3)
    model = _small_model().fit(cube)
    variances, correlations = _drd_panel(cube)
    pair_panel = pair_panel_from_correlations(correlations)
    y_d, x_d = har_response_design(variances, har_regression_indices(40))
    y_r, x_r = har_response_design(pair_panel, har_regression_indices(40))
    _ybar_d, _xbar_d, ytilde_d, x_tilde_d = within_transformed_arrays(y_d, x_d)
    _ybar_r, _xbar_r, ytilde_r, x_tilde_r = within_transformed_arrays(y_r, x_r)
    expected_d = np.sqrt(np.mean(x_tilde_d * x_tilde_d, axis=0))
    expected_r = np.sqrt(np.mean(x_tilde_r * x_tilde_r, axis=0))
    np.testing.assert_allclose(model.fit_state.scale_D, expected_d)
    np.testing.assert_allclose(model.fit_state.scale_R, expected_r)
    assert not np.allclose(ytilde_d, ytilde_d / np.sqrt(np.mean(ytilde_d * ytilde_d)))
    assert not np.allclose(ytilde_r, ytilde_r / np.sqrt(np.mean(ytilde_r * ytilde_r)))
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
    del y_tilde


def test_z_matches_ridge_scaled_within_on_same_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cube = _random_spd_cube(40, 3, SEED + 4)
    captured: dict[str, list[np.ndarray]] = {"Z": [], "y": []}
    original = fit_squared_error_booster

    def capture(predictors, response, **kwargs):
        captured["Z"].append(np.array(predictors, dtype=float, copy=True))
        captured["y"].append(np.array(response, dtype=float, copy=True))
        return original(predictors, response, **kwargs)

    monkeypatch.setattr(
        "covharness.models.xgboost_drd.fit_squared_error_booster", capture
    )
    xgb_model = _small_model().fit(cube)
    ridge = RidgeDRDRealizedCovariance(lambda_=1.0).fit(cube)
    variances, correlations = _drd_panel(cube)
    pair_panel = pair_panel_from_correlations(correlations)
    y_d, x_d = har_response_design(variances, har_regression_indices(40))
    y_r, x_r = har_response_design(pair_panel, har_regression_indices(40))
    _ybar_d, _xbar_d, ytilde_d, z_d, scale_d = scaled_within_design(y_d, x_d)
    _ybar_r, _xbar_r, ytilde_r, z_r, scale_r = scaled_within_design(y_r, x_r)
    np.testing.assert_allclose(xgb_model.fit_state.scale_D, ridge.fit_state.scale_D)
    np.testing.assert_allclose(xgb_model.fit_state.scale_R, ridge.fit_state.scale_R)
    np.testing.assert_allclose(xgb_model.fit_state.scale_D, scale_d)
    np.testing.assert_allclose(xgb_model.fit_state.scale_R, scale_r)
    assert len(captured["Z"]) == 2
    np.testing.assert_allclose(captured["Z"][0], z_d)
    np.testing.assert_allclose(captured["Z"][1], z_r)
    np.testing.assert_allclose(captured["y"][0], ytilde_d)
    np.testing.assert_allclose(captured["y"][1], ytilde_r)
    _ridge_ybar_d, _ridge_xbar_d, _ridge_ytilde_d, ridge_xtilde_d = (
        within_transformed_arrays(y_d, x_d)
    )
    ridge_z_d = scale_within_predictors(ridge_xtilde_d, ridge.fit_state.scale_D)
    np.testing.assert_allclose(captured["Z"][0], ridge_z_d)


def test_package_and_booster_configuration() -> None:
    assert xgboost.__version__ == PINNED_XGBOOST_VERSION
    assert metadata.version("xgboost") == PINNED_XGBOOST_VERSION
    license_text = metadata.metadata("xgboost").get("License") or metadata.metadata(
        "xgboost"
    ).get("License-Expression")
    assert license_text is not None
    assert "Apache" in license_text
    cube = _cube_from_variances(np.full((40, 2), 1.3), _constant_rho(2, 0.2))
    model = _small_model().fit(cube)
    state = model.fit_state
    assert state.xgboost_version == PINNED_XGBOOST_VERSION
    assert state.n_boosters == N_BOOSTERS
    boosters = (state.variance_booster, state.correlation_booster)
    assert len(boosters) == 2
    for booster in boosters:
        assert isinstance(booster, XGBRegressor)
        params = booster.get_params()
        assert params["objective"] == OBJECTIVE
        assert params["booster"] == BOOSTER_TYPE
        assert params["tree_method"] == TREE_METHOD
        assert params["device"] == DEVICE
        assert params["n_jobs"] == N_JOBS
        assert params["subsample"] == SUBSAMPLE
        assert params["colsample_bytree"] == COLSAMPLE_BYTREE
        assert params["colsample_bylevel"] == COLSAMPLE_BYLEVEL
        assert params["colsample_bynode"] == COLSAMPLE_BYNODE
        assert params["base_score"] == BASE_SCORE
        assert params["random_state"] == RANDOM_STATE
        assert params["grow_policy"] == GROW_POLICY
        assert params["max_bin"] == MAX_BIN
        assert params["validate_parameters"] is True
        assert params["early_stopping_rounds"] is None
        assert not hasattr(booster, "best_iteration")
        with pytest.raises(XGBoostError, match="eval_set"):
            booster.evals_result()
        assert booster.n_features_in_ == HAR_SLOPES
    assert EARLY_STOPPING is False
    source = inspect.getsource(fit_squared_error_booster)
    assert "eval_set" not in source
    assert "early_stopping_rounds=None" in source
    assert "device=DEVICE" in source


def test_repeated_identical_fit_is_deterministic() -> None:
    cube = _random_spd_cube(40, 3, SEED + 5)
    first = _small_model().fit(cube).forecast().matrix
    second = _small_model().fit(cube).forecast().matrix
    np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-12)
    third = _small_model().fit(cube.copy()).forecast().matrix
    np.testing.assert_allclose(first, third, rtol=0.0, atol=1e-12)


def test_boosted_map_is_not_a_single_affine_function() -> None:
    z0 = np.repeat(np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]), 8)
    design = np.column_stack([z0, np.zeros_like(z0), np.zeros_like(z0)])
    response = np.where(z0 > 0.0, 1.0, -1.0)
    booster = fit_squared_error_booster(
        design,
        response,
        n_estimators=20,
        max_depth=1,
        learning_rate=0.3,
        min_child_weight=0.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        gamma=0.0,
    )
    fitted = np.asarray(booster.predict(design), dtype=float)
    affine_design = np.column_stack([np.ones(z0.size), design])
    coef, *_ = np.linalg.lstsq(affine_design, fitted, rcond=None)
    affine = affine_design @ coef
    left = fitted[z0 < 0.0]
    right = fitted[z0 > 0.0]
    assert left.mean() < right.mean()
    piecewise = np.where(z0 > 0.0, right.mean(), left.mean())
    affine_mse = float(np.mean((fitted - affine) ** 2))
    piecewise_mse = float(np.mean((fitted - piecewise) ** 2))
    assert affine_mse > piecewise_mse
    assert affine_mse > 1e-4


def test_origin_forecast_uses_ybar_plus_booster_on_frozen_z() -> None:
    cube = _random_spd_cube(40, 3, SEED + 6)
    model = _small_model().fit(cube)
    state = model.fit_state
    z_d = origin_scaled_predictors(state.origin_X_D, state.Xbar_D, state.scale_D)
    z_r = origin_scaled_predictors(state.origin_X_R, state.Xbar_R, state.scale_R)
    expected_v = state.ybar_D + np.asarray(state.variance_booster.predict(z_d), dtype=float)
    expected_x = state.ybar_R + np.asarray(
        state.correlation_booster.predict(z_r), dtype=float
    )
    forecast = model.forecast()
    np.testing.assert_allclose(model._v_raw, expected_v)
    np.testing.assert_allclose(model._x_raw, expected_x)
    assert forecast.matrix.shape == (3, 3)


def test_update_window_freezes_maps_and_moves_origin_features() -> None:
    cube = _random_spd_cube(40, 3, SEED + 7)
    model = _small_model().fit(cube[:MIN_WINDOW_LENGTH])
    state = model.fit_state
    ybar_d = state.ybar_D.copy()
    xbar_d = state.Xbar_D.copy()
    scale_d = state.scale_D.copy()
    ybar_r = state.ybar_R.copy()
    xbar_r = state.Xbar_R.copy()
    scale_r = state.scale_R.copy()
    origin_d = state.origin_X_D.copy()
    origin_r = state.origin_X_R.copy()
    mean0 = state.window_mean.copy()
    bytes_d = booster_raw_bytes(state.variance_booster)
    bytes_r = booster_raw_bytes(state.correlation_booster)
    booster_d = state.variance_booster
    booster_r = state.correlation_booster
    n_estimators = state.n_estimators
    new_window = cube[4 : 4 + MIN_WINDOW_LENGTH]
    fit_calls = {"n": 0}
    original_fit = XGBRegressor.fit

    def counted(self, *args, **kwargs):
        fit_calls["n"] += 1
        return original_fit(self, *args, **kwargs)

    XGBRegressor.fit = counted  # type: ignore[method-assign]
    try:
        model.update_window(new_window)
    finally:
        XGBRegressor.fit = original_fit  # type: ignore[method-assign]
    assert fit_calls["n"] == 0
    updated = model.fit_state
    np.testing.assert_array_equal(updated.ybar_D, ybar_d)
    np.testing.assert_array_equal(updated.Xbar_D, xbar_d)
    np.testing.assert_array_equal(updated.scale_D, scale_d)
    np.testing.assert_array_equal(updated.ybar_R, ybar_r)
    np.testing.assert_array_equal(updated.Xbar_R, xbar_r)
    np.testing.assert_array_equal(updated.scale_R, scale_r)
    assert updated.variance_booster is booster_d
    assert updated.correlation_booster is booster_r
    assert booster_raw_bytes(updated.variance_booster) == bytes_d
    assert booster_raw_bytes(updated.correlation_booster) == bytes_r
    assert updated.n_estimators == n_estimators
    assert not np.allclose(updated.origin_X_D, origin_d)
    assert not np.allclose(updated.origin_X_R, origin_r)
    variances, correlations = _drd_panel(new_window)
    expected_v = forecast_predictors(variances).T
    expected_x = forecast_predictors(pair_panel_from_correlations(correlations)).T
    np.testing.assert_allclose(updated.origin_X_D, expected_v)
    np.testing.assert_allclose(updated.origin_X_R, expected_x)
    np.testing.assert_allclose(updated.window_mean, new_window.mean(axis=0))
    assert not np.allclose(updated.window_mean, mean0)


def test_valid_raw_forecast_is_not_repaired() -> None:
    cube = _cube_from_variances(np.full((40, 2), 1.3), _constant_rho(2, 0.2))
    model = _small_model().fit(cube)
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert validity.repaired is False
    assert validity.repair_method is None
    assert forecast.diagnostics.positive_definite is True


def test_nonpositive_variance_uses_window_mean_fallback() -> None:
    cube = _cube_from_variances(np.full((40, 2), 1.4), _constant_rho(2, 0.15))
    original = cube.copy()
    model = _small_model().fit(cube)
    state = model.fit_state
    object.__setattr__(state, "variance_booster", _ConstantPredictor(-state.ybar_D - 0.5))
    object.__setattr__(state, "correlation_booster", _ConstantPredictor(0.0))
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert validity.repaired is True
    assert validity.repair_method == REPAIR_METHOD
    assert validity.raw_nonpositive_variance is True
    np.testing.assert_allclose(forecast.matrix, original.mean(axis=0))
    np.testing.assert_array_equal(cube, original)


def test_out_of_bounds_correlation_uses_same_fallback() -> None:
    cube = _cube_from_variances(np.full((40, 2), 1.2), _constant_rho(2, 0.1))
    model = _small_model().fit(cube)
    state = model.fit_state
    object.__setattr__(state, "variance_booster", _ConstantPredictor(0.0))
    object.__setattr__(state, "correlation_booster", _ConstantPredictor(1.5 - state.ybar_R))
    forecast = model.forecast()
    validity = model.raw_validity
    assert validity is not None
    assert model._x_raw is not None
    assert np.any(np.abs(model._x_raw) > 1.0)
    assert validity.repaired is True
    assert validity.raw_correlation_out_of_bounds is True
    np.testing.assert_allclose(forecast.matrix, cube.mean(axis=0))


def test_in_bounds_non_pd_r_uses_same_fallback() -> None:
    cube = _cube_from_variances(np.full((40, 3), 1.1), _constant_rho(3, 0.05))
    model = _small_model().fit(cube)
    state = model.fit_state
    object.__setattr__(state, "variance_booster", _ConstantPredictor(0.0))
    bad_pairs = np.array([0.95, 0.95, -0.95]) - state.ybar_R
    object.__setattr__(state, "correlation_booster", _ConstantPredictor(bad_pairs))
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
    model = _small_model().fit(history)
    with pytest.raises(InvalidModelForecastError, match="No second repair"):
        model.forecast()


def test_no_clipping_nearest_pd_or_jitter() -> None:
    source = inspect.getsource(XGBoostDRDRealizedCovariance)
    lowered = source.lower()
    for token in ("clip", "nearest", "jitter", "higham", "cov_nearest", "eigval"):
        assert token not in lowered


def test_inputs_not_mutated_and_forecast_does_not_alias() -> None:
    cube = _cube_from_variances(np.full((40, 2), 1.4), _constant_rho(2, 0.1))
    original = cube.copy()
    model = _small_model().fit(cube)
    forecast = model.forecast()
    np.testing.assert_array_equal(cube, original)
    matrix = forecast.matrix
    matrix[0, 0] = -99.0
    second = model.forecast().matrix
    assert second[0, 0] != -99.0
    assert model.fit_state.window_mean[0, 0] != -99.0


def test_architecture_isolation_fixture_matches_har_and_ridge() -> None:
    cube = _random_spd_cube(40, 3, SEED + 8)
    har = HARDRDRealizedCovariance().fit(cube)
    ridge = RidgeDRDRealizedCovariance(lambda_=0.5).fit(cube)
    xgb = _small_model().fit(cube)
    assert har.fit_state.n_regression_dates == ridge.fit_state.n_regression_dates
    assert ridge.fit_state.n_regression_dates == xgb.fit_state.n_regression_dates
    assert har.fit_state.pair_ordering == xgb.fit_state.pair_ordering
    assert har.fit_state.lag_convention == xgb.fit_state.lag_convention
    np.testing.assert_allclose(har.fit_state.window_mean, xgb.fit_state.window_mean)
    np.testing.assert_allclose(ridge.fit_state.window_mean, xgb.fit_state.window_mean)
    np.testing.assert_allclose(ridge.fit_state.scale_D, xgb.fit_state.scale_D)
    np.testing.assert_allclose(ridge.fit_state.scale_R, xgb.fit_state.scale_R)
    variances, correlations = _drd_panel(cube)
    pair_panel = pair_panel_from_correlations(correlations)
    y_d, x_d = har_response_design(variances, har_regression_indices(40))
    y_r, x_r = har_response_design(pair_panel, har_regression_indices(40))
    np.testing.assert_allclose(xgb.fit_state.ybar_D, y_d.mean(axis=1))
    np.testing.assert_allclose(xgb.fit_state.ybar_R, y_r.mean(axis=1))
    origin_v = forecast_predictors(variances).T
    origin_x = forecast_predictors(pair_panel).T
    np.testing.assert_allclose(xgb.fit_state.origin_X_D, origin_v)
    np.testing.assert_allclose(xgb.fit_state.origin_X_R, origin_x)
    assert xgb.fit_state.scaling_convention == SCALING_CONVENTION
    assert RESPONSE_START == 22
    har.forecast()
    xgb.forecast()


def test_pooled_boosters_not_per_group_at_moderate_n() -> None:
    cube = _random_spd_cube(30, 25, SEED + 9)
    model = _tiny_model().fit(cube)
    state = model.fit_state
    assert state.n_assets == 25
    assert state.n_pairs == 25 * 24 // 2
    assert state.n_boosters == 2
    forecast = model.forecast()
    assert forecast.matrix.shape == (25, 25)
    shifted = cube[1:31]
    bytes_d = booster_raw_bytes(state.variance_booster)
    model.update_window(shifted)
    assert booster_raw_bytes(model.fit_state.variance_booster) == bytes_d
    updated = model.forecast()
    assert updated.matrix.shape == (25, 25)


def test_window_state_capability_without_class_name_dispatch() -> None:
    caps = capabilities_of(_small_model())
    assert caps.rolling_cadence is RollingCadence.WINDOW_STATE
    assert caps.fit_input.name == "REALIZED_COVARIANCE"
    assert caps.update_observable.name == "REALIZED_COVARIANCE_WINDOW"
    source = inspect.getsource(capabilities_of)
    assert "xgboost" not in source.lower()
    from covharness.protocol import runner as runner_module

    runner_source = inspect.getsource(runner_module._advance_at_origin)
    assert "XGBoost" not in runner_source
    assert "RidgeDRD" not in runner_source
