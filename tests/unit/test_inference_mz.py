"""Pooled-vech Mincer-Zarnowitz tests on synthetic covariance panels."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.inference import (
    INFERENCE_DETERMINISTIC_ALTERNATIVE,
    INFERENCE_DETERMINISTIC_NULL,
    INFERENCE_REGULAR,
    MZ_ALPHA,
    WEIGHTING_APPROXIMATE_PS21,
    WEIGHTING_NONE,
    RankDeficientMZDesignError,
    mincer_zarnowitz_pooled_vech,
    mincer_zarnowitz_two_finalists,
)
from covharness.inference.mz import (
    _design_and_response,
    hac_long_run_covariance,
    unique_entry_indices,
)


def _spd_panel(rng: np.random.Generator, n_obs: int, n_assets: int) -> np.ndarray:
    noise = rng.normal(size=(n_obs, n_assets, n_assets))
    return noise @ np.transpose(noise, (0, 2, 1)) + n_assets * np.eye(n_assets)


def test_mz_perfect_calibration() -> None:
    rng = np.random.default_rng(11)
    forecast = _spd_panel(rng, 40, 3)
    result = mincer_zarnowitz_pooled_vech(forecast, forecast)
    assert result.alpha == pytest.approx(0.0, abs=1e-10)
    assert result.beta == pytest.approx(1.0, abs=1e-10)
    assert result.p_value == pytest.approx(1.0)
    assert result.wald_statistic == pytest.approx(0.0)
    assert result.wald_df == 2
    assert result.weighting_rule == WEIGHTING_NONE
    assert result.alpha_level == MZ_ALPHA
    assert result.covariance is None
    assert result.covariance_degenerate is True
    assert result.inference_case == INFERENCE_DETERMINISTIC_NULL


def test_mz_intercept_bias() -> None:
    rng = np.random.default_rng(12)
    forecast = _spd_panel(rng, 50, 3)
    proxy = forecast + 0.4
    result = mincer_zarnowitz_pooled_vech(proxy, forecast)
    assert result.alpha == pytest.approx(0.4, abs=1e-8)
    assert result.beta == pytest.approx(1.0, abs=1e-8)
    assert result.p_value < 0.01


def test_mz_slope_bias() -> None:
    rng = np.random.default_rng(13)
    forecast = _spd_panel(rng, 50, 3)
    proxy = 1.6 * forecast
    result = mincer_zarnowitz_pooled_vech(proxy, forecast)
    assert result.alpha == pytest.approx(0.0, abs=1e-8)
    assert result.beta == pytest.approx(1.6, abs=1e-8)
    assert result.p_value < 0.01


def test_mz_unbiased_noisy_proxy_does_not_systematically_bias() -> None:
    rng = np.random.default_rng(14)
    forecast = _spd_panel(rng, 80, 3)
    noise = rng.normal(scale=0.05, size=forecast.shape)
    noise = 0.5 * (noise + np.transpose(noise, (0, 2, 1)))
    proxy = forecast + noise
    result = mincer_zarnowitz_pooled_vech(proxy, forecast)
    assert result.alpha == pytest.approx(0.0, abs=0.03)
    assert result.beta == pytest.approx(1.0, abs=0.03)
    assert result.p_value > 0.01


def test_mz_low_dimensional_hand_calculation() -> None:
    a_s = np.array([[2.0, 1.0], [1.0, 3.0]])
    b_s = np.array([[4.0, 0.0], [0.0, 5.0]])
    a_h = np.array([[1.0, 0.5], [0.5, 1.0]])
    b_h = np.array([[2.0, 1.5], [1.5, 2.0]])
    scales = [1.0, 1.0, 1.1, 1.1, 0.9, 0.9]
    proxy = np.stack([a_s * s if i % 2 == 0 else b_s * s for i, s in enumerate(scales)])
    forecast = np.stack([a_h * s if i % 2 == 0 else b_h * s for i, s in enumerate(scales)])
    y = []
    x = []
    for time in range(6):
        s = proxy[time]
        h = forecast[time]
        y.extend([s[0, 0], s[0, 1], s[1, 1]])
        x.extend([h[0, 0], h[0, 1], h[1, 1]])
    design = np.column_stack([np.ones(18), np.array(x)])
    coef_hand = np.linalg.lstsq(design, np.array(y), rcond=None)[0]
    result = mincer_zarnowitz_pooled_vech(proxy, forecast)
    assert result.alpha == pytest.approx(coef_hand[0])
    assert result.beta == pytest.approx(coef_hand[1])
    assert result.n_unique_entries == 3
    assert result.wald_df == 2
    assert result.p_value < 1.0


def test_mz_rank_deficient_forecast_design() -> None:
    forecast = np.zeros((8, 2, 2))
    proxy = forecast + 1.0
    with pytest.raises(RankDeficientMZDesignError, match="rank deficient"):
        mincer_zarnowitz_pooled_vech(proxy, forecast)


def test_mz_joint_wald_has_two_degrees_of_freedom() -> None:
    rng = np.random.default_rng(15)
    forecast = _spd_panel(rng, 30, 2)
    proxy = 0.2 + 0.7 * forecast
    result = mincer_zarnowitz_pooled_vech(proxy, forecast)
    assert result.wald_df == 2
    assert result.wald_statistic > 0.0
    assert 0.0 <= result.p_value <= 1.0


def test_mz_daily_score_sums_unique_entries() -> None:
    rng = np.random.default_rng(16)
    forecast = _spd_panel(rng, 12, 2)
    proxy = forecast + 0.1
    result = mincer_zarnowitz_pooled_vech(proxy, forecast)
    rows, cols = unique_entry_indices(2)
    design, response = _design_and_response(
        proxy, forecast, rows, cols, None, WEIGHTING_NONE
    )
    theta = np.array([result.alpha, result.beta])
    scores = np.empty((12, 2))
    for time in range(12):
        resid = response[time] - design[time] @ theta
        scores[time] = design[time].T @ resid
        assert scores[time].shape == (2,)
        assert resid.shape[0] == 3
    np.testing.assert_allclose(scores.sum(axis=0), 0.0, atol=1e-10)


def test_mz_sandwich_matches_date_clustered_hac_formula() -> None:
    rng = np.random.default_rng(24)
    forecast = _spd_panel(rng, 20, 2)
    noise = rng.normal(scale=0.1, size=forecast.shape)
    noise = 0.5 * (noise + np.transpose(noise, (0, 2, 1)))
    proxy = forecast + noise
    result = mincer_zarnowitz_pooled_vech(proxy, forecast)
    assert result.inference_case == INFERENCE_REGULAR
    assert result.covariance is not None
    rows, cols = unique_entry_indices(2)
    design, response = _design_and_response(
        proxy, forecast, rows, cols, None, WEIGHTING_NONE
    )
    n_obs, _n_entries, n_coef = design.shape
    gram = np.zeros((n_coef, n_coef))
    xy = np.zeros(n_coef)
    for time in range(n_obs):
        gram = gram + design[time].T @ design[time]
        xy = xy + design[time].T @ response[time]
    theta = np.linalg.solve(gram, xy)
    scores = np.empty((n_obs, n_coef))
    for time in range(n_obs):
        resid = response[time] - design[time] @ theta
        scores[time] = design[time].T @ resid
    omega, _lag = hac_long_run_covariance(scores)
    covariance_hand = np.linalg.solve(
        gram, np.linalg.solve(gram, n_obs * omega).T
    ).T
    np.testing.assert_allclose(result.covariance, covariance_hand)

    stacked_x = design.reshape(n_obs * 3, n_coef)
    stacked_y = response.reshape(n_obs * 3)
    stacked_resid = stacked_y - stacked_x @ theta
    sigma2 = float(np.sum(stacked_resid**2) / (n_obs * 3 - n_coef))
    iid_covariance = sigma2 * np.linalg.inv(gram)
    assert not np.allclose(result.covariance, iid_covariance)


def test_mz_deterministic_null_and_alternatives_omit_singular_covariance() -> None:
    rng = np.random.default_rng(25)
    forecast = _spd_panel(rng, 12, 2)
    calibrated = mincer_zarnowitz_pooled_vech(forecast, forecast)
    intercept = mincer_zarnowitz_pooled_vech(forecast + 0.4, forecast)
    slope = mincer_zarnowitz_pooled_vech(1.6 * forecast, forecast)
    assert calibrated.inference_case == INFERENCE_DETERMINISTIC_NULL
    assert calibrated.wald_statistic == pytest.approx(0.0)
    assert calibrated.p_value == pytest.approx(1.0)
    assert calibrated.covariance is None
    assert intercept.inference_case == INFERENCE_DETERMINISTIC_ALTERNATIVE
    assert intercept.wald_statistic == np.inf
    assert intercept.p_value == pytest.approx(0.0)
    assert intercept.covariance is None
    assert intercept.covariance_degenerate is True
    assert slope.inference_case == INFERENCE_DETERMINISTIC_ALTERNATIVE
    assert slope.wald_statistic == np.inf
    assert slope.p_value == pytest.approx(0.0)
    assert slope.covariance is None


def test_mz_regular_inference_returns_numeric_covariance() -> None:
    rng = np.random.default_rng(26)
    forecast = _spd_panel(rng, 16, 2)
    noise = rng.normal(scale=0.2, size=forecast.shape)
    noise = 0.5 * (noise + np.transpose(noise, (0, 2, 1)))
    result = mincer_zarnowitz_pooled_vech(forecast + noise, forecast)
    assert result.inference_case == INFERENCE_REGULAR
    assert result.covariance_degenerate is False
    assert result.covariance is not None
    assert result.covariance.shape == (2, 2)
    assert np.linalg.matrix_rank(result.covariance) == 2


def test_mz_serial_hac_uses_positive_lags_on_persistent_scores() -> None:
    rng = np.random.default_rng(17)
    n_obs = 80
    forecast = _spd_panel(rng, n_obs, 2)
    shock = rng.normal(size=n_obs)
    for t in range(1, n_obs):
        shock[t] = 0.8 * shock[t - 1] + 0.2 * shock[t]
    proxy = forecast + shock[:, None, None]
    default = mincer_zarnowitz_pooled_vech(proxy, forecast)
    hetero_only = mincer_zarnowitz_pooled_vech(proxy, forecast, maxlags=0)
    assert default.hac_lags > 0
    assert hetero_only.hac_lags == 0
    assert default.covariance[0, 0] != pytest.approx(hetero_only.covariance[0, 0])


def test_mz_does_not_mutate_inputs() -> None:
    rng = np.random.default_rng(18)
    forecast = _spd_panel(rng, 10, 2)
    proxy = forecast + 0.05
    proxy_copy = proxy.copy()
    forecast_copy = forecast.copy()
    mincer_zarnowitz_pooled_vech(proxy, forecast)
    np.testing.assert_array_equal(proxy, proxy_copy)
    np.testing.assert_array_equal(forecast, forecast_copy)


def test_augmented_mz_detects_state_dependent_miscalibration() -> None:
    rng = np.random.default_rng(19)
    n_obs = 160
    forecast = _spd_panel(rng, n_obs, 3)
    origin = _spd_panel(rng, n_obs, 3)
    from covharness.inference.gw import market_state_augmenting_instruments

    state = market_state_augmenting_instruments(origin)
    shift = state[:, 0] - state[:, 0].mean()
    proxy = forecast + 1.2 * shift[:, None, None]
    standard = mincer_zarnowitz_pooled_vech(proxy, forecast)
    augmented = mincer_zarnowitz_pooled_vech(
        proxy, forecast, origin_covariances=origin
    )
    assert standard.p_value > 0.05
    assert augmented.wald_df == 4
    assert augmented.p_value < 0.01
    assert augmented.gamma.shape == (2,)


def test_mz_two_finalists_bonferroni_family() -> None:
    rng = np.random.default_rng(20)
    forecast_a = _spd_panel(rng, 25, 2)
    forecast_b = _spd_panel(rng, 25, 2)
    result_a = mincer_zarnowitz_pooled_vech(forecast_a, forecast_a)
    result_b = mincer_zarnowitz_pooled_vech(forecast_b + 0.5, forecast_b)
    family = mincer_zarnowitz_two_finalists(result_a, result_b)
    assert family.family_size == 2
    assert family.per_test_cutoff == pytest.approx(0.025)
    assert family.p_value_a_bonferroni == pytest.approx(min(1.0, 2.0 * result_a.p_value))
    assert family.p_value_b_bonferroni == pytest.approx(min(1.0, 2.0 * result_b.p_value))


def test_ps21_scale_uses_square_root_not_product() -> None:
    base_forecast = np.array([[4.0, 2.0], [2.0, 9.0]])
    base_proxy = np.array([[5.0, 3.0], [3.0, 10.0]])
    multipliers = np.array([1.0, 1.1, 0.8, 1.3])
    forecast = np.stack([base_forecast * m for m in multipliers])
    proxy = np.stack([base_proxy * m for m in multipliers])
    y_list = []
    x1_list = []
    x2_list = []
    y_wrong = []
    x1_wrong = []
    x2_wrong = []
    for time in range(4):
        h = forecast[time]
        s = proxy[time]
        scale = np.array(
            [
                np.sqrt(h[0, 0] * h[0, 0]),
                np.sqrt(h[0, 0] * h[1, 1]),
                np.sqrt(h[1, 1] * h[1, 1]),
            ]
        )
        unique_s = np.array([s[0, 0], s[0, 1], s[1, 1]])
        unique_h = np.array([h[0, 0], h[0, 1], h[1, 1]])
        y_list.append(unique_s / scale)
        x1_list.append(1.0 / scale)
        x2_list.append(unique_h / scale)
        wrong = np.array([h[0, 0] * h[0, 0], h[0, 0] * h[1, 1], h[1, 1] * h[1, 1]])
        y_wrong.append(unique_s / wrong)
        x1_wrong.append(1.0 / wrong)
        x2_wrong.append(unique_h / wrong)
    design = np.column_stack([np.concatenate(x1_list), np.concatenate(x2_list)])
    response = np.concatenate(y_list)
    coef = np.linalg.lstsq(design, response, rcond=None)[0]
    result = mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_APPROXIMATE_PS21
    )
    assert result.alpha == pytest.approx(coef[0])
    assert result.beta == pytest.approx(coef[1])
    coef_wrong = np.linalg.lstsq(
        np.column_stack([np.concatenate(x1_wrong), np.concatenate(x2_wrong)]),
        np.concatenate(y_wrong),
        rcond=None,
    )[0]
    assert result.alpha != pytest.approx(coef_wrong[0])
    assert result.weighting_rule == WEIGHTING_APPROXIMATE_PS21
    assert result.weighting_rule != "exact_gls"


def test_ps21_diagonal_reduces_to_division_by_h_ii() -> None:
    base = np.array([[4.0, 0.0], [0.0, 9.0]])
    forecast = np.stack([base * m for m in (1.0, 1.2, 0.7, 1.4, 0.9, 1.1)])
    proxy = forecast.copy()
    result = mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_APPROXIMATE_PS21
    )
    rows, cols = unique_entry_indices(2)
    design, response = _design_and_response(
        proxy, forecast, rows, cols, None, WEIGHTING_APPROXIMATE_PS21
    )
    np.testing.assert_allclose(design[0, 0], [1.0 / 4.0, 1.0])
    np.testing.assert_allclose(design[0, 2], [1.0 / 9.0, 1.0])
    np.testing.assert_allclose(response[0, 0], 1.0)
    assert result.diagonal.beta == pytest.approx(1.0, abs=1e-10)


def test_ps21_transformed_forecast_equals_correlation() -> None:
    base = np.array([[4.0, 2.0], [2.0, 9.0]])
    forecast = np.repeat(base[None, :, :], 5, axis=0)
    proxy = forecast.copy()
    rows, cols = unique_entry_indices(2)
    design, _response = _design_and_response(
        proxy, forecast, rows, cols, None, WEIGHTING_APPROXIMATE_PS21
    )
    off_diag = 1
    assert rows[off_diag] == 0 and cols[off_diag] == 1
    assert design[0, off_diag, 1] == pytest.approx(2.0 / 6.0)


def test_ps21_same_scale_on_y_and_all_x_including_state() -> None:
    base = np.array([[4.0, 2.0], [2.0, 9.0]])
    forecast = np.repeat(base[None, :, :], 8, axis=0)
    proxy = forecast + 0.1
    instruments = np.column_stack(
        [np.linspace(0.1, 0.8, 8), np.linspace(-0.2, 0.3, 8)]
    )
    rows, cols = unique_entry_indices(2)
    design, response = _design_and_response(
        proxy, forecast, rows, cols, instruments, WEIGHTING_APPROXIMATE_PS21
    )
    scales = np.array([4.0, 6.0, 9.0])
    np.testing.assert_allclose(response[3] * scales, proxy[3, rows, cols])
    np.testing.assert_allclose(design[3, :, 0] * scales, 1.0)
    np.testing.assert_allclose(design[3, :, 1] * scales, forecast[3, rows, cols])
    np.testing.assert_allclose(design[3, :, 2] * scales, instruments[3, 0])
    np.testing.assert_allclose(design[3, :, 3] * scales, instruments[3, 1])


def test_ps21_nonpositive_scale_is_rejected() -> None:
    forecast = np.zeros((6, 2, 2))
    forecast[:, 0, 0] = 1.0
    forecast[:, 1, 1] = 0.0
    proxy = np.ones((6, 2, 2))
    with pytest.raises(Exception, match="strictly positive"):
        mincer_zarnowitz_pooled_vech(
            proxy, forecast, weighting=WEIGHTING_APPROXIMATE_PS21
        )


def test_ps21_never_labeled_exact_gls() -> None:
    rng = np.random.default_rng(21)
    forecast = _spd_panel(rng, 12, 2)
    result = mincer_zarnowitz_pooled_vech(
        forecast, forecast, weighting=WEIGHTING_APPROXIMATE_PS21
    )
    assert result.weighting_rule == "approximate_ps21"
    assert "exact" not in result.weighting_rule


def test_ps21_heteroskedastic_case_remains_finite() -> None:
    rng = np.random.default_rng(22)
    n_obs = 60
    forecast = _spd_panel(rng, n_obs, 2)
    scale = np.sqrt(np.diagonal(forecast, axis1=1, axis2=2))
    noise = rng.normal(size=forecast.shape) * scale[:, :, None]
    noise = 0.5 * (noise + np.transpose(noise, (0, 2, 1)))
    proxy = forecast + 0.2 * noise
    ols = mincer_zarnowitz_pooled_vech(proxy, forecast)
    wls = mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_APPROXIMATE_PS21
    )
    assert np.isfinite(wls.alpha)
    assert np.isfinite(wls.beta)
    assert wls.covariance[0, 0] > 0.0
    assert ols.weighting_rule == WEIGHTING_NONE
    assert wls.weighting_rule == WEIGHTING_APPROXIMATE_PS21


def test_ps21_weighted_fit_uses_positive_lag_daily_hac() -> None:
    rng = np.random.default_rng(27)
    n_obs = 80
    forecast = _spd_panel(rng, n_obs, 2)
    shock = rng.normal(size=n_obs)
    for t in range(1, n_obs):
        shock[t] = 0.8 * shock[t - 1] + 0.2 * shock[t]
    proxy = forecast + shock[:, None, None]
    default = mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_APPROXIMATE_PS21
    )
    hetero_only = mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_APPROXIMATE_PS21, maxlags=0
    )
    assert default.inference_case == INFERENCE_REGULAR
    assert default.hac_lags > 0
    assert hetero_only.hac_lags == 0
    assert default.covariance is not None
    assert hetero_only.covariance is not None
    assert default.covariance[0, 0] != pytest.approx(hetero_only.covariance[0, 0])


def test_ps21_does_not_mutate_inputs() -> None:
    rng = np.random.default_rng(23)
    forecast = _spd_panel(rng, 8, 2)
    proxy = forecast.copy()
    proxy_copy = proxy.copy()
    forecast_copy = forecast.copy()
    mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_APPROXIMATE_PS21
    )
    np.testing.assert_array_equal(proxy, proxy_copy)
    np.testing.assert_array_equal(forecast, forecast_copy)
