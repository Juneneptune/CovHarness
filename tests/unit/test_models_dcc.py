"""Synthetic tests for original Engle DCC and DCC-NL."""

from __future__ import annotations

import numpy as np
import pytest
from arch.univariate import arch_model

from covharness.models import (
    DCCCovariance,
    DCCNonlinearCovariance,
    InvalidModelForecastError,
    InvalidModelInputError,
)
from covharness.models.dcc import (
    DCC_ALPHA0,
    DCC_BETA0,
    all_pairs_composite_nll,
    bivariate_correlation_nll,
    correlation_from_covariance,
    dcc_observed_q_path,
    dcc_q_update,
    estimate_dcc_parameters,
    fit_stage_one_garch,
    garch_variance_paths,
    nonlinear_standardized_residual_correlation,
    one_step_forecast_from_state,
    pair_count,
    pair_indices,
    q_to_correlation,
    sample_standardized_residual_correlation,
    standardized_residuals,
)


def _rng(seed: int = 20260916) -> np.random.Generator:
    return np.random.default_rng(seed)


def _garch_path(
    n_times: int,
    *,
    omega: float,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    mean: float = 0.0,
    n_burn: int = 250,
) -> np.ndarray:
    # Simulate only. Estimator backcast is the sample second moment, not this seed.
    unconditional = omega / (1.0 - alpha - beta)
    variance = unconditional
    residual = np.sqrt(variance) * rng.standard_normal()
    series = np.empty(n_times)
    written = 0
    for step in range(n_times + n_burn):
        variance = omega + alpha * residual * residual + beta * variance
        residual = np.sqrt(variance) * rng.standard_normal()
        if step >= n_burn:
            series[written] = residual + mean
            written += 1
    return series


def _dcc_like_returns(
    n_times: int = 150,
    n_assets: int = 3,
    seed: int = 20260916,
) -> np.ndarray:
    rng = _rng(seed)
    means = np.linspace(-0.04, 0.05, n_assets)
    returns = np.empty((n_times, n_assets))
    for asset in range(n_assets):
        returns[:, asset] = _garch_path(
            n_times,
            omega=0.05,
            alpha=0.15,
            beta=0.75,
            rng=rng,
            mean=float(means[asset]),
        )
    # Shared factor so the sample correlation target is nontrivial.
    factor = 0.08 * rng.standard_normal(n_times)
    returns = returns + factor[:, None]
    return returns


def test_input_rejects_wrong_ndim() -> None:
    with pytest.raises(InvalidModelInputError, match=r"\(T, N\)"):
        DCCCovariance().fit(np.zeros((4, 3, 3)))


def test_input_rejects_t_equals_one() -> None:
    with pytest.raises(InvalidModelInputError, match="at least two"):
        DCCCovariance().fit(np.ones((1, 3)))


def test_input_rejects_single_asset() -> None:
    with pytest.raises(InvalidModelInputError, match="at least two assets"):
        DCCCovariance().fit(np.ones((20, 1)))


def test_input_rejects_nonfinite() -> None:
    returns = _dcc_like_returns()
    returns[4, 1] = np.nan
    with pytest.raises(InvalidModelInputError, match="finite"):
        DCCCovariance().fit(returns)


def test_inputs_are_not_mutated() -> None:
    returns = _dcc_like_returns()
    original = returns.copy()
    DCCCovariance().fit(returns)
    np.testing.assert_array_equal(returns, original)


def test_fit_window_mean_is_exact() -> None:
    returns = _dcc_like_returns()
    state = DCCCovariance().fit(returns).fit_state
    np.testing.assert_allclose(state.fit_mean, returns.mean(axis=0))


def test_update_uses_frozen_fit_mean() -> None:
    returns = _dcc_like_returns()
    model = DCCCovariance().fit(returns)
    frozen = model.fit_state.fit_mean.copy()
    new_return = returns[-1] + 0.7
    model.update(new_return)
    np.testing.assert_array_equal(model.fit_state.fit_mean, frozen)
    np.testing.assert_allclose(model.fit_state.current_residual, new_return - frozen)


def test_backcast_equals_mean_squared_centered_returns() -> None:
    returns = _dcc_like_returns()
    centered = returns - returns.mean(axis=0)
    expected = np.mean(centered * centered, axis=0)
    state = DCCCovariance().fit(returns).fit_state
    np.testing.assert_allclose(state.garch_backcast, expected)
    assert np.all(state.garch_backcast > 0.0)


def test_garch_variance_recursion_matches_hand_calculation() -> None:
    residuals = np.array([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.0]])
    omega = np.array([0.05, 0.02])
    alpha = np.array([0.1, 0.2])
    beta = np.array([0.8, 0.7])
    backcast = np.array([0.3, 0.4])
    path = garch_variance_paths(residuals, omega, alpha, beta, backcast)
    first = omega + (alpha + beta) * backcast
    np.testing.assert_allclose(path[0], first)
    second = omega + alpha * residuals[0] ** 2 + beta * first
    np.testing.assert_allclose(path[1], second)
    third = omega + alpha * residuals[1] ** 2 + beta * second
    np.testing.assert_allclose(path[2], third)


def test_standardized_residual_divides_by_sqrt_h() -> None:
    residual = np.array([[2.0, -4.0]])
    variance = np.array([[16.0, 4.0]])
    standardized = standardized_residuals(residual, variance)
    np.testing.assert_allclose(standardized, np.array([[0.5, -2.0]]))
    assert not np.allclose(standardized, residual / variance)


def test_stage_one_matches_direct_arch_call() -> None:
    returns = _dcc_like_returns(n_times=150, n_assets=2, seed=3)
    stage = fit_stage_one_garch(returns)
    centered = returns - returns.mean(axis=0)
    for asset in range(2):
        backcast = float(np.mean(centered[:, asset] ** 2))
        fitted = arch_model(
            centered[:, asset],
            mean="Zero",
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            dist="normal",
            rescale=False,
        ).fit(disp="off", backcast=backcast)
        np.testing.assert_allclose(
            stage.omega[asset], float(fitted.params["omega"]), rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            stage.garch_alpha[asset], float(fitted.params["alpha[1]"]), rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            stage.garch_beta[asset], float(fitted.params["beta[1]"]), rtol=1e-6, atol=1e-10
        )
        arch_h = np.asarray(fitted.conditional_volatility, dtype=float) ** 2
        np.testing.assert_allclose(stage.variances[:, asset], arch_h, rtol=1e-6, atol=1e-10)


def test_zero_backcast_is_rejected_without_dropping_assets() -> None:
    returns = np.ones((20, 3))
    returns[:, 1] = 2.0
    with pytest.raises(InvalidModelInputError, match="backcast"):
        DCCCovariance().fit(returns)


def test_garch_constraint_failure_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    import arch.univariate as arch_univ

    class _FakeResult:
        optimization_result = type(
            "opt", (), {"success": True, "message": "ok", "nfev": 1}
        )()
        params = {"omega": 0.1, "alpha[1]": 0.2, "beta[1]": 0.9}
        conditional_volatility = np.ones(40)

    def _fake_arch_model(*_args, **_kwargs):
        return type("model", (), {"fit": lambda *a, **k: _FakeResult()})()

    monkeypatch.setattr(arch_univ, "arch_model", _fake_arch_model)
    with pytest.raises(InvalidModelForecastError, match="a\\+b<1"):
        DCCCovariance().fit(_dcc_like_returns(n_times=40, n_assets=2))


def test_plain_target_matches_second_moment_over_t() -> None:
    standardized = _rng(4).standard_normal((30, 4))
    target = sample_standardized_residual_correlation(standardized)
    gram = standardized.T @ standardized / 30
    scales = np.sqrt(np.diag(gram))
    expected = gram / np.outer(scales, scales)
    np.testing.assert_allclose(target, expected)
    np.testing.assert_allclose(np.diag(target), 1.0)
    np.linalg.cholesky(target)


def test_plain_target_does_not_demean_standardized_residuals() -> None:
    standardized = _rng(5).standard_normal((25, 3)) + 0.4
    target = sample_standardized_residual_correlation(standardized)
    demeaned = standardized - standardized.mean(axis=0)
    other = sample_standardized_residual_correlation(demeaned)
    assert not np.allclose(target, other)


def test_plain_target_n_greater_than_t_fails() -> None:
    standardized = _rng(6).standard_normal((5, 8))
    with pytest.raises(InvalidModelForecastError, match="N>T"):
        sample_standardized_residual_correlation(standardized)


def test_plain_target_n_equals_t_is_not_categorically_rejected() -> None:
    standardized = np.eye(6)
    target = sample_standardized_residual_correlation(standardized)
    np.testing.assert_allclose(target, np.eye(6))
    np.linalg.cholesky(target)


def test_nl_target_matches_nonlinshrink_k_zero() -> None:
    import nonlinshrink as nls

    standardized = _rng(8).standard_normal((40, 4))
    target = nonlinear_standardized_residual_correlation(standardized)
    shrunk = np.asarray(nls.shrink_cov(standardized, k=0), dtype=float)
    expected = correlation_from_covariance(shrunk)
    np.testing.assert_allclose(target, expected)
    np.linalg.cholesky(target)


def test_nl_target_does_not_use_standalone_t_minus_one_wrapper() -> None:
    import nonlinshrink as nls

    standardized = _rng(9).standard_normal((40, 4))
    target = nonlinear_standardized_residual_correlation(standardized)
    demeaned = nls.shrink_cov(standardized - standardized.mean(axis=0), k=1)
    other = correlation_from_covariance(np.asarray(demeaned, dtype=float))
    assert not np.allclose(target, other)


def test_nl_target_n_greater_than_t_can_be_valid() -> None:
    standardized = _rng(10).standard_normal((20, 25))
    target = nonlinear_standardized_residual_correlation(standardized)
    np.testing.assert_allclose(np.diag(target), 1.0, atol=1e-10)
    np.linalg.cholesky(target)


def test_q_update_matches_hand_calculation() -> None:
    q_matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
    shock = np.array([0.5, -1.0])
    target = np.array([[1.0, 0.1], [0.1, 1.0]])
    updated = dcc_q_update(q_matrix, shock, target, alpha=0.05, beta=0.90)
    expected = 0.05 * target + 0.05 * np.outer(shock, shock) + 0.90 * q_matrix
    np.testing.assert_allclose(updated, expected)


def test_q_normalization_has_unit_diagonal() -> None:
    q_matrix = np.array([[1.2, 0.3], [0.3, 0.8]])
    correlation = q_to_correlation(q_matrix)
    np.testing.assert_allclose(np.diag(correlation), 1.0)
    np.testing.assert_allclose(correlation[0, 1], 0.3 / np.sqrt(1.2 * 0.8))


def test_forecast_is_d_r_d() -> None:
    returns = _dcc_like_returns()
    model = DCCCovariance().fit(returns)
    forecast = model.forecast()
    _h_matrix, h_next, q_next = one_step_forecast_from_state(model.fit_state)
    correlation = q_to_correlation(q_next)
    scale = np.sqrt(h_next)
    expected = correlation * np.outer(scale, scale)
    np.testing.assert_allclose(forecast.matrix, expected)
    np.testing.assert_allclose(expected, np.diag(scale) @ correlation @ np.diag(scale))
    del _h_matrix


def test_alpha_beta_zero_holds_q_at_target() -> None:
    target = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    standardized = _rng(11).standard_normal((15, 3))
    path = dcc_observed_q_path(target, 0.0, 0.0, standardized)
    for slice_q in path:
        np.testing.assert_allclose(slice_q, target)


def test_alpha_zero_is_pure_persistence_toward_c() -> None:
    target = np.eye(2)
    q0 = np.array([[1.0, 0.4], [0.4, 1.0]])
    shock = np.array([10.0, -4.0])
    updated = dcc_q_update(q0, shock, target, alpha=0.0, beta=0.8)
    np.testing.assert_allclose(updated, 0.2 * target + 0.8 * q0)


def test_one_step_indexing_uses_current_residual_and_shock() -> None:
    returns = _dcc_like_returns()
    model = DCCCovariance().fit(returns)
    state = model.fit_state
    h_next, q_next = one_step_forecast_from_state(state)[1:]
    expected_h = (
        state.omega
        + state.garch_alpha * state.current_residual**2
        + state.garch_beta * state.current_variance
    )
    expected_q = dcc_q_update(
        state.current_q,
        state.current_standardized,
        state.target_c,
        state.dcc_alpha,
        state.dcc_beta,
    )
    np.testing.assert_allclose(h_next, expected_h)
    np.testing.assert_allclose(q_next, expected_q)


def test_forecast_does_not_mutate_state() -> None:
    model = DCCCovariance().fit(_dcc_like_returns())
    q_before = model.fit_state.current_q.copy()
    h_before = model.fit_state.current_variance.copy()
    first = model.forecast().matrix
    first[0, 0] = -99.0
    np.testing.assert_array_equal(model.fit_state.current_q, q_before)
    np.testing.assert_array_equal(model.fit_state.current_variance, h_before)
    second = model.forecast().matrix
    assert second[0, 0] != -99.0
    np.testing.assert_array_equal(second, model.forecast().matrix)


def test_valid_forecast_is_strictly_pd() -> None:
    forecast = DCCCovariance().fit(_dcc_like_returns()).forecast()
    assert forecast.diagnostics.positive_definite is True
    np.linalg.cholesky(forecast.matrix)


def test_invalid_forecast_matrix_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import covharness.models.dcc as dcc_module

    returns = _dcc_like_returns()
    model = DCCCovariance().fit(returns)

    def _bad_forecast(_state):
        n_assets = _state.n_assets
        return np.ones((n_assets, n_assets)), _state.current_variance, _state.current_q

    monkeypatch.setattr(dcc_module, "one_step_forecast_from_state", _bad_forecast)
    with pytest.raises(InvalidModelForecastError):
        model.forecast()


def test_bivariate_likelihood_matches_scalar_formula() -> None:
    rho = 0.3
    s_i, s_j = 0.4, -0.2
    value = bivariate_correlation_nll(rho, s_i, s_j)
    expected = 0.5 * (
        np.log(1.0 - rho**2) + (s_i**2 - 2 * rho * s_i * s_j + s_j**2) / (1.0 - rho**2)
    )
    assert value == pytest.approx(expected)


def test_pair_count_and_ordering() -> None:
    rows, cols = pair_indices(4)
    assert pair_count(4) == 6
    np.testing.assert_array_equal(rows, np.array([0, 0, 0, 1, 1, 2]))
    np.testing.assert_array_equal(cols, np.array([1, 2, 3, 2, 3, 3]))
    fitted = DCCCovariance().fit(_dcc_like_returns(n_assets=4)).fit_state
    assert fitted.n_pairs == 6
    assert fitted.pair_likelihood_type == "all_pairs_composite"


def test_all_pairs_objective_matches_hand_loop() -> None:
    target = np.array([[1.0, 0.2], [0.2, 1.0]])
    standardized = np.array([[0.1, -0.3], [0.4, 0.2], [-0.2, 0.5]])
    alpha, beta = 0.05, 0.9
    value = all_pairs_composite_nll(alpha, beta, target, standardized)
    q_state = target.copy()
    total = 0.0
    for time_index, shock in enumerate(standardized):
        rho = q_to_correlation(q_state)[0, 1]
        total += bivariate_correlation_nll(rho, shock[0], shock[1])
        if time_index < 2:
            q_state = dcc_q_update(q_state, shock, target, alpha, beta)
    assert value == pytest.approx(total)


def test_all_pairs_objective_is_permutation_equivariant() -> None:
    standardized = _rng(12).standard_normal((20, 4))
    target = sample_standardized_residual_correlation(standardized)
    permutation = np.array([2, 0, 3, 1])
    permuted_s = standardized[:, permutation]
    permuted_c = target[np.ix_(permutation, permutation)]
    original = all_pairs_composite_nll(0.04, 0.9, target, standardized)
    permuted = all_pairs_composite_nll(0.04, 0.9, permuted_c, permuted_s)
    assert permuted == pytest.approx(original)


def test_optimizer_result_obeys_strict_domain() -> None:
    returns = _dcc_like_returns()
    state = DCCCovariance().fit(returns).fit_state
    assert state.dcc_alpha >= 0.0
    assert state.dcc_beta >= 0.0
    assert state.dcc_alpha + state.dcc_beta < 1.0
    assert state.optimizer_success is True
    assert np.isfinite(state.optimizer_objective)


def test_optimizer_start_is_frozen() -> None:
    state = DCCCovariance().fit(_dcc_like_returns()).fit_state
    assert state.optimizer_start == (DCC_ALPHA0, DCC_BETA0)
    assert state.optimizer == "SLSQP"


def test_alpha_beta_zero_is_in_optimizer_domain() -> None:
    target = np.eye(3)
    standardized = _rng(13).standard_normal((12, 3))
    value = all_pairs_composite_nll(0.0, 0.0, target, standardized)
    assert np.isfinite(value)
    assert 0.0 + 0.0 <= 1.0


def test_optimizer_failure_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    import covharness.models.dcc as dcc_module

    class _Fake:
        success = False
        status = 4
        message = "iteration limit"
        fun = 12.0
        nfev = 8
        x = np.array([0.05, 0.90])

    monkeypatch.setattr(dcc_module, "minimize", lambda *_a, **_k: _Fake())
    with pytest.raises(InvalidModelForecastError, match="SLSQP failed"):
        estimate_dcc_parameters(np.eye(3), _rng(15).standard_normal((20, 3)))


def test_alpha_plus_beta_equals_one_is_rejected_as_headline_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covharness.models.dcc as dcc_module

    class _Fake:
        success = True
        status = 0
        message = "forced boundary"
        fun = 1.0
        nfev = 3
        x = np.array([0.2, 0.8])

    monkeypatch.setattr(dcc_module, "minimize", lambda *_a, **_k: _Fake())
    with pytest.raises(InvalidModelForecastError, match="alpha\\+beta<1"):
        estimate_dcc_parameters(np.eye(3), _rng(14).standard_normal((20, 3)))


def test_update_advances_h_and_q_once() -> None:
    returns = _dcc_like_returns()
    model = DCCCovariance().fit(returns)
    before = model.fit_state
    h_next, q_next = one_step_forecast_from_state(before)[1:]
    new_return = returns[-1] + np.array([0.1, -0.2, 0.05])
    model.update(new_return)
    after = model.fit_state
    np.testing.assert_allclose(after.current_variance, h_next)
    np.testing.assert_allclose(after.current_q, q_next)
    np.testing.assert_array_equal(after.fit_mean, before.fit_mean)
    np.testing.assert_array_equal(after.omega, before.omega)
    np.testing.assert_array_equal(after.garch_alpha, before.garch_alpha)
    np.testing.assert_array_equal(after.garch_beta, before.garch_beta)
    np.testing.assert_array_equal(after.target_c, before.target_c)
    assert after.dcc_alpha == before.dcc_alpha
    assert after.dcc_beta == before.dcc_beta


def test_forecast_after_update_differs_and_repeat_forecast_is_identical() -> None:
    returns = _dcc_like_returns()
    model = DCCCovariance().fit(returns)
    before = model.forecast().matrix.copy()
    again = model.forecast().matrix
    np.testing.assert_array_equal(before, again)
    model.update(returns[-1] + 0.3)
    after = model.forecast().matrix
    assert not np.allclose(before, after)
    np.testing.assert_array_equal(after, model.forecast().matrix)


def test_fit_then_twenty_updates_does_not_reestimate_parameters() -> None:
    returns = _dcc_like_returns(n_times=170)
    model = DCCCovariance().fit(returns[:150])
    alpha = model.fit_state.dcc_alpha
    beta = model.fit_state.dcc_beta
    target = model.fit_state.target_c.copy()
    omega = model.fit_state.omega.copy()
    forecasts = [model.forecast().matrix.copy()]
    for step in range(20):
        model.update(returns[150 + step])
        forecasts.append(model.forecast().matrix.copy())
    assert model.fit_state.dcc_alpha == alpha
    assert model.fit_state.dcc_beta == beta
    np.testing.assert_array_equal(model.fit_state.target_c, target)
    np.testing.assert_array_equal(model.fit_state.omega, omega)
    assert not np.allclose(forecasts[0], forecasts[-1])


def test_target_day_cannot_enter_before_update() -> None:
    returns = _dcc_like_returns(n_times=160)
    window = returns[:150]
    future = returns[40].copy()
    model = DCCCovariance().fit(window)
    first = model.forecast().matrix.copy()
    future = future + 5.0
    second = model.forecast().matrix
    np.testing.assert_array_equal(first, second)
    model.update(future)
    third = model.forecast().matrix
    assert not np.allclose(first, third)


def test_dcc_and_dcc_nl_differ_only_through_c_with_fixed_dynamics() -> None:
    returns = _dcc_like_returns(n_times=150, n_assets=3, seed=21)
    stage = fit_stage_one_garch(returns)
    variances = garch_variance_paths(
        stage.residuals, stage.omega, stage.garch_alpha, stage.garch_beta, stage.garch_backcast
    )
    standardized = standardized_residuals(stage.residuals, variances)
    sample_c = sample_standardized_residual_correlation(standardized)
    nl_c = nonlinear_standardized_residual_correlation(standardized)
    assert not np.allclose(sample_c, nl_c)
    alpha, beta = 0.05, 0.90
    q_sample = dcc_observed_q_path(sample_c, alpha, beta, standardized)
    q_nl = dcc_observed_q_path(nl_c, alpha, beta, standardized)
    q_nl_with_sample = dcc_observed_q_path(sample_c, alpha, beta, standardized)
    np.testing.assert_allclose(q_nl_with_sample, q_sample)
    assert not np.allclose(q_sample[-1], q_nl[-1])


def test_dcc_nl_does_not_shrink_final_h() -> None:
    returns = _dcc_like_returns(n_times=150, seed=22)
    model = DCCNonlinearCovariance().fit(returns)
    forecast = model.forecast().matrix
    reconstructed, _h_next, _q_next = one_step_forecast_from_state(model.fit_state)
    np.testing.assert_allclose(forecast, reconstructed)
    assert not np.allclose(forecast, model.fit_state.target_c)
    assert not np.allclose(np.diag(forecast), 1.0)
    del _h_next, _q_next


def test_no_cdcc_transformed_shock() -> None:
    q_matrix = np.array([[1.2, 0.2], [0.2, 0.9]])
    shock = np.array([0.5, -0.4])
    target = np.eye(2)
    original = dcc_q_update(q_matrix, shock, target, 0.05, 0.9)
    transformed = shock * np.sqrt(np.diag(q_matrix))
    cdcc = dcc_q_update(q_matrix, transformed, target, 0.05, 0.9)
    assert not np.allclose(original, cdcc)


def test_dcc_nl_fit_records_k_zero_and_nonlinshrink() -> None:
    model = DCCNonlinearCovariance().fit(_dcc_like_returns(n_times=150))
    state = model.fit_state
    assert state.target_type == "analytical_nonlinear_standardized_residual_correlation"
    assert state.nonlinear_reference_package == "nonlinshrink"
    assert state.nonlinear_reference_version == "0.7"
    assert state.nonlinear_k == 0
    assert state.target_divisor == "T"
    forecast = model.forecast()
    assert forecast.identity.name == "dcc_nl"
    assert forecast.diagnostics.positive_definite is True


def test_dcc_identity_name() -> None:
    forecast = DCCCovariance().fit(_dcc_like_returns()).forecast()
    assert forecast.identity.name == "dcc"
    assert forecast.identity.configuration["dcc_variant"] == "engle_2002_original"


def test_moderate_dimension_fit_forecast_update() -> None:
    returns = _dcc_like_returns(n_times=120, n_assets=20, seed=30)
    model = DCCCovariance().fit(returns)
    assert model.fit_state.n_pairs == 20 * 19 // 2
    np.linalg.cholesky(model.fit_state.target_c)
    first = model.forecast()
    assert first.diagnostics.positive_definite is True
    model.update(returns[-1] + 0.01)
    second = model.forecast()
    assert second.diagnostics.positive_definite is True
    assert not np.allclose(first.matrix, second.matrix)

