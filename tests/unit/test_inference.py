"""Pairwise DM / HAC / Clark-West tests on synthetic differentials only."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from covharness.inference import (
    ALTERNATIVE_A_BETTER,
    ALTERNATIVE_B_BETTER,
    ALTERNATIVE_TWO_SIDED,
    ClarkWestScopeError,
    DegenerateLossDifferentialError,
    clark_west_squared_error,
    diebold_mariano,
    diebold_mariano_from_losses,
    hac_long_run_variance,
    loss_differential,
    loss_differential_diagnostics,
    naive_iid_t_statistic,
    newey_west_1994_lags,
    plot_hac_size_power,
    simulate_hac_size_power,
)
from covharness.inference.hac import bartlett_weights


def test_loss_differential_sign_convention() -> None:
    loss_a = np.array([1.0, 2.0, 0.5])
    loss_b = np.array([2.0, 2.0, 0.0])
    differential = loss_differential(loss_a, loss_b)
    np.testing.assert_allclose(differential, np.array([-1.0, 0.0, 0.5]))
    assert differential[0] < 0.0  # A wins date 0
    assert differential[2] > 0.0  # B wins date 2


def test_zero_mean_iid_dm_statistic_is_zero_after_demeaning() -> None:
    rng = np.random.default_rng(7)
    series = rng.normal(size=400)
    series = series - series.mean()
    result = diebold_mariano(series, alternative=ALTERNATIVE_TWO_SIDED)
    assert result.mean_differential == pytest.approx(0.0, abs=1e-15)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)
    assert result.hln_small_sample_correction is False


def test_positive_and_negative_mean_direction() -> None:
    negative = np.full(80, -0.4) + 0.01 * np.arange(80)
    negative = negative - negative.mean() - 0.3
    positive = -negative
    a_better = diebold_mariano(negative, alternative=ALTERNATIVE_TWO_SIDED)
    b_better = diebold_mariano(positive, alternative=ALTERNATIVE_TWO_SIDED)
    assert a_better.mean_differential < 0.0
    assert a_better.statistic < 0.0
    assert b_better.mean_differential > 0.0
    assert b_better.statistic > 0.0
    assert a_better.statistic == pytest.approx(-b_better.statistic)


def test_two_sided_and_one_sided_p_values() -> None:
    series = np.linspace(-1.0, -0.2, 60)
    two = diebold_mariano(series, alternative=ALTERNATIVE_TWO_SIDED)
    a_better = diebold_mariano(series, alternative=ALTERNATIVE_A_BETTER)
    b_better = diebold_mariano(series, alternative=ALTERNATIVE_B_BETTER)
    assert two.alternative == "two_sided"
    assert a_better.alternative == "a_better"
    assert b_better.alternative == "b_better"
    assert a_better.p_value < 0.01
    assert b_better.p_value > 0.99
    assert two.p_value == pytest.approx(2.0 * a_better.p_value, rel=1e-10)
    from_losses = diebold_mariano_from_losses(
        series, np.zeros_like(series), alternative="a_better"
    )
    assert from_losses.statistic == pytest.approx(a_better.statistic)


def test_hac_lag_zero_is_heteroskedasticity_only_variance() -> None:
    series = np.array([1.0, -0.5, 0.25, 2.0, -1.25, 0.4], dtype=float)
    hac = hac_long_run_variance(series, maxlags=0)
    centered = series - series.mean()
    gamma0 = float(np.dot(centered, centered) / series.size)
    assert hac.maxlags == 0
    assert hac.lag_rule == "user"
    assert hac.long_run_variance == pytest.approx(gamma0)
    assert hac.standard_error_of_mean == pytest.approx(np.sqrt(gamma0 / series.size))
    dm = diebold_mariano(series, maxlags=0)
    assert dm.standard_error == pytest.approx(hac.standard_error_of_mean)


def test_serial_correlation_inflates_hac_standard_error() -> None:
    rng = np.random.default_rng(11)
    innovations = rng.normal(size=500)
    ar = np.empty(500)
    ar[0] = innovations[0]
    for t in range(1, 500):
        ar[t] = 0.7 * ar[t - 1] + innovations[t]
    se_white = hac_long_run_variance(ar, maxlags=0).standard_error_of_mean
    se_hac = hac_long_run_variance(ar).standard_error_of_mean
    assert se_hac > se_white


def test_degenerate_differential_is_rejected() -> None:
    constants = (
        np.zeros(100),
        np.ones(100),
        np.full(40, 1.5),
        np.full(100, 0.1),
        np.full(250, 0.1),
    )
    for series in constants:
        with pytest.raises(DegenerateLossDifferentialError, match="equal"):
            diebold_mariano(series)
        with pytest.raises(DegenerateLossDifferentialError, match="equal"):
            hac_long_run_variance(series)
        with pytest.raises(DegenerateLossDifferentialError, match="equal"):
            hac_long_run_variance(series, maxlags=0)

    nearby = np.full(100, 0.1)
    nearby[-1] = 0.1 + 1e-9
    result = diebold_mariano(nearby)
    assert np.isfinite(result.statistic)
    assert result.hac_long_run_variance > 0.0


def test_nonfinite_hac_long_run_variance_is_rejected() -> None:
    # Finite inputs whose second-moment arithmetic overflows float64.
    series = np.array([1.0e200, -1.0e200], dtype=float)
    with pytest.raises(DegenerateLossDifferentialError, match="not finite"):
        hac_long_run_variance(series, maxlags=0)
    with pytest.raises(DegenerateLossDifferentialError, match="not finite"):
        diebold_mariano(series, maxlags=0)


def test_degenerate_diagnostics_report_attempted_hac_lag() -> None:
    series = np.full(100, 0.1)
    automatic = loss_differential_diagnostics(series, acf_lags=20, maxlags=None)
    assert automatic.hac_maxlags == newey_west_1994_lags(100)
    assert automatic.hac_maxlags == 4
    assert automatic.hac_long_run_variance is None
    lag0 = loss_differential_diagnostics(series, acf_lags=20, maxlags=0)
    assert lag0.hac_maxlags == 0
    assert lag0.hac_long_run_variance is None


def test_newey_west_bandwidth_rule_is_deterministic() -> None:
    assert newey_west_1994_lags(100) == 4
    assert newey_west_1994_lags(500) == int(np.floor(4.0 * (500 / 100.0) ** (2.0 / 9.0)))
    first = hac_long_run_variance(np.linspace(0.0, 1.0, 250))
    second = hac_long_run_variance(np.linspace(0.0, 1.0, 250))
    assert first.maxlags == second.maxlags == newey_west_1994_lags(250)
    assert first.lag_rule == "newey_west_1994"
    np.testing.assert_allclose(bartlett_weights(3), 1.0 - np.array([1, 2, 3]) / 4.0)


def test_explicit_lag_override() -> None:
    series = np.linspace(-1.0, 1.0, 120)
    default = diebold_mariano(series)
    overridden = diebold_mariano(series, maxlags=3)
    assert overridden.hac_maxlags == 3
    assert overridden.hac_lag_rule == "user"
    assert default.hac_lag_rule == "newey_west_1994"
    assert overridden.hac_maxlags != default.hac_maxlags or 3 == default.hac_maxlags


def test_hac_matches_statsmodels_intercept_only_ols() -> None:
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(21)
    series = rng.normal(size=180)
    series[1:] += 0.4 * series[:-1]
    lag = 5
    ours = hac_long_run_variance(series, maxlags=lag)
    ols = sm.OLS(series, np.ones(series.shape[0]))
    fitted = ols.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lag, "use_correction": False, "kernel": "bartlett"},
    )
    np.testing.assert_allclose(ours.mean, float(fitted.params[0]), rtol=1e-12)
    np.testing.assert_allclose(ours.standard_error_of_mean, float(fitted.bse[0]), rtol=1e-10)
    dm = diebold_mariano(series, maxlags=lag)
    np.testing.assert_allclose(dm.statistic, float(fitted.tvalues[0]), rtol=1e-10)


def test_cumulative_loss_differential() -> None:
    series = np.array([0.2, -0.5, 0.1, 0.4])
    diagnostic = loss_differential_diagnostics(series, acf_lags=2, maxlags=0)
    np.testing.assert_allclose(diagnostic.cumulative, np.cumsum(series))
    assert diagnostic.cumulative[-1] == pytest.approx(series.sum())


def test_acf_on_known_white_noise_and_ar1() -> None:
    alternating = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)
    alternating_diag = loss_differential_diagnostics(alternating, acf_lags=2, maxlags=0)
    assert alternating_diag.acf[0] == pytest.approx(-(len(alternating) - 1) / len(alternating))

    rng = np.random.default_rng(3)
    n_obs = 4000
    rho = 0.5
    innovations = rng.normal(size=n_obs)
    ar = np.empty(n_obs)
    ar[0] = innovations[0] / np.sqrt(1.0 - rho**2)
    for t in range(1, n_obs):
        ar[t] = rho * ar[t - 1] + innovations[t]
    ar_diag = loss_differential_diagnostics(ar, acf_lags=3)
    assert ar_diag.acf[0] == pytest.approx(rho, abs=0.05)


def test_kappa_and_t_eff() -> None:
    iid = np.array([1.0, -0.5, 0.25, 2.0, -1.25, 0.4, 0.1, -0.8])
    iid_lag0 = loss_differential_diagnostics(iid, acf_lags=2, maxlags=0)
    assert iid_lag0.kappa == pytest.approx(1.0)
    assert iid_lag0.t_eff == pytest.approx(float(iid.size))

    rng = np.random.default_rng(4)
    n_obs = 800
    rho = 0.6
    innovations = rng.normal(size=n_obs)
    ar = np.empty(n_obs)
    ar[0] = innovations[0]
    for t in range(1, n_obs):
        ar[t] = rho * ar[t - 1] + innovations[t]
    ar_diag = loss_differential_diagnostics(ar, acf_lags=10)
    assert ar_diag.kappa is not None and ar_diag.kappa > 1.0
    assert ar_diag.t_eff is not None and ar_diag.t_eff < n_obs

    neg = np.empty(n_obs)
    neg[0] = innovations[0]
    for t in range(1, n_obs):
        neg[t] = -0.4 * neg[t - 1] + innovations[t]
    neg_diag = loss_differential_diagnostics(neg, acf_lags=10)
    assert neg_diag.kappa is not None and neg_diag.kappa < 1.0
    assert neg_diag.t_eff is not None and neg_diag.t_eff > n_obs
    assert "exceeds T" in neg_diag.notes


def test_inputs_are_not_mutated() -> None:
    loss_a = np.array([1.0, 2.0, 3.0, 4.0])
    loss_b = np.array([1.5, 1.5, 3.5, 3.0])
    original_a = loss_a.copy()
    original_b = loss_b.copy()
    diebold_mariano_from_losses(loss_a, loss_b)
    loss_differential_diagnostics(loss_a - loss_b, acf_lags=2)
    np.testing.assert_array_equal(loss_a, original_a)
    np.testing.assert_array_equal(loss_b, original_b)


def test_clark_west_requires_explicit_nested_declaration() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0])
    restricted = np.zeros(4)
    unrestricted = np.array([0.5, 0.0, 0.5, 0.0])
    with pytest.raises(ClarkWestScopeError, match="nested=True"):
        clark_west_squared_error(y, restricted, unrestricted, nested=False)


def test_clark_west_rejects_matrix_losses() -> None:
    y = np.ones((4, 2))
    with pytest.raises(ClarkWestScopeError, match="scalar"):
        clark_west_squared_error(y, y, y, nested=True)


def test_clark_west_adjusted_differential_matches_formula() -> None:
    y = np.array([1.0, 0.0, 2.0])
    f_r = np.array([0.0, 0.0, 0.0])
    f_u = np.array([0.5, 0.0, 1.0])
    error_r = y - f_r
    error_u = y - f_u
    expected = error_r**2 - error_u**2 + (f_r - f_u) ** 2
    result = clark_west_squared_error(
        y, f_r, f_u, nested=True, alternative="b_better", maxlags=0
    )
    assert result.nested_declared is True
    assert result.loss == "squared_error"
    assert result.mean_adjusted_differential == pytest.approx(float(expected.mean()))
    assert result.alternative == "b_better"
    assert result.dm.hln_small_sample_correction is False


def test_synthetic_naive_overrejection_versus_hac() -> None:
    result = simulate_hac_size_power()
    assert result.seed == 20260912
    assert result.n_reps == 2000
    assert result.n_obs == 250
    # IID null. Both tests stay near the 5% nominal size.
    assert 0.03 <= result.iid_null_naive_rejection <= 0.08
    assert 0.03 <= result.iid_null_dm_rejection <= 0.08
    # AR(1) null. The naive t-test over-rejects. HAC DM is closer to nominal.
    assert result.ar_null_naive_rejection > 0.25
    assert result.ar_null_dm_rejection < result.ar_null_naive_rejection - 0.10
    assert result.ar_null_dm_rejection < 0.16
    # Alternative with negative mean. DM rejects in the A-better direction.
    assert result.alternative_dm_rejection > 0.7
    assert result.alternative_dm_a_better_rejection > 0.8
    figure = plot_hac_size_power(result, "results/dm_hac_size.png")
    assert figure.exists()


def test_normal_critical_value_used_by_size_demo() -> None:
    assert float(norm.ppf(0.975)) == pytest.approx(1.959963984540054, rel=0.0, abs=1e-12)
