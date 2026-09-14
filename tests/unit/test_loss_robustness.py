"""Proxy-robust ranking demonstration. Fixed seed. Not a flaky probabilistic test."""

from __future__ import annotations

import numpy as np

from covharness.losses.robustness import (
    ROBUSTNESS_N_DRAWS,
    ROBUSTNESS_SEED,
    analytic_expected_losses,
    median_matched_forecast,
    monte_carlo_proxy_ranking,
    true_forecast,
)


def test_analytic_expected_losses_rank_true_forecast_under_robust_criteria() -> None:
    analytic = analytic_expected_losses()
    assert analytic.squared_frobenius_true < analytic.squared_frobenius_median
    assert analytic.reduced_qlike_true < analytic.reduced_qlike_median
    # Unsquared Frobenius is minimized at the median of the Exp(1) multiplier.
    assert analytic.unsquared_frobenius_median < analytic.unsquared_frobenius_true
    np.testing.assert_allclose(analytic.squared_frobenius_true, analytic.sigma_frobenius_squared)
    np.testing.assert_allclose(
        analytic.unsquared_frobenius_true,
        (2.0 / np.e) * np.sqrt(analytic.sigma_frobenius_squared),
    )
    np.testing.assert_allclose(
        analytic.unsquared_frobenius_median,
        np.log(2.0) * np.sqrt(analytic.sigma_frobenius_squared),
    )


def test_monte_carlo_preserves_robust_ranking_and_flips_unsquared_frobenius() -> None:
    result = monte_carlo_proxy_ranking(
        n_draws=ROBUSTNESS_N_DRAWS,
        seed=ROBUSTNESS_SEED,
    )
    analytic = analytic_expected_losses()
    # Mean ranking under robust losses. The true forecast is H_A = Sigma.
    assert result.mean_squared_frobenius_true < result.mean_squared_frobenius_median
    assert result.mean_reduced_qlike_true < result.mean_reduced_qlike_median
    assert result.mean_full_stein_true < result.mean_full_stein_median
    # Non-robust unsquared Frobenius favors the median-matched inferior forecast.
    assert result.mean_unsquared_frobenius_median < result.mean_unsquared_frobenius_true
    np.testing.assert_allclose(
        result.mean_squared_frobenius_true,
        analytic.squared_frobenius_true,
        rtol=0.03,
    )
    np.testing.assert_allclose(
        result.mean_reduced_qlike_true,
        analytic.reduced_qlike_true,
        rtol=0.03,
    )
    # Single-draw majority ranking need not match expected-loss ranking.
    assert result.single_draw_hb_win_rate_unsquared_frobenius > 0.5
    assert result.n_draws == ROBUSTNESS_N_DRAWS
