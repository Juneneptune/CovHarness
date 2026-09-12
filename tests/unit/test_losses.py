"""Arithmetic and contract tests for covariance-space losses."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.diagnostics.epps import matrix_eigen_diagnostics
from covharness.losses import (
    ForecastNotPositiveDefiniteError,
    InvalidCovarianceMatrixError,
    ProxyNotPositiveDefiniteError,
    covariance_localization_diagnostic,
    full_stein_loss,
    reduced_qlike_loss,
    squared_frobenius_loss,
    unsquared_frobenius_loss,
)

IDENTITY_2 = np.eye(2)
SPD_S = np.array([[2.0, 0.4], [0.4, 1.0]])
SPD_H = np.array([[1.5, 0.2], [0.2, 1.2]])
SINGULAR_S = np.array([[1.0, 1.0], [1.0, 1.0]])
NOT_PD_H = np.array([[1.0, 2.0], [2.0, 1.0]])


def test_frobenius_zero_at_identity() -> None:
    assert squared_frobenius_loss(SPD_S, SPD_S) == 0.0


def test_frobenius_equals_elementwise_squared_sum() -> None:
    residual = SPD_S - SPD_H
    expected = float(np.sum(residual * residual))
    np.testing.assert_allclose(squared_frobenius_loss(SPD_S, SPD_H), expected)


def test_frobenius_counts_symmetric_off_diagonal_errors_twice() -> None:
    proxy = np.array([[1.0, 0.5], [0.5, 2.0]])
    forecast = np.array([[1.0, 0.0], [0.0, 2.0]])
    # Two off-diagonal residuals of 0.5. Squared sum is 0.25 + 0.25 = 0.5.
    np.testing.assert_allclose(squared_frobenius_loss(proxy, forecast), 0.5)


def test_frobenius_accepts_psd_and_singular_proxy() -> None:
    value = squared_frobenius_loss(SINGULAR_S, IDENTITY_2)
    assert np.isfinite(value)
    indefinite = np.array([[1.0, 0.0], [0.0, -0.5]])
    assert np.isfinite(squared_frobenius_loss(indefinite, IDENTITY_2))


def test_reduced_qlike_finite_at_matching_spd() -> None:
    value = reduced_qlike_loss(IDENTITY_2, IDENTITY_2)
    np.testing.assert_allclose(value, 2.0)


def test_reduced_qlike_ranking_optimal_on_scale_family() -> None:
    sigma = IDENTITY_2
    loss_true = reduced_qlike_loss(sigma, sigma)
    loss_too_large = reduced_qlike_loss(sigma, 2.0 * sigma)
    loss_too_small = reduced_qlike_loss(sigma, 0.5 * sigma)
    assert loss_true < loss_too_large
    assert loss_true < loss_too_small


def test_full_stein_approximately_zero_at_matching_spd() -> None:
    np.testing.assert_allclose(full_stein_loss(SPD_S, SPD_S), 0.0, atol=1e-12)


def test_full_stein_nonnegative_on_spd_examples() -> None:
    assert full_stein_loss(SPD_S, SPD_H) >= -1e-12
    assert full_stein_loss(SPD_H, SPD_S) >= -1e-12
    assert full_stein_loss(IDENTITY_2, SPD_H) >= -1e-12


def test_stein_is_asymmetric() -> None:
    forward = full_stein_loss(SPD_S, SPD_H)
    backward = full_stein_loss(SPD_H, SPD_S)
    assert forward != pytest.approx(backward)


def test_reduced_qlike_and_stein_share_pairwise_differentials() -> None:
    s = SPD_S
    h_a = IDENTITY_2
    h_b = SPD_H
    qlike_diff = reduced_qlike_loss(s, h_a) - reduced_qlike_loss(s, h_b)
    stein_diff = full_stein_loss(s, h_a) - full_stein_loss(s, h_b)
    np.testing.assert_allclose(qlike_diff, stein_diff, atol=1e-12)


def test_full_stein_scale_invariance() -> None:
    scale = 3.0
    base = full_stein_loss(SPD_S, SPD_H)
    scaled = full_stein_loss(scale * SPD_S, scale * SPD_H)
    np.testing.assert_allclose(scaled, base, atol=1e-12)


def test_singular_proxy_accepted_by_reduced_qlike() -> None:
    value = reduced_qlike_loss(SINGULAR_S, SPD_H)
    assert np.isfinite(value)


def test_singular_proxy_rejected_by_full_stein() -> None:
    with pytest.raises(ProxyNotPositiveDefiniteError, match="logdet"):
        full_stein_loss(SINGULAR_S, SPD_H)


def test_non_pd_forecast_rejected_by_qlike_and_stein() -> None:
    with pytest.raises(ForecastNotPositiveDefiniteError, match="not repair"):
        reduced_qlike_loss(SPD_S, NOT_PD_H)
    with pytest.raises(ForecastNotPositiveDefiniteError, match="not repair"):
        full_stein_loss(SPD_S, NOT_PD_H)
    singular_forecast = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ForecastNotPositiveDefiniteError):
        reduced_qlike_loss(SPD_S, singular_forecast)


def test_nonsymmetric_matrices_fail() -> None:
    skewed = np.array([[1.0, 0.4], [0.1, 1.0]])
    with pytest.raises(InvalidCovarianceMatrixError, match="symmetric"):
        squared_frobenius_loss(skewed, IDENTITY_2)
    with pytest.raises(InvalidCovarianceMatrixError, match="symmetric"):
        reduced_qlike_loss(SPD_S, skewed)
    with pytest.raises(InvalidCovarianceMatrixError, match="symmetric"):
        full_stein_loss(skewed, SPD_H)


def test_dimension_mismatch_fails() -> None:
    with pytest.raises(InvalidCovarianceMatrixError, match="same shape"):
        squared_frobenius_loss(IDENTITY_2, np.eye(3))
    with pytest.raises(InvalidCovarianceMatrixError, match="square"):
        reduced_qlike_loss(np.ones((2, 3)), IDENTITY_2)


def test_nonfinite_inputs_fail() -> None:
    bad = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(InvalidCovarianceMatrixError, match="finite"):
        squared_frobenius_loss(bad, IDENTITY_2)
    inf = np.array([[1.0, 0.0], [0.0, np.inf]])
    with pytest.raises(InvalidCovarianceMatrixError, match="finite"):
        reduced_qlike_loss(IDENTITY_2, inf)


def test_inputs_are_not_mutated() -> None:
    proxy = np.array([[2.0, 0.3], [0.3, 1.5]], dtype=float)
    forecast = np.array([[1.7, 0.2], [0.2, 1.1]], dtype=float)
    proxy_before = proxy.copy()
    forecast_before = forecast.copy()
    squared_frobenius_loss(proxy, forecast)
    reduced_qlike_loss(proxy, forecast)
    full_stein_loss(proxy, forecast)
    np.testing.assert_array_equal(proxy, proxy_before)
    np.testing.assert_array_equal(forecast, forecast_before)


def test_unsquared_frobenius_is_sqrt_of_squared() -> None:
    squared = squared_frobenius_loss(SPD_S, SPD_H)
    np.testing.assert_allclose(
        unsquared_frobenius_loss(SPD_S, SPD_H),
        np.sqrt(squared),
    )


def test_localization_variance_and_correlation_definitions() -> None:
    proxy = np.array([[4.0, 1.2], [1.2, 1.0]])
    forecast = np.array([[1.0, 0.0], [0.0, 1.0]])
    diagnostic = covariance_localization_diagnostic(proxy, forecast)
    np.testing.assert_allclose(diagnostic.variance_squared_error, (4.0 - 1.0) ** 2)
    expected_corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    np.testing.assert_allclose(diagnostic.proxy_correlation, expected_corr)
    np.testing.assert_allclose(diagnostic.forecast_correlation, np.eye(2))
    resid = expected_corr - np.eye(2)
    np.testing.assert_allclose(
        diagnostic.correlation_frobenius_squared,
        float(np.sum(resid * resid)),
    )


def test_localization_rejects_nonpositive_diagonal() -> None:
    singular_diag = np.array([[0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(InvalidCovarianceMatrixError, match="nonpositive diagonal"):
        covariance_localization_diagnostic(singular_diag, IDENTITY_2)


def test_matrix_diagnostic_reports_m_over_n() -> None:
    eigvals, rank, min_eig, psd, cond, symmetry_error, m_over_n = matrix_eigen_diagnostics(
        np.eye(5), n_returns=78
    )
    assert psd is True
    assert rank == 5
    np.testing.assert_allclose(m_over_n, 78 / 5)
    _, _, _, _, _, _, missing = matrix_eigen_diagnostics(np.eye(2))
    assert missing is None
