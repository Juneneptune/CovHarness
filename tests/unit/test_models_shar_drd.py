"""SHAR-DRD semivariance identity, equal-slope reduction, and lag index."""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from covharness.models.har_drd import HARDRDRealizedCovariance, har_predictors
from covharness.models.shar_drd import (
    SHARDRDRealizedCovariance,
    shar_variance_predictors,
)
from covharness.realized.rcov import realized_covariance, realized_semivariances


def _cube(variances: np.ndarray, rho: float = 0.2) -> np.ndarray:
    n_times, n_assets = variances.shape
    corr = np.full((n_assets, n_assets), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    cube = np.empty((n_times, n_assets, n_assets), dtype=float)
    for time in range(n_times):
        scale = np.sqrt(variances[time])
        cube[time] = corr * np.outer(scale, scale)
    return cube


def test_dyadic_semivariances_sum_bitwise_to_rv() -> None:
    returns = np.array([[0.5, -0.25], [0.0, 0.125], [-0.5, 0.0]], dtype=float)
    positive, negative = realized_semivariances(returns)
    variance = np.diag(realized_covariance(returns))
    assert np.array_equal(positive + negative, variance)
    assert positive[1] == 0.125 ** 2
    assert negative[0] == 0.5 ** 2
    assert positive[0] == 0.5 ** 2


def test_equal_semivariance_slopes_reproduce_har() -> None:
    rng = np.random.default_rng(11)
    variances = 0.4 + rng.random((48, 2))
    share = rng.uniform(0.15, 0.85, size=variances.shape)
    positive = share * variances
    negative = variances - positive
    cube = _cube(variances)
    har = HARDRDRealizedCovariance().fit(cube)
    shar = SHARDRDRealizedCovariance().fit(cube, positive, negative)
    beta = har.fit_state.beta_D
    shar._fit = replace(
        shar.fit_state,
        alpha_S=np.array(har.fit_state.alpha_D, dtype=float, copy=True),
        beta_S=np.array([beta[0], beta[0], beta[1], beta[2]], dtype=float),
        alpha_R=np.array(har.fit_state.alpha_R, dtype=float, copy=True),
        beta_R=np.array(har.fit_state.beta_R, dtype=float, copy=True),
        window_mean=np.array(har.fit_state.window_mean, dtype=float, copy=True),
    )
    np.testing.assert_allclose(shar.forecast().matrix, har.forecast().matrix, rtol=0.0, atol=1e-15)


def test_response_row_is_not_a_semivariance_feature() -> None:
    rng = np.random.default_rng(3)
    variances = 0.4 + rng.random((40, 2))
    positive = 0.4 * variances
    negative = variances - positive
    index = 30
    original = shar_variance_predictors(variances, positive, negative, index)
    shifted = positive.copy()
    shifted[index] = positive[index] + 5.0
    np.testing.assert_array_equal(
        shar_variance_predictors(variances, shifted, negative, index), original
    )
    lagged = positive.copy()
    lagged[index - 1] = positive[index - 1] + 5.0
    changed = shar_variance_predictors(variances, lagged, negative, index)
    assert not np.array_equal(changed, original)
    weekly = har_predictors(variances, index)
    np.testing.assert_array_equal(original[2], weekly[1])
    np.testing.assert_array_equal(original[3], weekly[2])
