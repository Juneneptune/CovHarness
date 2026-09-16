"""Synthetic tests for standalone Ledoit-Wolf linear and analytical NL shrinkage."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from covharness.models import (
    InvalidModelForecastError,
    InvalidModelInputError,
    LedoitWolfLinearCovariance,
    LedoitWolfNonlinearCovariance,
)
from covharness.models.ledoit_wolf import (
    apply_linear_identity_shrinkage,
    centered_return_moments,
)

LINEAR_ATOL = 1e-10
NL_ATOL = 1e-8
AFFINE_ATOL = 1e-8


def _rng(seed: int = 20260916) -> np.random.Generator:
    return np.random.default_rng(seed)


def _returns_with_mean(n_times: int = 40, n_assets: int = 5, seed: int = 20260916) -> np.ndarray:
    rng = _rng(seed)
    mean = np.linspace(-0.4, 0.6, n_assets)
    covariance = _spd_with_dispersed_spectrum(n_assets, rng)
    centered = rng.multivariate_normal(np.zeros(n_assets), covariance, size=n_times)
    return centered + mean


def _spd_with_dispersed_spectrum(n_assets: int, rng: np.random.Generator) -> np.ndarray:
    # Distinct positive eigenvalues so NL shrinkage is not an affine map.
    spectrum = np.linspace(0.25, 8.0, n_assets)
    orthogonal, _ = np.linalg.qr(rng.standard_normal((n_assets, n_assets)))
    return orthogonal @ np.diag(spectrum) @ orthogonal.T


def _independent_sample_covariance(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = returns.mean(axis=0)
    centered = returns - mean
    n_eff = returns.shape[0] - 1
    sample = centered.T @ centered / n_eff
    return mean, centered, sample


def _sklearn_reference(centered: np.ndarray) -> tuple[np.ndarray, float]:
    n_times = centered.shape[0]
    n_eff = n_times - 1
    scaled = np.sqrt(n_times / n_eff) * centered
    fitted = LedoitWolf(assume_centered=True, store_precision=False).fit(scaled)
    return np.asarray(fitted.covariance_, dtype=float), float(fitted.shrinkage_)


def _affine_residual(sample_eigenvalues: np.ndarray, shrunk: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(sample_eigenvalues), sample_eigenvalues])
    coef, *_ = np.linalg.lstsq(design, shrunk, rcond=None)
    return shrunk - design @ coef


def _assert_commutes(left: np.ndarray, right: np.ndarray, atol: float) -> None:
    np.testing.assert_allclose(left @ right, right @ left, atol=atol, rtol=0.0)


def test_return_input_rejects_wrong_ndim() -> None:
    with pytest.raises(InvalidModelInputError, match=r"\(T, N\)"):
        LedoitWolfLinearCovariance().fit(np.zeros((4, 3, 3)))


def test_return_input_rejects_t_equals_one() -> None:
    with pytest.raises(InvalidModelInputError, match="at least two"):
        LedoitWolfLinearCovariance().fit(np.ones((1, 3)))


def test_return_input_rejects_empty_assets() -> None:
    with pytest.raises(InvalidModelInputError, match="at least one asset"):
        LedoitWolfLinearCovariance().fit(np.ones((5, 0)))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_return_input_rejects_nonfinite(bad_value: float) -> None:
    returns = _returns_with_mean()
    returns = returns.copy()
    returns[3, 1] = bad_value
    with pytest.raises(InvalidModelInputError, match="finite"):
        LedoitWolfLinearCovariance().fit(returns)
    with pytest.raises(InvalidModelInputError, match="finite"):
        LedoitWolfNonlinearCovariance().fit(returns)


def test_inputs_are_not_mutated() -> None:
    returns = _returns_with_mean()
    original = returns.copy()
    linear = LedoitWolfLinearCovariance().fit(returns).forecast()
    nonlinear = LedoitWolfNonlinearCovariance().fit(returns).forecast()
    np.testing.assert_array_equal(returns, original)
    assert not np.shares_memory(linear.matrix, returns)
    assert not np.shares_memory(nonlinear.matrix, returns)
    linear.matrix[0, 0] = -99.0
    nonlinear.matrix[0, 0] = -99.0
    np.testing.assert_array_equal(returns, original)
    np.testing.assert_allclose(
        LedoitWolfLinearCovariance().fit(returns).forecast().matrix,
        LedoitWolfLinearCovariance().fit(original).forecast().matrix,
    )


def test_in_window_mean_is_removed_exactly() -> None:
    returns = _returns_with_mean()
    shifted = returns + np.linspace(1.0, 4.0, returns.shape[1])
    linear = LedoitWolfLinearCovariance().fit(returns)
    shifted_linear = LedoitWolfLinearCovariance().fit(shifted)
    np.testing.assert_allclose(linear.forecast().matrix, shifted_linear.forecast().matrix)
    np.testing.assert_allclose(
        shifted_linear.fit_state.mean,
        linear.fit_state.mean + np.linspace(1.0, 4.0, returns.shape[1]),
    )
    moments = centered_return_moments(returns)
    np.testing.assert_allclose(moments.centered.mean(axis=0), 0.0, atol=1e-14)


def test_sample_covariance_matches_independent_numpy() -> None:
    returns = _returns_with_mean()
    moments = centered_return_moments(returns)
    mean, centered, sample = _independent_sample_covariance(returns)
    np.testing.assert_allclose(moments.mean, mean)
    np.testing.assert_allclose(moments.centered, centered)
    np.testing.assert_allclose(moments.sample_covariance, sample)
    np.testing.assert_allclose(sample, centered.T @ centered / (returns.shape[0] - 1))


def test_sklearn_rescaling_matches_t_minus_one_gram() -> None:
    returns = _returns_with_mean()
    _mean, centered, sample = _independent_sample_covariance(returns)
    n_times = returns.shape[0]
    scaled = np.sqrt(n_times / (n_times - 1)) * centered
    sklearn_gram = scaled.T @ scaled / n_times
    np.testing.assert_allclose(sklearn_gram, sample, atol=1e-14, rtol=0.0)


def test_linear_matches_sklearn_2004b_on_rescaled_centered_returns() -> None:
    returns = _returns_with_mean()
    model = LedoitWolfLinearCovariance().fit(returns)
    _mean, centered, _sample = _independent_sample_covariance(returns)
    sklearn_cov, sklearn_rho = _sklearn_reference(centered)
    np.testing.assert_allclose(model.fit_state.rho, sklearn_rho, atol=LINEAR_ATOL)
    np.testing.assert_allclose(model.forecast().matrix, sklearn_cov, atol=LINEAR_ATOL)


def test_linear_rho_is_in_unit_interval() -> None:
    returns = _returns_with_mean()
    rho = LedoitWolfLinearCovariance().fit(returns).fit_state.rho
    assert 0.0 <= rho <= 1.0


def test_linear_equals_convex_combination_of_s_and_mu_i() -> None:
    returns = _returns_with_mean()
    model = LedoitWolfLinearCovariance().fit(returns)
    _mean, _centered, sample = _independent_sample_covariance(returns)
    expected = apply_linear_identity_shrinkage(
        sample, rho=model.fit_state.rho, mu=model.fit_state.mu
    )
    np.testing.assert_allclose(model.forecast().matrix, expected, atol=LINEAR_ATOL)
    mu = float(np.trace(sample) / sample.shape[0])
    assert model.fit_state.mu == pytest.approx(mu)


def test_linear_eigenvalues_obey_common_affine_map() -> None:
    returns = _returns_with_mean()
    model = LedoitWolfLinearCovariance().fit(returns)
    _mean, _centered, sample = _independent_sample_covariance(returns)
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    shrunk = (1.0 - model.fit_state.rho) * eigenvalues + model.fit_state.rho * model.fit_state.mu
    reconstructed = eigenvectors @ np.diag(shrunk) @ eigenvectors.T
    np.testing.assert_allclose(model.forecast().matrix, reconstructed, atol=LINEAR_ATOL)
    residual = _affine_residual(eigenvalues, shrunk)
    np.testing.assert_allclose(residual, 0.0, atol=AFFINE_ATOL)


def test_linear_retains_sample_eigenvectors_by_reconstruction() -> None:
    returns = _returns_with_mean()
    forecast = LedoitWolfLinearCovariance().fit(returns).forecast().matrix
    _mean, _centered, sample = _independent_sample_covariance(returns)
    _assert_commutes(forecast, sample, LINEAR_ATOL)
    _eigenvalues, eigenvectors = np.linalg.eigh(sample)
    projected = eigenvectors.T @ forecast @ eigenvectors
    off_diagonal = projected - np.diag(np.diag(projected))
    np.testing.assert_allclose(off_diagonal, 0.0, atol=LINEAR_ATOL)
    np.testing.assert_allclose(
        forecast,
        eigenvectors @ np.diag(np.diag(projected)) @ eigenvectors.T,
        atol=LINEAR_ATOL,
    )
    del _eigenvalues


def test_linear_helper_limits_rho_zero_and_one() -> None:
    _mean, _centered, sample = _independent_sample_covariance(_returns_with_mean())
    mu = float(np.trace(sample) / sample.shape[0])
    np.testing.assert_allclose(
        apply_linear_identity_shrinkage(sample, rho=0.0, mu=mu),
        sample,
    )
    np.testing.assert_allclose(
        apply_linear_identity_shrinkage(sample, rho=1.0, mu=mu),
        mu * np.eye(sample.shape[0]),
    )


def test_linear_permutation_equivariance() -> None:
    returns = _returns_with_mean()
    permutation = np.array([2, 0, 4, 1, 3])
    original = LedoitWolfLinearCovariance().fit(returns).forecast().matrix
    permuted = LedoitWolfLinearCovariance().fit(returns[:, permutation]).forecast().matrix
    np.testing.assert_allclose(permuted, original[np.ix_(permutation, permutation)])


def test_linear_no_lookahead_outside_supplied_window() -> None:
    window = _returns_with_mean(n_times=30)
    future = np.full((1, window.shape[1]), 99.0)
    first = LedoitWolfLinearCovariance().fit(window).forecast().matrix
    future[0, 0] = -50.0
    second = LedoitWolfLinearCovariance().fit(window).forecast().matrix
    np.testing.assert_array_equal(first, second)
    leaked = LedoitWolfLinearCovariance().fit(np.vstack([window, future])).forecast().matrix
    assert not np.allclose(leaked, first)


def test_linear_is_deterministic() -> None:
    returns = _returns_with_mean()
    first = LedoitWolfLinearCovariance().fit(returns)
    second = LedoitWolfLinearCovariance().fit(returns)
    np.testing.assert_array_equal(first.forecast().matrix, second.forecast().matrix)
    np.testing.assert_array_equal(first.forecast().matrix, first.forecast().matrix)
    assert first.fit_state.rho == second.fit_state.rho


def test_linear_output_does_not_alias_fit_state() -> None:
    returns = _returns_with_mean()
    model = LedoitWolfLinearCovariance().fit(returns)
    first = model.forecast().matrix
    first[0, 0] = -99.0
    second = model.forecast().matrix
    assert second[0, 0] != -99.0


def test_linear_singular_sample_is_strictly_pd_when_rho_positive() -> None:
    rng = _rng(11)
    n_times, n_assets = 6, 10
    factors = rng.standard_normal((n_times, 2))
    loadings = np.linspace(0.5, 3.0, n_assets)[:, None] * np.array([[1.0, 0.3]])
    returns = factors @ loadings.T + 0.15 * rng.standard_normal((n_times, n_assets))
    _mean, _centered, sample = _independent_sample_covariance(returns)
    assert sample.shape[0] > n_times - 1
    assert np.linalg.eigvalsh(sample).min() == pytest.approx(0.0, abs=1e-10)
    model = LedoitWolfLinearCovariance().fit(returns)
    assert model.fit_state.rho > 0.0
    assert model.fit_state.mu > 0.0
    forecast = model.forecast()
    assert forecast.diagnostics.positive_definite is True
    np.linalg.cholesky(forecast.matrix)


def test_nonlinear_matches_pinned_reference_on_same_centered_data() -> None:
    import nonlinshrink as nls

    returns = _returns_with_mean(n_times=40, n_assets=5)
    model = LedoitWolfNonlinearCovariance().fit(returns)
    _mean, centered, _sample = _independent_sample_covariance(returns)
    reference = np.asarray(nls.shrink_cov(centered, k=1), dtype=float)
    np.testing.assert_allclose(model.forecast().matrix, reference, atol=NL_ATOL)


def test_nonlinear_uses_same_sample_eigensystem_as_starting_point() -> None:
    returns = _returns_with_mean(n_times=40, n_assets=5)
    model = LedoitWolfNonlinearCovariance().fit(returns)
    _mean, centered, sample = _independent_sample_covariance(returns)
    eigenvalues, _eigenvectors = np.linalg.eigh(sample)
    np.testing.assert_allclose(model.fit_state.sample_eigenvalues, eigenvalues, atol=NL_ATOL)
    assert model.fit_state.n_eff == returns.shape[0] - 1
    np.testing.assert_allclose(
        sample,
        centered.T @ centered / (returns.shape[0] - 1),
    )


def test_nonlinear_retains_sample_eigenvectors_by_reconstruction() -> None:
    returns = _returns_with_mean(n_times=40, n_assets=5)
    forecast = LedoitWolfNonlinearCovariance().fit(returns).forecast().matrix
    _mean, _centered, sample = _independent_sample_covariance(returns)
    _assert_commutes(forecast, sample, NL_ATOL)
    _eigenvalues, eigenvectors = np.linalg.eigh(sample)
    projected = eigenvectors.T @ forecast @ eigenvectors
    off_diagonal = projected - np.diag(np.diag(projected))
    np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-6)
    np.testing.assert_allclose(
        forecast,
        eigenvectors @ np.diag(np.diag(projected)) @ eigenvectors.T,
        atol=1e-6,
    )


def test_nonlinear_eigenvalues_are_not_a_common_affine_map() -> None:
    returns = _returns_with_mean(n_times=50, n_assets=6, seed=7)
    model = LedoitWolfNonlinearCovariance().fit(returns)
    residual = _affine_residual(
        model.fit_state.sample_eigenvalues,
        model.fit_state.shrunk_eigenvalues,
    )
    assert np.max(np.abs(residual)) > 1e-4
    unique_sample = np.unique(np.round(model.fit_state.sample_eigenvalues, 8))
    assert unique_sample.size == model.fit_state.sample_eigenvalues.size


def test_nonlinear_forecast_is_symmetric() -> None:
    forecast = LedoitWolfNonlinearCovariance().fit(_returns_with_mean()).forecast()
    np.testing.assert_allclose(forecast.matrix, forecast.matrix.T, atol=1e-12)
    assert forecast.diagnostics.symmetry_error <= 1e-10


def test_nonlinear_is_strictly_pd_for_n_less_than_t() -> None:
    returns = _returns_with_mean(n_times=40, n_assets=5)
    assert returns.shape[1] < returns.shape[0]
    forecast = LedoitWolfNonlinearCovariance().fit(returns).forecast()
    assert forecast.diagnostics.positive_definite is True
    np.linalg.cholesky(forecast.matrix)


def test_nonlinear_n_at_least_t_branch_is_strictly_pd() -> None:
    returns = _returns_with_mean(n_times=20, n_assets=25, seed=21)
    assert returns.shape[1] >= returns.shape[0]
    forecast = LedoitWolfNonlinearCovariance().fit(returns).forecast()
    assert forecast.diagnostics.positive_definite is True
    np.linalg.cholesky(forecast.matrix)
    state = LedoitWolfNonlinearCovariance().fit(returns).fit_state
    assert state.n_assets >= state.n_observations


def test_nonlinear_rejects_short_windows_required_by_the_reference() -> None:
    returns = _returns_with_mean(n_times=12, n_assets=3)
    with pytest.raises(InvalidModelInputError, match="n_eff"):
        LedoitWolfNonlinearCovariance().fit(returns)


def test_nonlinear_permutation_equivariance() -> None:
    returns = _returns_with_mean(n_times=40, n_assets=5)
    permutation = np.array([4, 1, 3, 0, 2])
    original = LedoitWolfNonlinearCovariance().fit(returns).forecast().matrix
    permuted = LedoitWolfNonlinearCovariance().fit(returns[:, permutation]).forecast().matrix
    np.testing.assert_allclose(
        permuted,
        original[np.ix_(permutation, permutation)],
        atol=1e-8,
    )


def test_nonlinear_inputs_not_mutated_and_no_aliasing() -> None:
    returns = _returns_with_mean()
    original = returns.copy()
    model = LedoitWolfNonlinearCovariance().fit(returns)
    first = model.forecast().matrix
    first[0, 0] = -99.0
    np.testing.assert_array_equal(returns, original)
    second = model.forecast().matrix
    assert second[0, 0] != -99.0


def test_nonlinear_is_deterministic() -> None:
    returns = _returns_with_mean()
    first = LedoitWolfNonlinearCovariance().fit(returns).forecast().matrix
    second = LedoitWolfNonlinearCovariance().fit(returns).forecast().matrix
    np.testing.assert_array_equal(first, second)


def test_nonlinear_rejects_nonfinite_reference_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import covharness.models.ledoit_wolf as lw_module

    def _nonfinite(centered: np.ndarray, k: int | None = None) -> np.ndarray:
        del k
        return np.full((centered.shape[1], centered.shape[1]), np.nan)

    monkeypatch.setattr(lw_module, "_analytical_nonlinear_shrinkage", _nonfinite)
    with pytest.raises(InvalidModelForecastError, match="nonfinite"):
        LedoitWolfNonlinearCovariance().fit(_returns_with_mean())


def test_nonlinear_rejects_asymmetric_reference_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import covharness.models.ledoit_wolf as lw_module

    def _asymmetric(centered: np.ndarray, k: int | None = None) -> np.ndarray:
        del k
        n_assets = centered.shape[1]
        matrix = np.eye(n_assets)
        matrix[0, 1] = 0.4
        matrix[1, 0] = -0.1
        return matrix

    monkeypatch.setattr(lw_module, "_analytical_nonlinear_shrinkage", _asymmetric)
    with pytest.raises(InvalidModelForecastError, match="symmetric"):
        LedoitWolfNonlinearCovariance().fit(_returns_with_mean())


def test_nonlinear_rejects_indefinite_reference_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import covharness.models.ledoit_wolf as lw_module

    def _indefinite(centered: np.ndarray, k: int | None = None) -> np.ndarray:
        del k
        n_assets = centered.shape[1]
        matrix = np.eye(n_assets)
        matrix[0, 0] = -1.0
        return matrix

    monkeypatch.setattr(lw_module, "_analytical_nonlinear_shrinkage", _indefinite)
    with pytest.raises(InvalidModelForecastError, match="positive definite"):
        LedoitWolfNonlinearCovariance().fit(_returns_with_mean())


def test_linear_and_nonlinear_ladder_on_the_same_window() -> None:
    returns = _returns_with_mean(n_times=50, n_assets=6, seed=9)
    linear = LedoitWolfLinearCovariance().fit(returns)
    nonlinear = LedoitWolfNonlinearCovariance().fit(returns)
    _mean, _centered, sample = _independent_sample_covariance(returns)
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    np.testing.assert_allclose(linear.fit_state.sample_eigenvalues, eigenvalues)
    np.testing.assert_allclose(nonlinear.fit_state.sample_eigenvalues, eigenvalues)
    linear_matrix = linear.forecast().matrix
    nonlinear_matrix = nonlinear.forecast().matrix
    _assert_commutes(linear_matrix, sample, LINEAR_ATOL)
    _assert_commutes(nonlinear_matrix, sample, 1e-6)
    linear_map = (
        (1.0 - linear.fit_state.rho) * eigenvalues + linear.fit_state.rho * linear.fit_state.mu
    )
    np.testing.assert_allclose(
        _affine_residual(eigenvalues, linear_map),
        0.0,
        atol=AFFINE_ATOL,
    )
    nl_in_basis = np.diag(eigenvectors.T @ nonlinear_matrix @ eigenvectors)
    residual = _affine_residual(eigenvalues, nl_in_basis)
    assert np.max(np.abs(residual)) > 1e-4
    assert not np.allclose(linear_matrix, nonlinear_matrix)


def test_known_sigma_spectral_demo_is_deterministic() -> None:
    rng = _rng(20260916)
    n_times, n_assets = 60, 4
    true_eigenvalues = np.array([8.0, 3.0, 1.2, 0.4])
    orthogonal, _ = np.linalg.qr(rng.standard_normal((n_assets, n_assets)))
    true_sigma = orthogonal @ np.diag(true_eigenvalues) @ orthogonal.T
    returns = rng.multivariate_normal(
        np.array([0.1, -0.2, 0.05, 0.3]), true_sigma, size=n_times
    )
    _mean, _centered, sample = _independent_sample_covariance(returns)
    linear = LedoitWolfLinearCovariance().fit(returns)
    nonlinear = LedoitWolfNonlinearCovariance().fit(returns)
    sample_eigenvalues = np.linalg.eigvalsh(sample)
    linear_eigenvalues = np.linalg.eigvalsh(linear.forecast().matrix)
    nonlinear_eigenvalues = np.sort(np.asarray(nonlinear.fit_state.shrunk_eigenvalues))
    # One finite draw. Common linear map versus eigenvalue-specific NL map.
    linear_residual = _affine_residual(sample_eigenvalues, linear_eigenvalues)
    nonlinear_residual = _affine_residual(sample_eigenvalues, nonlinear_eigenvalues)
    np.testing.assert_allclose(linear_residual, 0.0, atol=1e-8)
    assert np.max(np.abs(nonlinear_residual)) > 1e-4
    repeat_linear = LedoitWolfLinearCovariance().fit(returns).forecast().matrix
    repeat_nonlinear = LedoitWolfNonlinearCovariance().fit(returns).forecast().matrix
    np.testing.assert_array_equal(repeat_linear, linear.forecast().matrix)
    np.testing.assert_array_equal(repeat_nonlinear, nonlinear.forecast().matrix)
    assert linear.identity.name == "lw_linear"
    assert nonlinear.identity.name == "lw_nl"
    assert linear.fit_state.centering is True
    assert nonlinear.fit_state.reference_package == "nonlinshrink"
    assert nonlinear.fit_state.reference_version == "0.7"
    assert nonlinear.fit_state.covariance_divisor == "T_minus_1"
    assert sample_eigenvalues[0] < sample_eigenvalues[-1]
    del true_sigma


def test_linear_identity_before_fit_has_no_window_metadata() -> None:
    identity = LedoitWolfLinearCovariance().identity
    assert identity.name == "lw_linear"
    assert "rho" not in identity.configuration
    assert identity.configuration["estimator"] == "ledoit_wolf_2004b_mu_identity"
