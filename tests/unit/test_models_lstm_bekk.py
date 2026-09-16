"""Synthetic tests for faithful daily-return LSTM-BEKK."""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from covharness.models import (
    CovarianceForecast,
    InvalidModelConfigurationError,
    InvalidModelForecastError,
    InvalidModelInputError,
    LSTMBEKKCovariance,
)
from covharness.models.lstm_bekk import (
    A0,
    B0,
    COVARIANCE_RESCALE,
    DEVICE_NAME,
    DTYPE_NAME,
    H0_DIAGONAL,
    H0_SAMPLE,
    INTERNAL_RETURN_SCALE,
    MODEL_NAME,
    OUTPUT_HEAD_BIAS,
    RMSPROP_ALPHA,
    RMSPROP_CENTERED,
    RMSPROP_EPS,
    RMSPROP_MOMENTUM,
    RMSPROP_WEIGHT_DECAY,
    SWISH_BETA0,
    TORCH_DEVICE,
    TORCH_DTYPE,
    W0,
    LSTMBEKKNetwork,
    bekk_recursion,
    construct_h0,
    gaussian_nll_cholesky_numpy,
    gaussian_nll_cholesky_torch,
    initialize_lstm_bekk_network,
    inverse_softplus,
    lstm_step_input,
    pack_dynamic_triangle,
    pack_lower_triangle,
    packed_static_c_from_h0,
    prepare_internal_returns,
    run_bekk_sequence,
    stationarity_logits_from_probabilities,
    swish,
    triangle_parameter_count,
    unpack_lower_triangle,
    unpack_static_c_numpy,
    unpack_static_c_torch,
)

SOURCE_PATH = Path(__file__).resolve().parents[2] / "src" / "covharness" / "models" / "lstm_bekk.py"
MODELS_INIT = Path(__file__).resolve().parents[2] / "src" / "covharness" / "models" / "__init__.py"


def _rng(seed: int = 20260916) -> np.random.Generator:
    return np.random.default_rng(seed)


def _returns(n_times: int = 12, n_assets: int = 3, seed: int = 20260916) -> np.ndarray:
    rng = _rng(seed)
    return rng.normal(loc=0.001, scale=0.01, size=(n_times, n_assets))


def _tiny_model(**overrides: object) -> LSTMBEKKCovariance:
    kwargs = dict(
        seed=0,
        num_layers=3,
        dropout=0.1,
        learning_rate=0.05,
        gradient_clip_norm=1.0,
        max_epochs=2,
    )
    kwargs.update(overrides)
    return LSTMBEKKCovariance(**kwargs)


def _source_text() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8")


def test_input_rejects_wrong_ndim() -> None:
    model = _tiny_model()
    with pytest.raises(InvalidModelInputError, match="shape"):
        model.fit(np.zeros(5))


def test_input_rejects_t_equals_one() -> None:
    model = _tiny_model()
    with pytest.raises(InvalidModelInputError, match="two observations"):
        model.fit(np.ones((1, 3)))


def test_input_rejects_single_asset() -> None:
    model = _tiny_model()
    with pytest.raises(InvalidModelInputError, match="two assets"):
        model.fit(np.ones((5, 1)))


def test_input_rejects_nonfinite() -> None:
    model = _tiny_model()
    bad = _returns()
    bad[0, 0] = np.nan
    with pytest.raises(InvalidModelInputError, match="finite"):
        model.fit(bad)


def test_inputs_are_not_mutated() -> None:
    model = _tiny_model()
    returns = _returns()
    original = returns.copy()
    model.fit(returns)
    np.testing.assert_array_equal(returns, original)


def test_fit_mean_is_exact() -> None:
    model = _tiny_model()
    returns = _returns()
    model.fit(returns)
    np.testing.assert_allclose(model._fit.fit_mean, returns.mean(axis=0))


def test_internal_residual_scale_is_exactly_100() -> None:
    returns = _returns()
    fit_mean, native, internal = prepare_internal_returns(returns)
    np.testing.assert_allclose(native, returns - fit_mean)
    np.testing.assert_allclose(internal, 100.0 * native)
    assert INTERNAL_RETURN_SCALE == 100.0


def test_forecast_is_rescaled_by_10000() -> None:
    model = _tiny_model()
    model.fit(_returns())
    forecast = model.forecast()
    np.testing.assert_allclose(
        forecast.matrix,
        model._fit.current_forecast_internal / COVARIANCE_RESCALE,
    )
    assert COVARIANCE_RESCALE == 10000.0


def test_packed_static_c_reconstructs_lower_triangle() -> None:
    matrix = np.array([[1.0, 0.0], [0.3, 2.0]])
    packed = pack_lower_triangle(matrix)
    rebuilt = unpack_lower_triangle(packed, 2)
    np.testing.assert_allclose(rebuilt, matrix)


def test_static_c_diagonal_uses_softplus_and_is_positive() -> None:
    packed = np.array([-2.0, 0.4, 1.5])
    reconstructed = unpack_static_c_numpy(packed, 2)
    assert reconstructed[0, 0] > 0.0
    assert reconstructed[1, 1] > 0.0
    assert reconstructed[0, 1] == 0.0
    np.testing.assert_allclose(reconstructed[1, 0], 0.4)
    tensor = unpack_static_c_torch(torch.tensor(packed, dtype=TORCH_DTYPE), 2)
    np.testing.assert_allclose(tensor.detach().numpy(), reconstructed, rtol=1e-10, atol=1e-10)


def test_inverse_softplus_initialization_round_trip() -> None:
    target = np.array([0.2, 1.7, 3.4])
    raw = inverse_softplus(target)
    recovered = np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0.0)
    np.testing.assert_allclose(recovered, target, rtol=1e-12, atol=1e-12)


def test_initial_c_matches_w0_times_h0() -> None:
    returns = _returns()
    _mean, _native, internal = prepare_internal_returns(returns)
    h0, method = construct_h0(internal)
    assert method == H0_SAMPLE
    packed = packed_static_c_from_h0(h0, W0)
    static_c = unpack_static_c_numpy(packed, internal.shape[1])
    np.testing.assert_allclose(static_c @ static_c.T, W0 * h0, rtol=1e-10, atol=1e-10)


def test_h0_sample_covariance_branch() -> None:
    rng = _rng(1)
    internal = rng.normal(size=(20, 3))
    h0, method = construct_h0(internal)
    expected = internal.T @ internal / (internal.shape[0] - 1)
    assert method == H0_SAMPLE
    np.testing.assert_allclose(h0, expected)


def test_h0_diagonal_fallback_for_singular_sample() -> None:
    # T=2, N=3 makes S0 rank-at-most-1 while keeping positive column variances.
    internal = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 4.0]], dtype=float)
    h0, method = construct_h0(internal)
    sample = internal.T @ internal / (internal.shape[0] - 1)
    assert method == H0_DIAGONAL
    np.testing.assert_allclose(h0, np.diag(np.diag(sample)))
    assert np.all(np.diag(h0) > 0.0)


def test_zero_variance_asset_is_rejected() -> None:
    returns = _returns()
    returns[:, 1] = 0.3
    model = _tiny_model()
    with pytest.raises(InvalidModelInputError, match="positive centered variance"):
        model.fit(returns)


def test_lstm_hidden_size_equals_n() -> None:
    model = _tiny_model()
    returns = _returns(n_assets=4)
    model.fit(returns)
    assert model._network.lstm.hidden_size == 4
    assert model._fit.hidden_size == 4


def test_num_layers_outside_range_is_rejected() -> None:
    with pytest.raises(InvalidModelConfigurationError, match="num_layers"):
        LSTMBEKKCovariance(
            seed=0,
            num_layers=2,
            dropout=0.1,
            learning_rate=0.01,
            gradient_clip_norm=1.0,
            max_epochs=1,
        )
    with pytest.raises(InvalidModelConfigurationError, match="num_layers"):
        _tiny_model(num_layers=6)


def test_dropout_outside_interval_is_rejected() -> None:
    with pytest.raises(InvalidModelConfigurationError, match="dropout"):
        _tiny_model(dropout=0.09)
    with pytest.raises(InvalidModelConfigurationError, match="dropout"):
        _tiny_model(dropout=0.21)


def test_output_head_width_and_bias() -> None:
    model = _tiny_model()
    returns = _returns(n_assets=3)
    model.fit(returns)
    n_triangle = triangle_parameter_count(3)
    assert model._network.triangle_head.out_features == n_triangle
    assert model._network.triangle_head.in_features == 3
    assert model._network.triangle_head.bias is not None
    assert OUTPUT_HEAD_BIAS is True
    assert model._fit.n_triangle == n_triangle
    assert model._fit.output_head_bias is True


def test_dynamic_triangle_packing_and_swish_only_on_diagonal() -> None:
    n_assets = 3
    raw = torch.tensor([-0.4, 0.2, -1.1, 0.5, 0.7, 0.3], dtype=TORCH_DTYPE)
    beta = torch.tensor(1.25, dtype=TORCH_DTYPE)
    dynamic = pack_dynamic_triangle(raw, n_assets, beta)
    unpacked = unpack_lower_triangle(raw.numpy(), n_assets)
    np.testing.assert_allclose(dynamic[1, 0].item(), unpacked[1, 0])
    np.testing.assert_allclose(dynamic[2, 0].item(), unpacked[2, 0])
    np.testing.assert_allclose(dynamic[2, 1].item(), unpacked[2, 1])
    expected_diag = swish(torch.diag(torch.tensor(unpacked, dtype=TORCH_DTYPE)), beta)
    np.testing.assert_allclose(torch.diag(dynamic).detach().numpy(), expected_diag.detach().numpy())
    assert dynamic[0, 1].item() == 0.0


def test_global_beta_initialized_to_one() -> None:
    network = LSTMBEKKNetwork(3, 3, 0.1)
    initialize_lstm_bekk_network(network, packed_static_c_from_h0(np.eye(3), W0))
    assert float(network.swish_beta.detach()) == pytest.approx(SWISH_BETA0)
    assert SWISH_BETA0 == 1.0


def test_dynamic_gram_is_psd_with_negative_diagonal() -> None:
    n_assets = 2
    raw = torch.tensor([-2.0, 0.3, -0.8], dtype=TORCH_DTYPE)
    beta = torch.tensor(1.0, dtype=TORCH_DTYPE)
    dynamic = pack_dynamic_triangle(raw, n_assets, beta)
    assert float(dynamic[0, 0]) < 0.0
    gram = (dynamic @ dynamic.T).detach().numpy()
    eigenvalues = np.linalg.eigvalsh(gram)
    assert np.all(eigenvalues >= -1e-12)


def test_softmax_logits_recover_initial_stationarity() -> None:
    logits = stationarity_logits_from_probabilities()
    weights = np.exp(logits) / np.sum(np.exp(logits))
    np.testing.assert_allclose(weights, [W0, A0, B0])
    model = _tiny_model(max_epochs=1)
    model.fit(_returns())
    # After one epoch the values may move, but initialization used these logits.
    init_logits = stationarity_logits_from_probabilities()
    np.testing.assert_allclose(np.exp(init_logits) / np.exp(init_logits).sum(), [0.05, 0.05, 0.90])


def test_learned_a_b_stay_in_open_simplex() -> None:
    model = _tiny_model(max_epochs=3)
    model.fit(_returns())
    assert model._fit.bekk_a > 0.0
    assert model._fit.bekk_b > 0.0
    assert model._fit.bekk_a + model._fit.bekk_b < 1.0
    assert model._fit.stationarity_w > 0.0
    assert model._fit.stationarity_w + model._fit.bekk_a + model._fit.bekk_b == pytest.approx(1.0)


def test_recursion_helper_a_zero_and_b_zero() -> None:
    static_c = np.array([[0.2, 0.0], [0.1, 0.3]])
    dynamic_c = np.array([[0.05, 0.0], [-0.02, 0.04]])
    shock = np.array([0.3, -0.1])
    h_prev = np.array([[0.8, 0.1], [0.1, 0.6]])
    a_zero = bekk_recursion(static_c, dynamic_c, 0.0, 0.4, shock, h_prev)
    expected_a = static_c @ static_c.T + dynamic_c @ dynamic_c.T + 0.4 * h_prev
    np.testing.assert_allclose(a_zero, expected_a)
    b_zero = bekk_recursion(static_c, dynamic_c, 0.2, 0.0, shock, h_prev)
    expected_b = static_c @ static_c.T + dynamic_c @ dynamic_c.T + 0.2 * np.outer(shock, shock)
    np.testing.assert_allclose(b_zero, expected_b)


def test_one_step_recursion_matches_numpy_hand_calculation() -> None:
    static_c = np.array([[0.25, 0.0, 0.0], [0.05, 0.2, 0.0], [-0.01, 0.03, 0.18]])
    dynamic_c = np.array([[0.04, 0.0, 0.0], [0.02, -0.03, 0.0], [0.01, 0.00, 0.05]])
    shock = np.array([0.2, -0.4, 0.1])
    h_prev = np.eye(3)
    got = bekk_recursion(static_c, dynamic_c, 0.07, 0.8, shock, h_prev)
    hand = (
        static_c @ static_c.T
        + dynamic_c @ dynamic_c.T
        + 0.07 * np.outer(shock, shock)
        + 0.8 * h_prev
    )
    np.testing.assert_allclose(got, hand)


def test_dynamic_c_zero_reduces_to_scalar_bekk() -> None:
    static_c = np.array([[0.3, 0.0], [0.1, 0.2]])
    shock = np.array([0.5, -0.2])
    h_prev = np.array([[1.0, 0.2], [0.2, 0.8]])
    zero_dynamic = np.zeros((2, 2))
    got = bekk_recursion(static_c, zero_dynamic, 0.1, 0.7, shock, h_prev)
    expected = static_c @ static_c.T + 0.1 * np.outer(shock, shock) + 0.7 * h_prev
    np.testing.assert_allclose(got, expected)


def test_valid_components_produce_strictly_pd_h() -> None:
    static_c = np.array([[0.4, 0.0], [0.1, 0.3]])
    dynamic_c = np.array([[-0.2, 0.0], [0.05, 0.1]])
    h = bekk_recursion(static_c, dynamic_c, 0.05, 0.9, np.array([0.2, 0.1]), np.eye(2))
    np.linalg.cholesky(h)


def test_no_jitter_or_repair_helper_exists() -> None:
    source = _source_text()
    for token in ("jitter", "nearest_pd", "cov_nearest", "eigenvalue_floor", "diagonal_load"):
        assert token not in source


def test_cholesky_nll_matches_dense_small_spd() -> None:
    covariance = np.array([[1.5, 0.2], [0.2, 0.9]])
    residual = np.array([0.3, -0.4])
    chol = gaussian_nll_cholesky_numpy(covariance, residual)
    _sign, logdet = np.linalg.slogdet(covariance)
    precision = np.linalg.inv(covariance)
    dense = 0.5 * (float(logdet) + float(residual @ precision @ residual))
    torch_value = gaussian_nll_cholesky_torch(
        torch.tensor(covariance, dtype=TORCH_DTYPE),
        torch.tensor(residual, dtype=TORCH_DTYPE),
    )
    assert chol == pytest.approx(dense, rel=1e-12, abs=1e-12)
    assert float(torch_value) == pytest.approx(chol, rel=1e-12, abs=1e-12)


def test_production_nll_does_not_call_inverse() -> None:
    source = _source_text()
    assert "torch.inverse" not in source
    assert "torch.linalg.inv" not in source
    nll_fn = inspect.getsource(gaussian_nll_cholesky_torch)
    assert "inv(" not in nll_fn
    assert "cholesky" in nll_fn


def test_failed_cholesky_surfaces_clearly() -> None:
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(InvalidModelForecastError, match="Cholesky"):
        gaussian_nll_cholesky_numpy(bad, np.array([0.1, 0.2]))
    with pytest.raises(InvalidModelForecastError, match="Cholesky"):
        gaussian_nll_cholesky_torch(
            torch.tensor(bad, dtype=TORCH_DTYPE),
            torch.tensor([0.1, 0.2], dtype=TORCH_DTYPE),
        )


def test_omitted_n_log_2pi_is_the_known_constant() -> None:
    covariance = np.array([[1.2, 0.1], [0.1, 0.8]])
    residual = np.array([0.2, 0.3])
    reduced = gaussian_nll_cholesky_numpy(covariance, residual)
    full = reduced + 0.5 * 2 * np.log(2.0 * np.pi)
    assert full - reduced == pytest.approx(0.5 * 2 * np.log(2.0 * np.pi))


def test_autograd_gradients_are_finite_on_n3() -> None:
    n_assets = 3
    apply_seed = 7
    random.seed(apply_seed)
    np.random.seed(apply_seed)
    torch.manual_seed(apply_seed)
    network = LSTMBEKKNetwork(n_assets, 3, 0.1)
    packed = packed_static_c_from_h0(np.eye(n_assets), W0)
    initialize_lstm_bekk_network(network, packed)
    internal = torch.tensor(_returns(n_times=8, n_assets=3, seed=3), dtype=TORCH_DTYPE)
    h0 = torch.eye(n_assets, dtype=TORCH_DTYPE)
    nll = run_bekk_sequence(network, internal, h0).nll
    nll.backward()
    for parameter in network.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_identical_seeds_initialize_identical_cpu_models() -> None:
    returns = _returns(seed=11)
    first = _tiny_model(seed=4, max_epochs=1)
    second = _tiny_model(seed=4, max_epochs=1)
    first.fit(returns)
    second.fit(returns)
    np.testing.assert_allclose(first._fit.packed_static_c, second._fit.packed_static_c)
    np.testing.assert_allclose(first._fit.stationarity_logits, second._fit.stationarity_logits)
    for (name_a, param_a), (name_b, param_b) in zip(
        first._network.named_parameters(), second._network.named_parameters(), strict=True
    ):
        assert name_a == name_b
        np.testing.assert_allclose(param_a.detach().numpy(), param_b.detach().numpy())


def test_different_seeds_can_differ() -> None:
    returns = _returns(seed=11)
    first = _tiny_model(seed=0, max_epochs=1)
    second = _tiny_model(seed=1, max_epochs=1)
    first.fit(returns)
    second.fit(returns)
    assert not np.allclose(
        first._network.triangle_head.weight.detach().numpy(),
        second._network.triangle_head.weight.detach().numpy(),
    )


def test_declared_stochastic_sources_are_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def _random_seed(seed: int) -> None:
        called.append(f"random:{seed}")
        return original_random(seed)

    def _numpy_seed(seed: int) -> None:
        called.append(f"numpy:{seed}")
        return original_numpy(seed)

    def _torch_seed(seed: int) -> None:
        called.append(f"torch:{seed}")
        return original_torch(seed)

    original_random = random.seed
    original_numpy = np.random.seed
    original_torch = torch.manual_seed
    monkeypatch.setattr(random, "seed", _random_seed)
    monkeypatch.setattr(np.random, "seed", _numpy_seed)
    monkeypatch.setattr(torch, "manual_seed", _torch_seed)
    model = _tiny_model(seed=9, max_epochs=1)
    model.fit(_returns())
    assert "random:9" in called
    assert "numpy:9" in called
    assert "torch:9" in called


def test_training_uses_float64_cpu() -> None:
    model = _tiny_model(max_epochs=1)
    model.fit(_returns())
    assert model._fit.dtype_name == DTYPE_NAME
    assert model._fit.device_name == DEVICE_NAME
    assert next(model._network.parameters()).dtype == TORCH_DTYPE
    assert next(model._network.parameters()).device.type == TORCH_DEVICE.type
    assert not next(model._network.parameters()).is_cuda


def test_rmsprop_configuration_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    original = torch.optim.RMSprop

    def _capture(params, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return original(params, **kwargs)

    monkeypatch.setattr(torch.optim, "RMSprop", _capture)
    model = _tiny_model(learning_rate=0.03, max_epochs=1)
    model.fit(_returns())
    assert captured["lr"] == 0.03
    assert captured["alpha"] == RMSPROP_ALPHA
    assert captured["eps"] == RMSPROP_EPS
    assert captured["momentum"] == RMSPROP_MOMENTUM
    assert captured["centered"] == RMSPROP_CENTERED
    assert captured["weight_decay"] == RMSPROP_WEIGHT_DECAY
    assert model._fit.optimizer_name == "RMSprop"


def test_gradient_clipping_uses_constructor_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}
    original = torch.nn.utils.clip_grad_norm_

    def _capture(parameters, max_norm, **kwargs):  # type: ignore[no-untyped-def]
        captured["max_norm"] = float(max_norm)
        return original(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _capture)
    model = _tiny_model(gradient_clip_norm=1.7, max_epochs=1)
    model.fit(_returns())
    assert captured["max_norm"] == pytest.approx(1.7)


def test_full_bptt_has_no_detach_in_training_recursion() -> None:
    source = inspect.getsource(run_bekk_sequence)
    assert ".detach(" not in source
    assert "detach(" not in source
    model = _tiny_model(max_epochs=1)
    model.fit(_returns())
    assert model._fit.full_bptt is True


def test_no_paper_split_and_no_early_stopping() -> None:
    source = _source_text()
    assert "70/15/15" not in source
    assert "early_stopping" in source
    model = _tiny_model(max_epochs=1)
    model.fit(_returns())
    assert model._fit.early_stopping is False


def test_training_runs_exactly_max_epochs() -> None:
    model = _tiny_model(max_epochs=3)
    model.fit(_returns())
    assert len(model._fit.training_loss_history) == 3
    assert model._fit.max_epochs == 3


def test_tiny_training_losses_are_finite() -> None:
    model = _tiny_model(max_epochs=2)
    model.fit(_returns(n_times=10, n_assets=3, seed=5))
    assert np.isfinite(model._fit.initial_training_nll)
    assert np.isfinite(model._fit.final_training_nll)
    assert np.all(np.isfinite(model._fit.training_loss_history))


def test_controlled_example_decreases_nll() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(scale=0.01, size=(10, 3))
    _mean, _native, internal = prepare_internal_returns(returns)
    h0, _method = construct_h0(internal)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    network = LSTMBEKKNetwork(3, 3, 0.1)
    initialize_lstm_bekk_network(network, packed_static_c_from_h0(h0, W0))
    x_tensor = torch.tensor(internal, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    h0_tensor = torch.tensor(h0, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    optimizer = torch.optim.RMSprop(
        network.parameters(),
        lr=0.05,
        alpha=RMSPROP_ALPHA,
        eps=RMSPROP_EPS,
        momentum=RMSPROP_MOMENTUM,
        centered=RMSPROP_CENTERED,
        weight_decay=RMSPROP_WEIGHT_DECAY,
    )
    # Dropout-off evaluation of the same objective before and after steps.
    network.eval()
    initial = float(run_bekk_sequence(network, x_tensor, h0_tensor).nll.detach())
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        nll = run_bekk_sequence(network, x_tensor, h0_tensor).nll
        nll.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
        optimizer.step()
    final = float(run_bekk_sequence(network, x_tensor, h0_tensor).nll.detach())
    assert np.isfinite(initial) and np.isfinite(final)
    assert final < initial


def test_first_return_scored_under_h0_and_all_t_counted() -> None:
    model = _tiny_model(max_epochs=1)
    returns = _returns(n_times=9, n_assets=3)
    model.fit(returns)
    assert model._fit.n_likelihood_terms == 9
    _mean, _native, internal = prepare_internal_returns(returns)
    h0, _method = construct_h0(internal)
    np.testing.assert_allclose(model._fit.h0, h0)
    source = inspect.getsource(run_bekk_sequence)
    assert "internal_returns[0]" in source
    assert "presample" not in _source_text()


def test_forecast_is_h_t_over_10000_and_not_in_likelihood() -> None:
    model = _tiny_model(max_epochs=1)
    returns = _returns()
    model.fit(returns)
    source = inspect.getsource(run_bekk_sequence)
    assert "nll + gaussian_nll_cholesky_torch(current_h, internal_returns[time_index + 1])" in source
    assert "forecast_h" in source
    assert "nll + gaussian_nll_cholesky_torch(forecast_h" not in source
    forecast = model.forecast()
    np.testing.assert_allclose(
        forecast.matrix, model._fit.current_forecast_internal / 10000.0
    )


def test_forecast_does_not_mutate_and_repeats() -> None:
    model = _tiny_model(max_epochs=1)
    model.fit(_returns())
    hidden_before = model._fit.hidden_state.copy()
    cell_before = model._fit.cell_state.copy()
    h_before = model._fit.current_forecast_internal.copy()
    first = model.forecast().matrix.copy()
    second = model.forecast().matrix.copy()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(model._fit.hidden_state, hidden_before)
    np.testing.assert_array_equal(model._fit.cell_state, cell_before)
    np.testing.assert_array_equal(model._fit.current_forecast_internal, h_before)


def test_update_uses_current_forecast_and_frozen_mean() -> None:
    model = _tiny_model(max_epochs=1)
    returns = _returns()
    model.fit(returns)
    frozen_mean = model._fit.fit_mean.copy()
    packed = model._fit.packed_static_c.copy()
    a_before = model._fit.bekk_a
    hidden_before = model._fit.hidden_state.copy()
    h_before = model._fit.current_forecast_internal.copy()
    new_return = returns[-1] + 0.01
    model.update(new_return)
    np.testing.assert_array_equal(model._fit.fit_mean, frozen_mean)
    np.testing.assert_array_equal(model._fit.packed_static_c, packed)
    assert model._fit.bekk_a == a_before
    assert not np.allclose(model._fit.hidden_state, hidden_before)
    assert not np.allclose(model._fit.current_forecast_internal, h_before)
    expected_internal = 100.0 * (new_return - frozen_mean)
    np.testing.assert_allclose(model._fit.last_observed_internal_return, expected_internal)


def test_update_advances_one_step_and_forecast_moves_forward() -> None:
    model = _tiny_model(max_epochs=1)
    returns = _returns()
    model.fit(returns)
    before = model.forecast().matrix.copy()
    model.update(returns[-1] + 0.02)
    after = model.forecast().matrix.copy()
    assert after.shape == before.shape
    assert not np.allclose(before, after)
    again = model.forecast().matrix.copy()
    np.testing.assert_array_equal(after, again)


def test_twenty_updates_do_not_retrain() -> None:
    model = _tiny_model(max_epochs=1)
    returns = _returns(n_times=12, seed=8)
    model.fit(returns)
    packed = model._fit.packed_static_c.copy()
    head = model._network.triangle_head.weight.detach().clone()
    rng = _rng(8)
    for _ in range(20):
        model.update(rng.normal(scale=0.01, size=3))
    np.testing.assert_array_equal(model._fit.packed_static_c, packed)
    np.testing.assert_allclose(
        model._network.triangle_head.weight.detach().numpy(), head.numpy()
    )


def test_refit_resets_recurrent_state() -> None:
    model = _tiny_model(max_epochs=1)
    first = _returns(seed=1)
    second = _returns(seed=2)
    model.fit(first)
    hidden_after_first = model._fit.hidden_state.copy()
    model.fit(second)
    # A new fit rebuilds the network from the constructor seed and zero state.
    assert model._fit.hidden_state.shape == hidden_after_first.shape
    source = inspect.getsource(run_bekk_sequence)
    assert "zero_recurrent_state" in source


def test_target_day_cannot_enter_before_update() -> None:
    model = _tiny_model(max_epochs=1)
    returns = _returns()
    model.fit(returns)
    forecast_before = model.forecast().matrix.copy()
    held_out = np.array([10.0, -10.0, 5.0])
    forecast_again = model.forecast().matrix.copy()
    np.testing.assert_array_equal(forecast_before, forecast_again)
    model.update(held_out)
    forecast_after = model.forecast().matrix.copy()
    assert not np.allclose(forecast_before, forecast_after)


def test_fit_signature_is_daily_returns_only() -> None:
    signature = inspect.signature(LSTMBEKKCovariance.fit)
    assert list(signature.parameters) == ["self", "returns"]
    source = inspect.getsource(LSTMBEKKCovariance.fit)
    assert "realized" not in source
    assert "quarticity" not in source


def test_rc_seam_exists_but_is_unused() -> None:
    model = _tiny_model(max_epochs=1)
    vector = torch.tensor([0.1, -0.2, 0.3], dtype=TORCH_DTYPE)
    np.testing.assert_allclose(model.build_lstm_input(vector).numpy(), vector.numpy())
    np.testing.assert_allclose(lstm_step_input(vector).numpy(), vector.numpy())
    source = _source_text()
    assert "origin-measurable" in source
    assert "class LSTMBEKKRC" not in source
    assert "xgboost" not in source.lower()
    init_text = MODELS_INIT.read_text(encoding="utf-8")
    assert "LSTMBEKKRC" not in init_text
    assert "XGBoostDRDRealizedCovariance" in init_text


def test_identity_is_lstm_bekk() -> None:
    model = _tiny_model()
    assert model.identity.name == MODEL_NAME
    assert MODEL_NAME == "lstm_bekk"
    model.fit(_returns())
    forecast = model.forecast()
    assert isinstance(forecast, CovarianceForecast)
    assert forecast.identity.name == "lstm_bekk"
    assert forecast.diagnostics.positive_definite is True


def test_non_pd_forecast_raises() -> None:
    model = _tiny_model(max_epochs=1)
    model.fit(_returns())
    from covharness.models.lstm_bekk import LSTMBEKKFitState

    broken = LSTMBEKKFitState(
        **{
            **model._fit.__dict__,
            "current_forecast_internal": np.array([[1.0, 2.0], [2.0, 1.0]]),
        }
    )
    model._fit = broken
    with pytest.raises(InvalidModelForecastError, match="positive definite"):
        model.forecast()


def test_moderate_dimension_forward_only() -> None:
    rng = _rng(21)
    returns = rng.normal(scale=0.01, size=(8, 20))
    model = _tiny_model(max_epochs=1, learning_rate=0.01)
    model.fit(returns)
    forecast = model.forecast()
    assert forecast.matrix.shape == (20, 20)
    assert model._fit.n_triangle == triangle_parameter_count(20)
    assert model._network.triangle_head.out_features == triangle_parameter_count(20)
    np.linalg.cholesky(forecast.matrix)
    assert forecast.diagnostics.positive_definite is True
