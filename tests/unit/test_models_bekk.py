"""BEKK recursion, stationarity, QMLE recovery, and the zero-intercept ablation."""

from __future__ import annotations

import numpy as np
import torch

from covharness.models.bekk import (
    LSTMBEKKNoDynamicC,
    bekk_transition,
    matrix_parameters_stationary,
)
from covharness.models.lstm_bekk import (
    LSTMBEKKNetwork,
    TORCH_DEVICE,
    TORCH_DTYPE,
    apply_cpu_seeds,
    bekk_recursion,
    construct_h0,
    initialize_lstm_bekk_network,
    packed_static_c_from_h0,
    prepare_internal_returns,
    run_bekk_sequence,
    unpack_static_c_numpy,
)


def test_scalar_step_matches_a_hand_computed_update() -> None:
    intercept = np.eye(2)
    shock = np.array([1.0, 0.0])
    previous = np.eye(2)
    got = bekk_transition(intercept, 0.1 * np.eye(2), 0.2 * np.eye(2), shock, previous, "scalar")
    expected = np.array([[1.3, 0.0], [0.0, 1.2]])
    assert np.allclose(got, expected, rtol=0.0, atol=1e-12)
    direct = bekk_recursion(np.eye(2), np.zeros((2, 2)), 0.1, 0.2, shock, previous)
    assert np.allclose(got, direct, rtol=0.0, atol=1e-12)


def test_diagonal_and_full_steps_stay_positive_definite() -> None:
    rng = np.random.default_rng(1)
    shock = rng.normal(size=3)
    previous = np.eye(3)
    intercept = 0.4 * np.eye(3)
    diagonal = bekk_transition(
        intercept, np.diag([0.2, 0.1, 0.3]), np.diag([0.4, 0.5, 0.2]), shock, previous, "diagonal"
    )
    full_a = 0.05 * np.eye(3)
    full_b = 0.2 * np.eye(3)
    full = bekk_transition(intercept, full_a, full_b, shock, previous, "full")
    np.linalg.cholesky(diagonal)
    np.linalg.cholesky(full)
    assert matrix_parameters_stationary(np.diag([0.2, 0.1, 0.3]), np.diag([0.4, 0.5, 0.2]), "diagonal")
    assert matrix_parameters_stationary(full_a, full_b, "full")


def test_scalar_qmle_recovers_the_simulated_coefficients() -> None:
    rng = np.random.default_rng(5)
    a_true, b_true = 0.05, 0.80
    unconditional = np.array([[1.0, 0.2], [0.2, 0.8]])
    intercept = (1.0 - a_true - b_true) * unconditional
    factor = np.linalg.cholesky(intercept)
    current = np.array(unconditional, dtype=float, copy=True)
    shocks = []
    # Covariance targeting. The estimator sees the same scalar recursion.
    for _ in range(1500):
        shock = rng.multivariate_normal(np.zeros(2), current)
        shocks.append(shock)
        current = bekk_recursion(factor, np.zeros((2, 2)), a_true, b_true, shock, current)
    from covharness.models.bekk import BEKKCovariance
    from covharness.models.stationarity import POLICY_CARRY_FORWARD

    model = BEKKCovariance(
        form="scalar",
        targeting=True,
        failure_policy=POLICY_CARRY_FORWARD,
        specification_id="BEKK-TEST",
    )
    model.fit(np.asarray(shocks) / 100.0)
    assert abs(model.identity.configuration["scalar_a"] - a_true) < 0.05
    assert abs(model.identity.configuration["scalar_b"] - b_true) < 0.05


def test_zero_dynamic_intercept_matches_the_scalar_recursion() -> None:
    returns = np.random.default_rng(3).normal(0.0, 0.01, size=(6, 2))
    _mean, _native, internal = prepare_internal_returns(returns)
    h0, _method = construct_h0(internal)
    apply_cpu_seeds(0)
    network = LSTMBEKKNetwork(2, 3, 0.0)
    initialize_lstm_bekk_network(network, packed_static_c_from_h0(h0))
    x_tensor = torch.tensor(internal, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    h0_tensor = torch.tensor(h0, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    path = run_bekk_sequence(network, x_tensor, h0_tensor, zero_dynamic_c=True)
    static_c = unpack_static_c_numpy(network.packed_static_c.detach().cpu().numpy(), 2)
    _w, a_value, b_value = network.stationarity_weights()
    current = np.array(h0, dtype=float, copy=True)
    for time_index in range(internal.shape[0]):
        current = bekk_recursion(
            static_c,
            np.zeros((2, 2)),
            float(a_value.detach()),
            float(b_value.detach()),
            internal[time_index],
            current,
        )
    assert np.allclose(path.forecast_internal.detach().cpu().numpy(), current, rtol=0.0, atol=1e-10)
    ablation = LSTMBEKKNoDynamicC(
        seed=0,
        num_layers=3,
        dropout=0.0,
        learning_rate=0.001,
        gradient_clip_norm=1.0,
        max_epochs=1,
    )
    assert ablation._zero_dynamic_c is True
