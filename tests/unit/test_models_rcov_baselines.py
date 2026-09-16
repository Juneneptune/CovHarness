"""Synthetic tests for random-walk and EWMA realized-covariance baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covharness.models import (
    EWMARealizedCovariance,
    InvalidModelConfigurationError,
    InvalidModelInputError,
    RandomWalkRealizedCovariance,
)
from covharness.protocol import (
    BlockName,
    ConfirmLockedError,
    build_temporal_protocol,
)

SPD_A = np.array([[2.0, 0.4], [0.4, 1.0]])
SPD_B = np.array([[1.5, 0.2], [0.2, 1.2]])
SPD_C = np.array([[3.0, -0.1], [-0.1, 2.5]])
SINGULAR_PSD = np.array([[1.0, 1.0], [1.0, 1.0]])
# Common null direction e_2. Scales vary; the second axis stays exactly zero.
NULL_SHARED = np.stack(
    [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[4.0, 0.0], [0.0, 0.0]]),
        np.array([[9.0, 0.0], [0.0, 0.0]]),
    ]
)


def _history(*matrices: np.ndarray) -> np.ndarray:
    return np.stack(matrices, axis=0)


def test_rw_shape_and_asset_order() -> None:
    history = _history(SPD_A, SPD_B)
    forecast = RandomWalkRealizedCovariance().fit(history).forecast()
    assert forecast.matrix.shape == (2, 2)
    np.testing.assert_allclose(forecast.matrix, SPD_B)
    assert forecast.matrix[0, 0] == SPD_B[0, 0]
    assert forecast.matrix[1, 1] == SPD_B[1, 1]


def test_rw_ignores_earlier_matrices() -> None:
    first = RandomWalkRealizedCovariance().fit(_history(SPD_A, SPD_C)).forecast()
    second = RandomWalkRealizedCovariance().fit(_history(SPD_B, SPD_C)).forecast()
    np.testing.assert_allclose(first.matrix, second.matrix)
    np.testing.assert_allclose(first.matrix, SPD_C)


def test_inputs_not_mutated_and_output_does_not_alias() -> None:
    history = _history(SPD_A, SPD_B)
    original = history.copy()
    rw = RandomWalkRealizedCovariance().fit(history).forecast()
    ewma = EWMARealizedCovariance(decay=0.5).fit(history).forecast()
    np.testing.assert_array_equal(history, original)
    assert rw.matrix.base is None or rw.matrix.base is not history[-1]
    assert not np.shares_memory(rw.matrix, history)
    assert not np.shares_memory(ewma.matrix, history)
    rw.matrix[0, 0] = -99.0
    ewma.matrix[0, 0] = -99.0
    np.testing.assert_array_equal(history, original)


def test_deterministic_repeated_calls() -> None:
    history = _history(SPD_A, SPD_B, SPD_C)
    rw = RandomWalkRealizedCovariance()
    ewma = EWMARealizedCovariance(decay=0.94)
    first_rw = rw.fit(history).forecast().matrix
    second_rw = rw.forecast().matrix
    first_ewma = ewma.fit(history).forecast().matrix
    second_ewma = ewma.forecast().matrix
    np.testing.assert_array_equal(first_rw, second_rw)
    np.testing.assert_array_equal(first_ewma, second_ewma)
    np.testing.assert_array_equal(
        RandomWalkRealizedCovariance().fit(history).forecast().matrix,
        first_rw,
    )
    np.testing.assert_array_equal(
        EWMARealizedCovariance(decay=0.94).fit(history).forecast().matrix,
        first_ewma,
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 2)),
        np.ones((0, 2, 2)),
        np.ones((2, 2, 3)),
        np.array([[[np.nan, 0.0], [0.0, 1.0]]]),
        np.array([[[1.0, 0.5], [0.0, 1.0]]]),
        np.array([[[1.0, 0.0], [0.0, -0.5]]]),
    ],
)
def test_malformed_history_is_rejected(bad: np.ndarray) -> None:
    with pytest.raises(InvalidModelInputError):
        RandomWalkRealizedCovariance().fit(bad)
    with pytest.raises(InvalidModelInputError):
        EWMARealizedCovariance(decay=0.5).fit(bad)


def test_forecast_before_fit_raises() -> None:
    with pytest.raises(InvalidModelInputError):
        RandomWalkRealizedCovariance().forecast()
    with pytest.raises(InvalidModelInputError):
        EWMARealizedCovariance(decay=0.5).forecast()


def test_rw_singular_psd_is_not_repaired() -> None:
    forecast = (
        RandomWalkRealizedCovariance().fit(_history(SPD_A, SINGULAR_PSD)).forecast()
    )
    np.testing.assert_allclose(forecast.matrix, SINGULAR_PSD)
    assert forecast.diagnostics.positive_semidefinite is True
    assert forecast.diagnostics.positive_definite is False
    assert forecast.diagnostics.min_eigenvalue == pytest.approx(0.0, abs=1e-12)
    assert forecast.diagnostics.condition_number is None


def test_rw_no_lookahead_future_matrix() -> None:
    window = _history(SPD_A, SPD_B)
    future = SPD_C.copy()
    first = RandomWalkRealizedCovariance().fit(window).forecast().matrix
    future[0, 0] = 99.0
    future[1, 1] = 99.0
    future[0, 1] = 0.1
    future[1, 0] = 0.1
    second = RandomWalkRealizedCovariance().fit(window).forecast().matrix
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, SPD_B)
    leaked = RandomWalkRealizedCovariance().fit(_history(SPD_A, SPD_B, future)).forecast()
    assert not np.allclose(leaked.matrix, first)


def test_ewma_hand_calculated_2x2() -> None:
    decay = 0.5
    history = _history(SPD_A, SPD_B)
    expected = decay * SPD_A + (1.0 - decay) * SPD_B
    forecast = EWMARealizedCovariance(decay=decay).fit(history).forecast()
    np.testing.assert_allclose(forecast.matrix, expected)


def test_ewma_closed_form_short_sequence() -> None:
    decay = 0.5
    history = _history(SPD_A, SPD_B, SPD_C)
    t_last = history.shape[0] - 1
    expected = (decay**t_last) * history[0]
    for time_index in range(1, history.shape[0]):
        expected = expected + (1.0 - decay) * (decay ** (t_last - time_index)) * history[
            time_index
        ]
    forecast = EWMARealizedCovariance(decay=decay).fit(history).forecast()
    np.testing.assert_allclose(forecast.matrix, expected)


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
def test_ewma_rejects_decay_outside_open_unit_interval(decay: float) -> None:
    with pytest.raises(InvalidModelConfigurationError):
        EWMARealizedCovariance(decay=decay)


def test_ewma_one_matrix_returns_that_matrix() -> None:
    forecast = EWMARealizedCovariance(decay=0.94).fit(_history(SPD_A)).forecast()
    np.testing.assert_allclose(forecast.matrix, SPD_A)


def test_ewma_spd_sequence_is_spd() -> None:
    history = _history(SPD_A, SPD_B, SPD_C)
    forecast = EWMARealizedCovariance(decay=0.94).fit(history).forecast()
    assert forecast.diagnostics.positive_definite is True
    assert forecast.diagnostics.positive_semidefinite is True
    np.linalg.cholesky(forecast.matrix)


def test_ewma_shared_null_space_stays_singular() -> None:
    forecast = EWMARealizedCovariance(decay=0.5).fit(NULL_SHARED).forecast()
    np.testing.assert_allclose(forecast.matrix[:, 1], 0.0)
    np.testing.assert_allclose(forecast.matrix[1, :], 0.0)
    assert forecast.diagnostics.positive_semidefinite is True
    assert forecast.diagnostics.positive_definite is False
    assert forecast.matrix[0, 0] == pytest.approx(0.5 * (0.5 * 1.0 + 0.5 * 4.0) + 0.5 * 9.0)


def test_ewma_no_lookahead_future_matrix() -> None:
    window = _history(SPD_A, SPD_B)
    future = SPD_C.copy()
    first = EWMARealizedCovariance(decay=0.5).fit(window).forecast().matrix
    future += np.eye(2)
    second = EWMARealizedCovariance(decay=0.5).fit(window).forecast().matrix
    np.testing.assert_array_equal(first, second)
    leaked = EWMARealizedCovariance(decay=0.5).fit(_history(SPD_A, SPD_B, future)).forecast()
    assert not np.allclose(leaked.matrix, first)


def test_ewma_different_decay_values_differ() -> None:
    history = _history(SPD_A, SPD_B, SPD_C)
    slow = EWMARealizedCovariance(decay=0.94).fit(history).forecast().matrix
    fast = EWMARealizedCovariance(decay=0.5).fit(history).forecast().matrix
    assert not np.allclose(slow, fast)
    np.testing.assert_allclose(fast, 0.5 * (0.5 * SPD_A + 0.5 * SPD_B) + 0.5 * SPD_C)


def test_synthetic_protocol_window_no_lookahead() -> None:
    """RW and EWMA consume only the protocol estimation window through origin t."""
    calendar = pd.bdate_range("1990-01-01", periods=1500)
    protocol = build_temporal_protocol(calendar)
    target = protocol.validation_targets()[0]
    step = protocol.forecast_step(protocol.origin_for_target(target))
    assert step.target == target
    assert step.target_index == step.origin_index + 1
    n_dates = len(calendar)
    cube = np.stack(
        [(1.0 + 0.01 * float(i)) * np.eye(2) for i in range(n_dates)],
        axis=0,
    )
    window = cube[step.window_start_index : step.window_end_index]
    assert window.shape[0] == protocol.m
    rw = RandomWalkRealizedCovariance().fit(window).forecast()
    ewma = EWMARealizedCovariance(decay=0.94).fit(window).forecast()
    np.testing.assert_allclose(rw.matrix, cube[step.origin_index])
    assert not np.allclose(rw.matrix, cube[step.target_index])
    assert not np.allclose(ewma.matrix, cube[step.target_index])
    with pytest.raises(ConfirmLockedError):
        protocol.confirm_targets()
    with pytest.raises(ConfirmLockedError):
        protocol.forecast_schedule(BlockName.CONFIRM)
    assert protocol.confirm_locked is True
