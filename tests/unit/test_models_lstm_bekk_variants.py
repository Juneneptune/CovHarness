"""Information-matched LSTM-BEKK checks. The faithful class stays bitwise."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from covharness.losses.qlike import reduced_qlike_loss
from covharness.models.exceptions import InvalidModelInputError
from covharness.models.lstm_bekk import LSTMBEKKCovariance
from covharness.models.lstm_bekk_variants import (
    apply_feature_scaler,
    fit_feature_scaler,
    realized_feature_matrix,
    reduced_qlike_torch,
)
from covharness.protocol.rolling import ForecastStep
from covharness.protocol.runner import build_origin_payload


def test_faithful_forecast_is_bitwise_unchanged() -> None:
    returns = np.random.default_rng(0).normal(0.001, 0.01, size=(12, 3))
    model = LSTMBEKKCovariance(
        seed=0,
        num_layers=3,
        dropout=0.0,
        learning_rate=0.05,
        gradient_clip_norm=1.0,
        max_epochs=2,
    )
    model.fit(returns)
    expected = np.array(
        [
            [8.3287577851434865e-05, -7.0026554612614362e-05, -2.6788602359810297e-05],
            [-7.0026554612614362e-05, 3.5384739946465159e-04, -2.5249924699728579e-04],
            [-2.6788602359810297e-05, -2.5249924699728579e-04, 6.9252096060572503e-04],
        ]
    )
    assert np.array_equal(model.forecast().matrix, expected)


def test_reduced_qlike_torch_matches_the_package_loss() -> None:
    proxy = np.array([[0.2, 0.05], [0.05, 0.3]])
    forecast = np.array([[0.4, 0.02], [0.02, 0.5]])
    tensor = reduced_qlike_torch(
        torch.tensor(forecast, dtype=torch.float64),
        torch.tensor(proxy, dtype=torch.float64),
    )
    assert float(tensor.detach()) == pytest.approx(reduced_qlike_loss(proxy, forecast), abs=0.0, rel=0.0)


def test_scaler_uses_only_the_supplied_window() -> None:
    window = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    outside = np.array([[100.0, -100.0]])
    mean, scale = fit_feature_scaler(window)
    full_mean, _full_scale = fit_feature_scaler(np.vstack([window, outside]))
    assert np.allclose(mean, np.array([3.0, 4.0]))
    assert not np.allclose(mean, full_mean)
    scaled = apply_feature_scaler(window, mean, scale)
    assert np.allclose(scaled.mean(axis=0), 0.0)


def test_feature_row_uses_that_day_only() -> None:
    rcov = np.array(
        [
            [[0.04, 0.01], [0.01, 0.09]],
            [[0.05, 0.00], [0.00, 0.08]],
        ]
    )
    positive = np.array([[0.02, 0.04], [0.03, 0.05]])
    negative = np.array([[0.02, 0.05], [0.02, 0.03]])
    quarticity = np.array([[0.001, 0.002], [0.004, 0.003]])
    original = realized_feature_matrix(positive, negative, rcov, quarticity)
    shifted_positive = positive.copy()
    shifted_positive[1] = positive[1] * 4.0
    shifted = realized_feature_matrix(shifted_positive, negative, rcov, quarticity)
    assert np.array_equal(original[0], shifted[0])
    assert not np.array_equal(original[1], shifted[1])


def test_origin_feature_window_excludes_the_target() -> None:
    calendar = pd.date_range("2024-01-01", periods=6, freq="D")
    step = ForecastStep(
        origin=calendar[3],
        target=calendar[4],
        window_start=calendar[1],
        origin_index=3,
        target_index=4,
        window_start_index=1,
        window_end_index=4,
        refit=True,
        m=3,
    )
    features = np.arange(6, dtype=float).reshape(6, 1)
    payload = build_origin_payload(step=step, calendar=calendar, feature_panel=features)
    assert payload.feature_window is not None
    assert np.array_equal(payload.feature_window[:, 0], np.array([1.0, 2.0, 3.0]))
    assert payload.current_feature == pytest.approx(3.0)
    assert payload.feature_window[-1, 0] != 4.0


def test_zero_semivariance_is_rejected() -> None:
    rcov = np.array([[[0.04, 0.0], [0.0, 0.09]]])
    positive = np.array([[0.0, 0.04]])
    negative = np.array([[0.04, 0.05]])
    quarticity = np.array([[0.001, 0.002]])
    with pytest.raises(InvalidModelInputError):
        realized_feature_matrix(positive, negative, rcov, quarticity)
