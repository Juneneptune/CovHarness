"""Known-scale, known-shrinkage, and no-lookahead checks for recalibration."""

from __future__ import annotations

import numpy as np
import pytest

from covharness.evaluation.recalibration import (
    LAMBDA_GRID,
    apply_recalibrated,
    fitting_indices,
    recalibrate_forecast_sequence,
)


def _schedule(n_rows: int, *, segment: str = "SCREEN"):
    origins = [f"2024-01-{day:02d}" for day in range(1, n_rows + 1)]
    targets = [f"2024-01-{day:02d}" for day in range(2, n_rows + 2)]
    segments = [segment] * n_rows
    refit = np.zeros(n_rows, dtype=bool)
    refit[::21] = True
    return origins, targets, segments, refit


def _spd(values: list[float]) -> np.ndarray:
    base = np.diag(np.asarray(values, dtype=float))
    base[0, 1] = base[1, 0] = 0.002
    return base


def test_r1_recovers_a_planted_scale():
    origins, targets, segments, refit = _schedule(30)
    proxies = np.stack([_spd([0.02, 0.03]) for _ in range(30)])
    planted = 2.5
    forecasts = proxies / planted
    result = recalibrate_forecast_sequence(
        forecasts,
        proxies,
        origin_dates=origins,
        target_dates=targets,
        segments=segments,
        refit=refit,
    )
    assert result.scale_r1[21] == pytest.approx(planted)
    assert np.allclose(result.r1[21], proxies[21])


def test_r2_recovers_a_planted_lambda_when_the_forecast_overreacts():
    origins, targets, segments, refit = _schedule(30)
    proxies = np.stack(
        [_spd([0.01 + 0.001 * index, 0.02 + 0.002 * index]) for index in range(30)]
    )
    window = proxies[:21]
    hbar = 0.5 * (window.mean(axis=0) + window.mean(axis=0).T)
    forecasts = hbar + (proxies - hbar) / 0.7
    result = recalibrate_forecast_sequence(
        forecasts,
        proxies,
        origin_dates=origins,
        target_dates=targets,
        segments=segments,
        refit=refit,
    )
    assert result.lambda_r2[21] == pytest.approx(0.7)
    assert result.scale_r2[21] == pytest.approx(1.0)
    assert np.allclose(result.r2[21], proxies[21])
    assert 0.7 in LAMBDA_GRID or any(np.isclose(LAMBDA_GRID, 0.7))


def test_fitting_indices_reject_a_one_day_shift():
    origins, targets, segments, _ = _schedule(30)
    correct = fitting_indices(origins, targets, segments, 21)
    assert correct[-1] == 20
    assert 21 not in correct
    shifted_targets = [f"2024-01-{day:02d}" for day in range(1, 31)]
    shifted = fitting_indices(origins, shifted_targets, segments, 21)
    assert 21 in shifted
    assert not np.array_equal(correct, shifted)


def test_a_validation_pair_does_not_enter_a_screen_fit():
    origins, targets, segments, _ = _schedule(30)
    segments = ["VALIDATION"] * 10 + ["SCREEN"] * 20
    indices = fitting_indices(origins, targets, segments, 21)
    assert indices.min() >= 10
    assert set(np.asarray(segments)[indices]) == {"SCREEN"}


def test_identity_before_21_pairs():
    origins, targets, segments, refit = _schedule(30)
    proxies = np.stack([_spd([0.02, 0.04]) for _ in range(30)])
    forecasts = proxies / 3.0
    result = recalibrate_forecast_sequence(
        forecasts,
        proxies,
        origin_dates=origins,
        target_dates=targets,
        segments=segments,
        refit=refit,
    )
    assert np.allclose(result.r1[:21], forecasts[:21])
    assert np.allclose(result.r2[:21], forecasts[:21])
    assert result.identity_r1[:21].all()
    assert result.identity_r2[:21].all()


def test_r2_with_lambda_one_equals_r1():
    forecast = _spd([0.02, 0.03])
    hbar = _spd([0.05, 0.07])
    scaled = apply_recalibrated(forecast, 1.4, 1.0, hbar)
    assert np.array_equal(scaled, 1.4 * forecast)
    origins, targets, segments, refit = _schedule(30)
    proxies = np.stack(
        [_spd([0.01 + 0.001 * index, 0.03]) for index in range(30)]
    )
    forecasts = proxies / 1.8
    forced = recalibrate_forecast_sequence(
        forecasts,
        proxies,
        origin_dates=origins,
        target_dates=targets,
        segments=segments,
        refit=refit,
        lambda_grid=(1.0,),
    )
    free = recalibrate_forecast_sequence(
        forecasts,
        proxies,
        origin_dates=origins,
        target_dates=targets,
        segments=segments,
        refit=refit,
        lambda_grid=(1.0,),
    )
    assert np.allclose(forced.r2, free.r1)
    assert np.allclose(forced.r2[21], 1.8 * forecasts[21])
