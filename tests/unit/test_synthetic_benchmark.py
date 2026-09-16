"""Synthetic DGP and end-to-end rolling evaluation tests. No market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covharness.evaluation import (
    LOSS_REDUCED_QLIKE,
    LOSS_SQUARED_FROBENIUS,
    EvaluationAlignmentError,
    build_loss_panel,
    score_forecast_records,
    summarize_loss_panel,
    target_covariance_panel,
)
from covharness.models import (
    EWMARealizedCovariance,
    HARDRDRealizedCovariance,
    LSTMBEKKCovariance,
    RandomWalkRealizedCovariance,
    RidgeDRDRealizedCovariance,
)
from covharness.protocol import (
    DEFAULT_REFIT_CADENCE,
    BlockName,
    ConfirmLockedError,
    build_schedule,
    build_temporal_protocol,
    run_block_forecasts,
    run_rolling_forecasts,
)
from covharness.protocol.runner import RollingAction, RollingForecastRecord
from covharness.simulation import synthetic_benchmark_panel

SEED = 20260916
M = 23
N_TARGETS = 4
N_ASSETS = 2
N_TIMES = M + N_TARGETS


def _cheap_models() -> list[object]:
    return [
        RandomWalkRealizedCovariance(),
        EWMARealizedCovariance(decay=0.94),
        HARDRDRealizedCovariance(),
        RidgeDRDRealizedCovariance(lambda_=0.5),
    ]


def test_same_seed_repeats_and_other_seed_can_differ() -> None:
    first = synthetic_benchmark_panel(n_times=12, n_assets=3, seed=SEED)
    second = synthetic_benchmark_panel(n_times=12, n_assets=3, seed=SEED)
    np.testing.assert_array_equal(first.daily_returns, second.daily_returns)
    np.testing.assert_array_equal(first.realized_covariances, second.realized_covariances)
    np.testing.assert_array_equal(first.realized_quarticity, second.realized_quarticity)
    other = synthetic_benchmark_panel(n_times=12, n_assets=3, seed=SEED + 1)
    assert not np.allclose(first.daily_returns, other.daily_returns)


def test_synthetic_rcov_is_strictly_pd_and_panels_align() -> None:
    panel = synthetic_benchmark_panel(n_times=15, n_assets=3, seed=SEED)
    assert len(panel.calendar) == 15
    assert panel.daily_returns.shape == (15, 3)
    assert panel.realized_covariances.shape == (15, 3, 3)
    assert panel.realized_quarticity.shape == (15, 3)
    assert np.isfinite(panel.realized_quarticity).all()
    assert np.all(panel.realized_quarticity >= 0.0)
    for time in range(15):
        np.linalg.cholesky(panel.realized_covariances[time])
    first = panel.realized_covariances[0]
    later = panel.realized_covariances[7]
    assert not np.allclose(first, later)


def test_end_to_end_qlike_and_frobenius_share_targets() -> None:
    panel = synthetic_benchmark_panel(n_times=N_TIMES, n_assets=N_ASSETS, seed=SEED)
    schedule = build_schedule(
        panel.calendar,
        panel.calendar[M:],
        m=M,
        refit_cadence=DEFAULT_REFIT_CADENCE,
    )
    assert len(schedule) == N_TARGETS
    targets = target_covariance_panel(panel.calendar, panel.realized_covariances)
    scored_all = []
    model_names = []
    for model in _cheap_models():
        records = run_rolling_forecasts(
            model=model,
            schedule=schedule,
            calendar=panel.calendar,
            daily_returns=panel.daily_returns,
            realized_covariances=panel.realized_covariances,
            realized_quarticity=panel.realized_quarticity,
        )
        model_names.append(records[0].model_name)
        scored_all.extend(score_forecast_records(records, targets))
        for record, step in zip(records, schedule, strict=True):
            assert record.target == step.target
            assert record.origin == step.origin
            origin_s = panel.realized_covariances[step.origin_index]
            target_s = panel.realized_covariances[step.target_index]
            # Evaluation uses S_{t+1}. The origin matrix is a different date.
            assert step.target_index == step.origin_index + 1
            if not np.allclose(origin_s, target_s):
                assert not np.array_equal(origin_s, target_s)
    qlike = build_loss_panel(
        scored_all,
        loss_name=LOSS_REDUCED_QLIKE,
        model_order=tuple(model_names),
    )
    frobenius = build_loss_panel(
        scored_all,
        loss_name=LOSS_SQUARED_FROBENIUS,
        model_order=tuple(model_names),
    )
    assert qlike.values.shape == (N_TARGETS, len(model_names))
    assert frobenius.values.shape == qlike.values.shape
    np.testing.assert_array_equal(qlike.targets, frobenius.targets)
    assert list(qlike.model_names) == model_names
    assert np.isfinite(qlike.values).all()
    assert np.isfinite(frobenius.values).all()
    summaries = summarize_loss_panel(qlike)
    assert [row.model_name for row in summaries] == model_names
    assert all(np.isfinite(row.mean_loss) for row in summaries)


def test_end_to_end_has_no_target_leakage_into_runner() -> None:
    panel = synthetic_benchmark_panel(n_times=N_TIMES, n_assets=N_ASSETS, seed=SEED + 3)
    schedule = build_schedule(
        panel.calendar,
        panel.calendar[M:],
        m=M,
        refit_cadence=DEFAULT_REFIT_CADENCE,
    )
    captured: list[np.ndarray] = []
    model = RandomWalkRealizedCovariance()
    original_fit = model.fit

    def counted(history):
        captured.append(np.array(history, dtype=float, copy=True))
        return original_fit(history)

    model.fit = counted  # type: ignore[method-assign]
    run_rolling_forecasts(
        model=model,
        schedule=schedule,
        calendar=panel.calendar,
        daily_returns=panel.daily_returns,
        realized_covariances=panel.realized_covariances,
    )
    first = schedule[0]
    assert captured[0].shape[0] == M
    np.testing.assert_array_equal(
        captured[0][-1], panel.realized_covariances[first.origin_index]
    )
    assert not np.array_equal(
        captured[0][-1], panel.realized_covariances[first.target_index]
    )


def test_confirm_remains_locked() -> None:
    calendar = pd.bdate_range("1990-01-01", periods=1500)
    protocol = build_temporal_protocol(calendar)
    panel = synthetic_benchmark_panel(n_times=40, n_assets=2, seed=SEED)
    with pytest.raises(ConfirmLockedError):
        run_block_forecasts(
            model=RandomWalkRealizedCovariance(),
            protocol=protocol,
            block=BlockName.CONFIRM,
            realized_covariances=np.repeat(
                panel.realized_covariances[:1], len(calendar), axis=0
            ),
        )


def test_lstm_records_score_through_the_same_evaluator() -> None:
    panel = synthetic_benchmark_panel(n_times=8, n_assets=2, seed=SEED)
    origin = panel.calendar[3]
    target = panel.calendar[4]
    forecast = np.array(panel.realized_covariances[3], dtype=float, copy=True)
    forecast = forecast + np.eye(2) * 0.05
    record = RollingForecastRecord(
        model_name=LSTMBEKKCovariance(
            seed=0,
            num_layers=3,
            dropout=0.1,
            learning_rate=0.05,
            gradient_clip_norm=1.0,
            max_epochs=1,
        ).identity.name,
        origin=origin,
        target=target,
        refit=True,
        action=RollingAction.FIT,
        covariance=forecast,
    )
    assert record.model_name == "lstm_bekk"
    scored = score_forecast_records(
        [record],
        target_covariance_panel(panel.calendar, panel.realized_covariances),
    )
    assert {row.loss_name for row in scored} == {
        LOSS_REDUCED_QLIKE,
        LOSS_SQUARED_FROBENIUS,
    }
    assert all(np.isfinite(row.loss_value) for row in scored)


def test_mismatched_support_still_fails_on_synthetic_records() -> None:
    panel = synthetic_benchmark_panel(n_times=N_TIMES, n_assets=N_ASSETS, seed=SEED)
    schedule = build_schedule(
        panel.calendar,
        panel.calendar[M:],
        m=M,
        refit_cadence=DEFAULT_REFIT_CADENCE,
    )
    targets = target_covariance_panel(panel.calendar, panel.realized_covariances)
    rw = run_rolling_forecasts(
        model=RandomWalkRealizedCovariance(),
        schedule=schedule,
        calendar=panel.calendar,
        realized_covariances=panel.realized_covariances,
    )
    har = run_rolling_forecasts(
        model=HARDRDRealizedCovariance(),
        schedule=schedule[:-1],
        calendar=panel.calendar,
        realized_covariances=panel.realized_covariances,
    )
    scored = list(score_forecast_records(rw, targets)) + list(
        score_forecast_records(har, targets)
    )
    with pytest.raises(EvaluationAlignmentError, match="identical target dates"):
        build_loss_panel(scored, loss_name=LOSS_REDUCED_QLIKE)
