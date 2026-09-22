"""Focused tests for the Binance VALIDATION orchestrator. No market-data ranking."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from covharness.models.random_walk import RandomWalkRealizedCovariance
from covharness.protocol.binance import BinanceBlock, build_binance_segmented_protocol
from covharness.protocol.binance_validation import (
    FROZEN_ASSETS,
    FROZEN_CONFIG_SHA256,
    FROZEN_PANEL_SHA256,
    MAX_MODEL_OBSERVATION_DATE,
    MAX_SCORING_TARGET_DATE,
    REQUIRED_VALIDATION_TARGETS,
    CandidateRunResult,
    FrozenHashMismatchError,
    ValidationLeakError,
    ValidationView,
    assert_no_future_fit_dates,
    ensemble_covariances,
    evaluate_records,
    load_complete_checkpoint,
    load_validation_view,
    lstm_ensemble_result,
    request_validation_schedule,
    require_frozen_hashes,
    run_one_configuration,
    select_family_winner,
    sha256_file,
    write_checkpoint,
)
from covharness.protocol.exceptions import ConfirmLockedError
from covharness.protocol.runner import RollingAction, RollingForecastRecord


def _spd(n: int = 2, scale: float = 0.2) -> np.ndarray:
    return np.eye(n) * scale + 0.05


def _synthetic_view() -> tuple[object, ValidationView]:
    protocol = build_binance_segmented_protocol()
    schedule = request_validation_schedule(protocol)
    calendar = protocol.calendar_for_block(BinanceBlock.VALIDATION)
    dates = tuple(stamp.date() for stamp in calendar)
    n_times = len(dates)
    n_assets = 2
    rcov = np.stack([_spd(n_assets) for _ in range(n_times)], axis=0)
    daily = np.full((n_times, n_assets), 0.001)
    rq = np.full((n_times, n_assets), 1e-6)
    view = ValidationView(
        dates=dates,
        calendar=calendar,
        assets=("A", "B"),
        daily_returns=daily,
        rcov=rcov,
        rq=rq,
        panel_sha256="panel-test",
        config_sha256="config-test",
        max_model_observation_date=MAX_MODEL_OBSERVATION_DATE,
        max_scoring_target_date=MAX_SCORING_TARGET_DATE,
    )
    return schedule, view


def test_only_validation_schedule_is_requested() -> None:
    protocol = build_binance_segmented_protocol()
    schedule = request_validation_schedule(protocol)
    assert schedule.block == "VALIDATION"
    assert len(schedule.steps) == REQUIRED_VALIDATION_TARGETS
    assert schedule.steps[0].origin.date() == dt.date(2022, 7, 16)
    assert schedule.steps[-1].target.date() == dt.date(2023, 3, 23)


def test_default_confirm_remains_locked() -> None:
    protocol = build_binance_segmented_protocol()
    with pytest.raises(ConfirmLockedError):
        protocol.confirm_targets()
    with pytest.raises(ConfirmLockedError):
        protocol.forecast_schedule(BinanceBlock.CONFIRM)
    request_validation_schedule(protocol)


def test_post_validation_rows_cannot_enter_model_fitting() -> None:
    origin = dt.date(2023, 3, 22)
    with pytest.raises(ValidationLeakError, match="after origin"):
        assert_no_future_fit_dates([origin, dt.date(2023, 3, 23)], origin)
    with pytest.raises(ValidationLeakError, match="after origin"):
        assert_no_future_fit_dates([dt.date(2023, 3, 21), dt.date(2023, 3, 23)], origin)
    with pytest.raises(ValidationLeakError, match="exceeds 2023-03-22"):
        assert_no_future_fit_dates([dt.date(2023, 3, 25)], dt.date(2023, 3, 25))
    assert_no_future_fit_dates([dt.date(2023, 3, 21), origin], origin)


def test_250_targets_are_required() -> None:
    schedule = request_validation_schedule()
    assert len(schedule.steps) == 250
    halt = dt.date(2023, 3, 24)
    assert halt not in [step.origin.date() for step in schedule.steps]
    assert halt not in [step.target.date() for step in schedule.steps]


def test_incomplete_candidate_support_is_invalid() -> None:
    records = [
        RollingForecastRecord(
            "rw",
            pd.Timestamp("2022-07-16"),
            pd.Timestamp("2022-07-17"),
            True,
            RollingAction.FIT,
            _spd(),
        )
    ]
    result = evaluate_records(
        "RW",
        "RW01",
        None,
        records,
        np.stack([_spd()]),
        np.array([False]),
        0,
        0.1,
    )
    assert result.complete_support is False
    assert result.mean_qlike is None


def test_qlike_not_frobenius_selects() -> None:
    low_qlike = CandidateRunResult(
        "EWMA", "EWMA02", None, True, 250, 1.0, 9.0, 0, 0, 0, 0, 0, None, 1.0,
        (), (), None, None, None, None, None,
    )
    high_qlike = CandidateRunResult(
        "EWMA", "EWMA01", None, True, 250, 2.0, 0.1, 0, 0, 0, 0, 0, None, 1.0,
        (), (), None, None, None, None, None,
    )
    winner = select_family_winner("EWMA", [high_qlike, low_qlike])
    assert winner["selected_id"] == "EWMA02"


def test_exact_tie_picks_lexicographically_smaller_id() -> None:
    first = CandidateRunResult(
        "Ridge-DRD", "RIDGE10", None, True, 250, 1.5, 3.0, 0, 0, 0, 0, 0, None, 1.0,
        (), (), None, None, None, None, None,
    )
    second = CandidateRunResult(
        "Ridge-DRD", "RIDGE02", None, True, 250, 1.5, 0.0, 0, 0, 0, 0, 0, None, 1.0,
        (), (), None, None, None, None, None,
    )
    winner = select_family_winner("Ridge-DRD", [first, second])
    assert winner["selected_id"] == "RIDGE02"


def test_lstm_ensemble_averages_covariances_before_loss() -> None:
    cubes = [np.stack([_spd(2, 0.2 + 0.01 * seed) for _ in range(250)]) for seed in range(5)]
    ensemble = ensemble_covariances(cubes)
    expected = np.mean(np.stack(cubes, axis=0), axis=0)
    np.testing.assert_allclose(ensemble, expected)
    proxy = np.stack([_spd(2, 0.25) for _ in range(250)])
    seed_rows = []
    for seed, cube in enumerate(cubes):
        seed_rows.append(
            CandidateRunResult(
                "LSTM-BEKK",
                "LSTM01",
                seed,
                True,
                250,
                1.0 + seed,
                1.0,
                0,
                0,
                0,
                0,
                0,
                None,
                1.0,
                tuple(dt.date(2022, 7, 17) for _ in range(250)),
                tuple(dt.date(2022, 7, 17) for _ in range(250)),
                cube,
                np.ones(250),
                np.ones(250),
                np.zeros(250, dtype=bool),
                np.zeros(250, dtype=bool),
            )
        )
    result = lstm_ensemble_result(seed_rows, proxy)
    assert result.complete_support is True
    assert result.seed is None
    assert result.mean_qlike is not None
    seed_mean_qlike = float(np.mean([row.mean_qlike for row in seed_rows]))
    assert abs(float(result.mean_qlike) - seed_mean_qlike) > 1e-6
    np.testing.assert_allclose(result.forecasts, expected)


def test_best_seed_is_never_selected() -> None:
    rows = []
    for seed in range(5):
        rows.append(
            CandidateRunResult(
                "LSTM-BEKK", "LSTM03", seed, True, 250, 0.1 * seed, 1.0, 0, 0, 0, 0, 0,
                None, 1.0, (), (), None, None, None, None, None,
            )
        )
    winner = select_family_winner(
        "LSTM-BEKK",
        [
            CandidateRunResult(
                "LSTM-BEKK", "LSTM03", None, True, 250, 1.2, 1.0, 0, 0, 0, 0, 0, None, 1.0,
                (), (), None, None, None, None, None,
            ),
            CandidateRunResult(
                "LSTM-BEKK", "LSTM01", None, True, 250, 1.1, 9.0, 0, 0, 0, 0, 0, None, 1.0,
                (), (), None, None, None, None, None,
            ),
        ],
    )
    assert winner["selected_id"] == "LSTM01"
    assert min(row.mean_qlike for row in rows) == 0.0


def test_failed_lstm_seed_invalidates_candidate() -> None:
    cubes = [np.stack([_spd() for _ in range(250)]) for _ in range(4)]
    seeds = []
    for seed, cube in enumerate(cubes):
        seeds.append(
            CandidateRunResult(
                "LSTM-BEKK", "LSTM02", seed, True, 250, 1.0, 1.0, 0, 0, 0, 0, 0, None, 1.0,
                (), (), cube, np.ones(250), np.ones(250), np.zeros(250, dtype=bool),
                np.zeros(250, dtype=bool),
            )
        )
    seeds.append(
        CandidateRunResult(
            "LSTM-BEKK", "LSTM02", 4, False, 0, None, None, 0, 250, 250, 0, 1, "fit failed",
            0.1, (), (), None, None, None, None, None,
        )
    )
    result = lstm_ensemble_result(seeds, np.stack([_spd() for _ in range(250)]))
    assert result.complete_support is False
    assert result.mean_qlike is None
    assert "failed seed" in str(result.failure_reason)


def test_frozen_hashes_are_checked(tmp_path: Path) -> None:
    config = Path("configs/binance_open_data_core.yaml")
    panel = Path("data/processed/binance_five_asset_panel.npz")
    assert sha256_file(config) == FROZEN_CONFIG_SHA256
    if not panel.is_file():
        pytest.skip("local production panel NPZ is not a Git artifact")
    assert sha256_file(panel) == FROZEN_PANEL_SHA256
    require_frozen_hashes(panel, config)
    bogus = tmp_path / "bogus.yaml"
    bogus.write_text("not-the-frozen-config\n", encoding="utf-8")
    with pytest.raises(FrozenHashMismatchError):
        require_frozen_hashes(panel, bogus)


def test_resume_cannot_substitute_a_partial_candidate(tmp_path: Path) -> None:
    required = [dt.date(2022, 7, 17) + dt.timedelta(days=i) for i in range(250)]
    partial = CandidateRunResult(
        "EWMA",
        "EWMA01",
        None,
        False,
        10,
        None,
        None,
        0,
        240,
        0,
        0,
        0,
        "interrupted",
        0.2,
        (),
        (),
        None,
        None,
        None,
        None,
        None,
    )
    write_checkpoint(tmp_path, partial, panel_sha256="p", config_sha256="c")
    loaded = load_complete_checkpoint(
        tmp_path,
        "EWMA",
        "EWMA01",
        None,
        panel_sha256="p",
        config_sha256="c",
        required_targets=required,
    )
    assert loaded is None


def test_run_one_configuration_uses_validation_only(tmp_path: Path) -> None:
    schedule, view = _synthetic_view()
    result = run_one_configuration(
        family="RW",
        row={"id": "RW01"},
        seed=None,
        view=view,
        schedule=schedule,
        checkpoint_dir=tmp_path,
        model_factory=lambda family, row, seed: RandomWalkRealizedCovariance(),
    )
    assert result.n_targets == 250
    assert result.targets[-1] == MAX_SCORING_TARGET_DATE
    assert result.origins[-1] == MAX_MODEL_OBSERVATION_DATE
    assert dt.date(2023, 3, 24) not in result.targets
    reused = run_one_configuration(
        family="RW",
        row={"id": "RW01"},
        seed=None,
        view=view,
        schedule=schedule,
        checkpoint_dir=tmp_path,
        model_factory=lambda family, row, seed: (_ for _ in ()).throw(
            RuntimeError("should not refit")
        ),
    )
    assert reused.n_targets == 250
    assert FROZEN_ASSETS[0] == "BTCUSDT"


def test_validation_view_omits_later_npz_rows() -> None:
    panel = Path("data/processed/binance_five_asset_panel.npz")
    if not panel.is_file():
        pytest.skip("local production panel NPZ is not a Git artifact")
    view = load_validation_view(
        panel,
        Path("configs/binance_open_data_core.yaml"),
    )
    assert view.assets == FROZEN_ASSETS
    assert len(view.dates) == 500
    assert view.dates[0] == dt.date(2021, 11, 9)
    assert view.max_model_observation_date == MAX_MODEL_OBSERVATION_DATE
    assert view.max_scoring_target_date == MAX_SCORING_TARGET_DATE
    assert dt.date(2023, 3, 24) not in view.dates
    assert all(day <= MAX_SCORING_TARGET_DATE for day in view.dates)
    assert dt.date(2023, 3, 25) not in view.dates
    assert dt.date(2023, 11, 30) not in view.dates
    assert dt.date(2025, 4, 13) not in view.dates
