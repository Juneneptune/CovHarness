"""Focused tests for the Binance segmented schedule and core configuration."""

from __future__ import annotations

import datetime as dt
import inspect

import pandas as pd
import pytest

from covharness.data.binance_calendar import HALT_DATE, production_dates
from covharness.protocol.binance import (
    CORE_ROSTER,
    BinanceBlock,
    BinanceSegmentedProtocol,
    BinanceSelectionError,
    CandidateValidationRecord,
    build_binance_segmented_protocol,
    core_config_sha256,
    frozen_core_config,
    load_core_config,
    select_validation_configuration,
    write_core_config,
)
from covharness.protocol.exceptions import ConfirmLockedError
from covharness.protocol.splits import TemporalProtocol


def _dates(index: pd.DatetimeIndex) -> list[dt.date]:
    return [stamp.date() for stamp in index]


def test_validation_has_exactly_250_targets() -> None:
    protocol = build_binance_segmented_protocol()
    schedule = protocol.forecast_schedule(BinanceBlock.VALIDATION)
    assert len(protocol.validation_targets()) == 250
    assert len(schedule.steps) == 250


def test_screen_has_exactly_500_targets() -> None:
    protocol = build_binance_segmented_protocol()
    schedule = protocol.forecast_schedule(BinanceBlock.SCREEN)
    assert len(protocol.screen_targets()) == 500
    assert len(schedule.steps) == 500


def test_confirm_has_exactly_500_targets_when_unlocked() -> None:
    protocol = build_binance_segmented_protocol()
    schedule = protocol.forecast_schedule(BinanceBlock.CONFIRM, unlock_confirm=True)
    assert len(protocol.confirm_targets(unlock_confirm=True)) == 500
    assert len(schedule.steps) == 500


def test_default_confirm_access_raises() -> None:
    protocol = build_binance_segmented_protocol()
    with pytest.raises(ConfirmLockedError):
        protocol.confirm_targets()
    with pytest.raises(ConfirmLockedError):
        protocol.forecast_schedule(BinanceBlock.CONFIRM)
    with pytest.raises(ConfirmLockedError):
        protocol.confirm_hidden_helper  # noqa: B018


def test_first_validation_origin_and_window() -> None:
    protocol = build_binance_segmented_protocol()
    schedule = protocol.forecast_schedule(BinanceBlock.VALIDATION)
    first = schedule.steps[0]
    assert first.origin.date() == dt.date(2022, 7, 16)
    assert first.target.date() == dt.date(2022, 7, 17)
    window = schedule.estimation_dates(first)
    assert _dates(window) == _dates(protocol.history_a_dates())
    assert window[0].date() == dt.date(2021, 11, 9)
    assert window[-1].date() == dt.date(2022, 7, 16)


def test_first_screen_origin_and_window() -> None:
    protocol = build_binance_segmented_protocol()
    schedule = protocol.forecast_schedule(BinanceBlock.SCREEN)
    first = schedule.steps[0]
    assert first.origin.date() == dt.date(2023, 11, 29)
    assert first.target.date() == dt.date(2023, 11, 30)
    window = schedule.estimation_dates(first)
    assert _dates(window) == _dates(protocol.history_b_dates())
    assert window[0].date() == dt.date(2023, 3, 25)
    assert window[-1].date() == dt.date(2023, 11, 29)


def test_first_confirm_origin_and_preceding_screen_window() -> None:
    protocol = build_binance_segmented_protocol()
    schedule = protocol.forecast_schedule(BinanceBlock.CONFIRM, unlock_confirm=True)
    first = schedule.steps[0]
    assert first.origin.date() == dt.date(2025, 4, 12)
    assert first.target.date() == dt.date(2025, 4, 13)
    window = schedule.estimation_dates(first)
    expected = protocol.screen_targets()[-250:]
    assert len(window) == 250
    assert _dates(window) == _dates(expected)
    assert window[-1].date() == dt.date(2025, 4, 12)


def test_halt_is_absent_from_windows_origins_and_targets() -> None:
    protocol = build_binance_segmented_protocol()
    halt = pd.Timestamp(HALT_DATE)
    for block, unlock in (
        (BinanceBlock.VALIDATION, False),
        (BinanceBlock.SCREEN, False),
        (BinanceBlock.CONFIRM, True),
    ):
        schedule = protocol.forecast_schedule(block, unlock_confirm=unlock)
        for step in schedule.steps:
            assert step.origin != halt
            assert step.target != halt
            window = schedule.estimation_dates(step)
            assert halt not in window
            assert HALT_DATE not in _dates(window)


def test_refit_flags_and_block_local_reset() -> None:
    protocol = build_binance_segmented_protocol()
    validation = protocol.forecast_schedule(BinanceBlock.VALIDATION)
    screen = protocol.forecast_schedule(BinanceBlock.SCREEN)
    confirm = protocol.forecast_schedule(BinanceBlock.CONFIRM, unlock_confirm=True)
    for schedule in (validation, screen, confirm):
        assert schedule.steps[0].refit is True
        assert schedule.steps[21].refit is True
        assert schedule.steps[42].refit is True
        assert schedule.steps[63].refit is True
        assert schedule.steps[1].refit is False
        assert schedule.steps[20].refit is False
    assert validation.steps[-1].refit is False
    assert screen.steps[0].refit is True
    assert confirm.steps[0].refit is True


def test_origin_precedes_target_by_one_utc_day() -> None:
    protocol = build_binance_segmented_protocol()
    for block, unlock in (
        (BinanceBlock.VALIDATION, False),
        (BinanceBlock.SCREEN, False),
        (BinanceBlock.CONFIRM, True),
    ):
        schedule = protocol.forecast_schedule(block, unlock_confirm=unlock)
        for step in schedule.steps:
            assert (step.target.date() - step.origin.date()).days == 1
            assert len(schedule.estimation_dates(step)) == 250
            assert step.m == 250


def test_no_packed_validation_to_history_b_step() -> None:
    protocol = build_binance_segmented_protocol()
    validation = protocol.forecast_schedule(BinanceBlock.VALIDATION)
    screen = protocol.forecast_schedule(BinanceBlock.SCREEN)
    assert validation.steps[-1].origin.date() == dt.date(2023, 3, 22)
    assert validation.steps[-1].target.date() == dt.date(2023, 3, 23)
    assert screen.steps[0].origin.date() == dt.date(2023, 11, 29)
    pairs = {(step.origin.date(), step.target.date()) for step in validation.steps}
    pairs.update((step.origin.date(), step.target.date()) for step in screen.steps)
    assert (dt.date(2023, 3, 23), dt.date(2023, 3, 25)) not in pairs


def test_schedule_does_not_read_market_measurements() -> None:
    import covharness.protocol.binance as module

    source = inspect.getsource(module)
    assert "binance_panel" not in source
    assert "daily_returns" not in source
    assert "rcov_5min" not in source
    assert "rq_per_asset" not in source
    assert "covharness.models" not in source
    build_binance_segmented_protocol().forecast_schedule(BinanceBlock.VALIDATION)
    assert HALT_DATE not in production_dates()


def test_generic_temporal_protocol_class_is_unchanged_by_import() -> None:
    assert TemporalProtocol.__name__ == "TemporalProtocol"
    protocol = build_binance_segmented_protocol()
    assert type(protocol) is BinanceSegmentedProtocol


def test_configuration_registry_counts() -> None:
    document = load_core_config()
    candidates = document["candidates"]
    assert list(document["roster"]) == list(CORE_ROSTER)
    assert len(candidates["EWMA"]) == 20
    assert len(candidates["Ridge-DRD"]) == 20
    assert len(candidates["XGBoost-DRD"]["grid"]) == 20
    assert len(candidates["LSTM-BEKK"]["grid"]) == 20
    for name in ("RW", "HAR-DRD", "HARQ-DRD", "LW-linear", "LW-NL", "DCC", "DCC-NL"):
        assert len(candidates[name]) == 1
    assert candidates["XGBoost-DRD"]["grid"][0]["id"] == "XGB01"
    assert candidates["XGBoost-DRD"]["grid"][-1]["id"] == "XGB20"
    assert candidates["XGBoost-DRD"]["grid"][0]["capacity"] == "C1"
    assert candidates["XGBoost-DRD"]["grid"][0]["regularization"] == "R1"
    assert candidates["XGBoost-DRD"]["grid"][-1]["capacity"] == "C4"
    assert candidates["XGBoost-DRD"]["grid"][-1]["regularization"] == "R5"
    assert candidates["LSTM-BEKK"]["grid"][0]["id"] == "LSTM01"
    assert candidates["LSTM-BEKK"]["grid"][-1]["id"] == "LSTM20"


def test_lstm_seeds_live_at_experiment_layer() -> None:
    document = load_core_config()
    assert document["seeds"] == [0, 1, 2, 3, 4]
    assert document["seed_aggregation"] == "mean_ensemble"
    assert document["allow_best_seed"] is False
    for row in document["candidates"]["LSTM-BEKK"]["grid"]:
        assert "seed" not in row
        assert "seeds" not in row


def test_selection_helper_minimum_qlike_and_lexicographic_tie() -> None:
    winner = select_validation_configuration(
        [
            CandidateValidationRecord("EWMA02", True, 1.5),
            CandidateValidationRecord("EWMA01", True, 1.2),
            CandidateValidationRecord("EWMA03", False, 0.1),
        ]
    )
    assert winner == "EWMA01"
    tied = select_validation_configuration(
        [
            {"config_id": "RIDGE10", "complete_support": True, "primary_score": 2.0},
            {"config_id": "RIDGE02", "complete_support": True, "primary_score": 2.0},
            {"config_id": "RIDGE01", "complete_support": False, "primary_score": 2.0},
        ]
    )
    assert tied == "RIDGE02"


def test_selection_helper_rejects_incomplete_and_nonfinite_scores() -> None:
    with pytest.raises(BinanceSelectionError, match="invalid"):
        select_validation_configuration(
            [
                CandidateValidationRecord("EWMA01", False, 0.1),
                CandidateValidationRecord("EWMA02", True, None),
                CandidateValidationRecord("EWMA03", True, float("nan")),
                CandidateValidationRecord("EWMA04", True, float("inf")),
            ]
        )


def test_selection_helper_does_not_run_models_or_read_panel() -> None:
    source = inspect.getsource(select_validation_configuration)
    assert "fit(" not in source
    assert "binance_panel" not in source
    assert "covharness.models" not in source
    select_validation_configuration(
        [CandidateValidationRecord("EWMA01", True, 1.0)]
    )


def test_written_config_matches_frozen_document(tmp_path) -> None:
    path = tmp_path / "core.yaml"
    write_core_config(path)
    loaded = load_core_config(path)
    expected = frozen_core_config()
    assert loaded["roster"] == expected["roster"]
    assert loaded["candidates"]["EWMA"][6]["id"] == "EWMA07"
    assert loaded["candidates"]["EWMA"][6]["decay"] == pytest.approx(0.94)
    assert core_config_sha256(path) == core_config_sha256(path)
