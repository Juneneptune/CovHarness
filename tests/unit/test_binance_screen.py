"""Focused tests for Binance SCREEN orchestration. No confirmatory ranking."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from covharness.inference.exceptions import DegenerateLossDifferentialError
from covharness.protocol.binance import BinanceBlock, build_binance_segmented_protocol
from covharness.protocol.binance_screen import (
    CONFIRM_START,
    DL_ELIGIBLE,
    ECON_ELIGIBLE,
    MAX_SCORING_TARGET_DATE,
    REQUIRED_SCREEN_TARGETS,
    REQUIRED_SELECTED_IDS,
    SCREEN_MODEL_ORDER,
    SHALLOW_ML,
    SUBBLOCK_BOUNDS,
    SUBBLOCK_SIZE,
    ScreenLeakError,
    assert_no_future_screen_fit_dates,
    average_ranks,
    econ_median_rank_finalist,
    freeze_screen_configuration,
    load_screen_config,
    representatives_from_screen_config,
    request_screen_schedule,
    screen_load_complete_checkpoint,
    screen_mcs,
    screen_spa,
    screen_write_checkpoint,
    verify_validation_selection,
)
from covharness.protocol.binance_validation import (
    CandidateRunResult,
    ensemble_covariances,
)
from covharness.protocol.exceptions import ConfirmLockedError


def _qlike_panel() -> tuple[np.ndarray, list[str]]:
    labels = list(SCREEN_MODEL_ORDER)
    losses = np.zeros((REQUIRED_SCREEN_TARGETS, len(labels)))
    lookup = {name: index for index, name in enumerate(labels)}
    # EWMA has the best median rank. HAR has the best full-sample mean.
    losses[:, lookup["EWMA"]] = np.concatenate(
        [np.full(125, 1.0), np.full(125, 1.0), np.full(125, 1.0), np.full(125, 10.0)]
    )
    losses[:, lookup["HAR-DRD"]] = 2.0
    losses[:, lookup["Ridge-DRD"]] = losses[:, lookup["HAR-DRD"]]
    losses[:, lookup["HARQ-DRD"]] = 3.0
    losses[:, lookup["RW"]] = 4.0
    losses[:, lookup["DCC"]] = 5.0
    losses[:, lookup["DCC-NL"]] = 6.0
    losses[:, lookup["LW-linear"]] = 7.0
    losses[:, lookup["LW-NL"]] = 8.0
    losses[:, lookup["XGBoost-DRD"]] = 0.5
    losses[:, lookup["LSTM-BEKK"]] = 9.0
    return losses, labels


def _spa_mcs_panel() -> tuple[np.ndarray, list[str]]:
    labels = list(SCREEN_MODEL_ORDER)
    rng = np.random.default_rng(20260913)
    losses = rng.normal(loc=2.0, scale=0.15, size=(REQUIRED_SCREEN_TARGETS, len(labels)))
    har = labels.index("HAR-DRD")
    ridge = labels.index("Ridge-DRD")
    losses[:, ridge] = losses[:, har]
    return losses, labels


def test_frozen_selected_configs_are_used(tmp_path: Path) -> None:
    verified = verify_validation_selection(
        Path("results/binance_validation_selection.json"),
        Path("results/binance_validation_candidate_summary.csv"),
    )
    assert verified["selected_ids"]["EWMA"] == "EWMA01"
    assert verified["selected_ids"]["Ridge-DRD"] == "RIDGE01"
    assert verified["selected_ids"]["XGBoost-DRD"] == "XGB08"
    assert verified["selected_ids"]["LSTM-BEKK"] == "LSTM19"
    yaml_path = tmp_path / "screen.yaml"
    freeze_screen_configuration(
        core_path=Path("configs/binance_open_data_core.yaml"),
        selection_path=Path("results/binance_validation_selection.json"),
        output_path=yaml_path,
        summary_path=Path("results/binance_validation_candidate_summary.csv"),
    )
    document = load_screen_config(yaml_path)
    rows = representatives_from_screen_config(document)
    assert {family: row["id"] for family, row in rows.items()} == {
        **{
            "RW": "RW01",
            "HAR-DRD": "HARDRD01",
            "HARQ-DRD": "HARQDRD01",
            "LW-linear": "LWLIN01",
            "LW-NL": "LWNL01",
            "DCC": "DCC01",
            "DCC-NL": "DCCNL01",
        },
        **REQUIRED_SELECTED_IDS,
    }
    text = yaml_path.read_text(encoding="utf-8")
    assert "EWMA02" not in text
    assert "RIDGE02" not in text
    assert "XGB01" not in text
    assert "LSTM01" not in text
    assert document["representatives"]["EWMA"]["id"] == "EWMA01"
    assert document["representatives"]["Ridge-DRD"]["lambda"] == 0.0
    assert document["allow_best_seed"] is False


def test_validation_grids_are_not_rerun(tmp_path: Path) -> None:
    yaml_path = tmp_path / "screen.yaml"
    freeze_screen_configuration(
        core_path=Path("configs/binance_open_data_core.yaml"),
        selection_path=Path("results/binance_validation_selection.json"),
        output_path=yaml_path,
        summary_path=Path("results/binance_validation_candidate_summary.csv"),
    )
    document = load_screen_config(yaml_path)
    for family in SCREEN_MODEL_ORDER:
        spec = document["representatives"][family]
        if family in ("XGBoost-DRD", "LSTM-BEKK"):
            assert "grid" not in spec
            assert spec["parameters"]["id"] == document["selected_ids"][family]
        else:
            assert spec["id"] == document["selected_ids"][family]


def test_screen_has_exactly_500_targets() -> None:
    schedule = request_screen_schedule()
    assert len(schedule.steps) == REQUIRED_SCREEN_TARGETS
    assert schedule.steps[0].origin.date() == dt.date(2023, 11, 29)
    assert schedule.steps[-1].target.date() == dt.date(2025, 4, 12)
    assert [index for index, step in enumerate(schedule.steps) if step.refit] == list(
        range(0, 500, 21)
    )


def test_default_confirm_remains_inaccessible() -> None:
    protocol = build_binance_segmented_protocol()
    with pytest.raises(ConfirmLockedError):
        protocol.confirm_targets()
    with pytest.raises(ConfirmLockedError):
        protocol.forecast_schedule(BinanceBlock.CONFIRM)
    request_screen_schedule(protocol)


def test_no_post_screen_target_enters_screen() -> None:
    schedule = request_screen_schedule()
    assert all(step.target.date() <= MAX_SCORING_TARGET_DATE for step in schedule.steps)
    assert all(step.target.date() < CONFIRM_START for step in schedule.steps)
    origin = dt.date(2025, 4, 11)
    with pytest.raises(ScreenLeakError, match="after origin"):
        assert_no_future_screen_fit_dates([origin, dt.date(2025, 4, 12)], origin)
    with pytest.raises(ScreenLeakError, match="exceeds 2025-04-11"):
        assert_no_future_screen_fit_dates([dt.date(2025, 4, 13)], dt.date(2025, 4, 13))
    assert_no_future_screen_fit_dates([dt.date(2025, 4, 10), origin], origin)


def test_four_subblocks_have_125_dates() -> None:
    assert len(SUBBLOCK_BOUNDS) == 4
    for start, stop in SUBBLOCK_BOUNDS:
        assert stop - start == SUBBLOCK_SIZE
    assert SUBBLOCK_BOUNDS[0] == (0, 125)
    assert SUBBLOCK_BOUNDS[-1] == (375, 500)


def test_median_rank_follows_frozen_rule() -> None:
    losses, labels = _qlike_panel()
    result = econ_median_rank_finalist(losses, labels)
    assert result["econ_finalist"] == "EWMA"
    assert result["robustness_full_mean_qlike_leader"] == "HAR-DRD"
    np.testing.assert_allclose(average_ranks([1.0, 1.0, 2.0]), [1.5, 1.5, 3.0])


def test_frobenius_cannot_change_headline_finalist() -> None:
    qlike, labels = _qlike_panel()
    frobenius = np.zeros_like(qlike)
    frobenius[:] = 9.0
    frobenius[:, labels.index("HAR-DRD")] = 0.1
    result = econ_median_rank_finalist(qlike, labels, frobenius=frobenius)
    assert result["econ_finalist"] == "EWMA"


def test_mean_loss_robustness_cannot_replace_headline() -> None:
    losses, labels = _qlike_panel()
    result = econ_median_rank_finalist(losses, labels)
    assert result["econ_finalist"] == "EWMA"
    assert result["robustness_full_mean_qlike_leader"] == "HAR-DRD"
    assert result["econ_finalist"] != result["robustness_full_mean_qlike_leader"]


def test_shallow_ml_cannot_occupy_headline_slots() -> None:
    losses, labels = _qlike_panel()
    result = econ_median_rank_finalist(losses, labels)
    assert result["econ_finalist"] not in SHALLOW_ML
    assert result["econ_finalist"] in ECON_ELIGIBLE
    assert DL_ELIGIBLE == ("LSTM-BEKK",)
    assert "XGBoost-DRD" not in ECON_ELIGIBLE
    assert "Ridge-DRD" not in ECON_ELIGIBLE


def test_lstm_ensemble_averages_covariances_before_loss() -> None:
    cubes = [np.stack([np.eye(2) * (0.2 + 0.01 * seed) for _ in range(5)]) for seed in range(5)]
    ensemble = ensemble_covariances(cubes)
    expected = np.mean(np.stack(cubes, axis=0), axis=0)
    np.testing.assert_allclose(ensemble, expected)


def test_best_seed_cannot_be_selected() -> None:
    losses, labels = _qlike_panel()
    result = econ_median_rank_finalist(losses, labels)
    assert result["econ_finalist"] != "LSTM-BEKK"
    assert DL_ELIGIBLE == ("LSTM-BEKK",)


def test_exact_zero_spa_benchmark_tie_is_recorded() -> None:
    losses, labels = _spa_mcs_panel()
    payload = screen_spa(losses, labels, benchmark="HAR-DRD", n_boot=200, seed=20260913)
    assert "Ridge-DRD" in payload["exact_benchmark_ties"]
    assert "Ridge-DRD" in payload["excluded_from_spa_statistic"]
    assert "Ridge-DRD" not in payload["alternative_labels"]
    assert payload["n_models_in_universe"] == 11
    assert payload["n_models_in_spa"] == 10


def test_nonzero_constant_spa_differential_still_raises() -> None:
    losses, labels = _spa_mcs_panel()
    losses[:, labels.index("HAR-DRD")] = 1.0
    losses[:, labels.index("HARQ-DRD")] = 2.0
    with pytest.raises(DegenerateLossDifferentialError, match="equal"):
        screen_spa(losses, labels, benchmark="HAR-DRD", n_boot=50, seed=20260913)


def test_mcs_receives_all_eleven_columns() -> None:
    losses, labels = _spa_mcs_panel()
    payload = screen_mcs(losses, labels, n_boot=200, seed=20260913)
    assert payload["primary"]["n_models"] == 11
    assert payload["companion"]["n_models"] == 11
    assert payload["primary"]["labels"] == labels
    assert "Ridge-DRD" in payload["primary"]["labels"]
    assert "HAR-DRD" in payload["primary"]["labels"]


def test_screen_execution_never_requests_confirm() -> None:
    protocol = build_binance_segmented_protocol()
    request_screen_schedule(protocol)
    with pytest.raises(ConfirmLockedError):
        protocol.forecast_schedule(BinanceBlock.CONFIRM)


def test_failed_econ_model_is_not_replaced() -> None:
    losses, labels = _qlike_panel()
    keep = [name for name in labels if name not in ("DCC", "DCC-NL")]
    subset = losses[:, [labels.index(name) for name in keep]]
    result = econ_median_rank_finalist(subset, keep)
    assert result["econ_finalist"] == "EWMA"
    assert "DCC" not in result["families"]
    assert "DCC-NL" not in result["families"]
    assert result["econ_finalist"] not in SHALLOW_ML


def test_partial_checkpoint_is_not_reused(tmp_path: Path) -> None:
    required = [dt.date(2023, 11, 30) + dt.timedelta(days=i) for i in range(500)]
    partial = CandidateRunResult(
        "EWMA",
        "EWMA01",
        None,
        False,
        10,
        None,
        None,
        0,
        490,
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
    screen_write_checkpoint(tmp_path, partial, panel_sha256="p", config_sha256="c")
    loaded = screen_load_complete_checkpoint(
        tmp_path,
        "EWMA",
        "EWMA01",
        None,
        panel_sha256="p",
        config_sha256="c",
        required_targets=required,
    )
    assert loaded is None
