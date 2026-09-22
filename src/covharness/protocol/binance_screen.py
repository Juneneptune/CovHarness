"""Binance SCREEN execution, SPA/MCS, and median-rank finalist selection.

Uses only the SCREEN block. CONFIRM remains locked. VALIDATION grids are not
rerun. Model mathematics are unchanged.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from covharness.data.binance_calendar import (
    BINANCE_ASSETS,
    HALT_DATE,
    SEGMENT_BOUNDS,
    segment_dates,
)
from covharness.inference.bootstrap import BOOTSTRAP_SEED, DEFAULT_N_BOOT, default_block_length
from covharness.inference.exceptions import DegenerateLossDifferentialError
from covharness.inference.mcs import PROCEDURE_MAX, PROCEDURE_RANGE, model_confidence_set
from covharness.inference.spa import superior_predictive_ability
from covharness.protocol.binance import (
    CORE_ROSTER,
    PRODUCTION_PANEL_SHA256,
    BinanceBlock,
    BinanceSegmentedProtocol,
    build_binance_segmented_protocol,
    load_core_config,
)
from covharness.protocol.binance_validation import (
    CandidateRunResult,
    FROZEN_CONFIG_SHA256,
    FrozenHashMismatchError,
    atomic_save_npz,
    atomic_write_text,
    build_model,
    checkpoint_stem,
    complete_support,
    ensemble_covariances,
    score_forecast_cube,
    sha256_file,
)
from covharness.protocol.exceptions import ConfirmLockedError
from covharness.protocol.runner import (
    RollingForecastRecord,
    _advance_at_origin,
    _fit_at_origin,
    build_origin_payload,
)

FROZEN_PANEL_SHA256 = PRODUCTION_PANEL_SHA256
FROZEN_CORE_CONFIG_SHA256 = FROZEN_CONFIG_SHA256
FROZEN_VALIDATION_SELECTION_SHA256 = (
    "f6d9830302b03afa2f3f45cd15576d70fbae4b96482af5c71d935822bc132f4b"
)
REQUIRED_SELECTED_IDS = {
    "EWMA": "EWMA01",
    "Ridge-DRD": "RIDGE01",
    "XGBoost-DRD": "XGB08",
    "LSTM-BEKK": "LSTM19",
}
FIXED_SELECTED_IDS = {
    "RW": "RW01",
    "HAR-DRD": "HARDRD01",
    "HARQ-DRD": "HARQDRD01",
    "LW-linear": "LWLIN01",
    "LW-NL": "LWNL01",
    "DCC": "DCC01",
    "DCC-NL": "DCCNL01",
}
SCREEN_MODEL_ORDER = CORE_ROSTER
ECON_ELIGIBLE = (
    "RW",
    "EWMA",
    "HAR-DRD",
    "HARQ-DRD",
    "LW-linear",
    "LW-NL",
    "DCC",
    "DCC-NL",
)
SHALLOW_ML = ("Ridge-DRD", "XGBoost-DRD")
DL_ELIGIBLE = ("LSTM-BEKK",)
REQUIRED_SCREEN_TARGETS = 500
SUBBLOCK_SIZE = 125
SUBBLOCK_BOUNDS = ((0, 125), (125, 250), (250, 375), (375, 500))
MAX_MODEL_OBSERVATION_DATE = dt.date(2025, 4, 11)
MAX_SCORING_TARGET_DATE = dt.date(2025, 4, 12)
CONFIRM_START = dt.date(2025, 4, 13)
HISTORY_B_START = SEGMENT_BOUNDS["HISTORY_B"][0]
SPA_BENCHMARK = "HAR-DRD"
MCS_MEMBERSHIP_ALPHA = 0.10
SEEDS = (0, 1, 2, 3, 4)
FROZEN_ASSETS = BINANCE_ASSETS
SCREEN_BRANCH = "binance_open_data_screen"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


class ScreenLeakError(RuntimeError):
    """A date outside the SCREEN bound was offered to a model or selector."""


@dataclass(frozen=True)
class ScreenView:
    """HISTORY_B plus SCREEN arrays. CONFIRM rows are omitted."""

    dates: tuple[dt.date, ...]
    calendar: pd.DatetimeIndex
    assets: tuple[str, ...]
    daily_returns: NDArray[np.floating]
    rcov: NDArray[np.floating]
    rq: NDArray[np.floating]
    panel_sha256: str
    core_config_sha256: str
    screen_config_sha256: str
    max_model_observation_date: dt.date
    max_scoring_target_date: dt.date


def verify_validation_selection(
    selection_path: Path,
    summary_path: Path | None = None,
) -> dict[str, object]:
    """Confirm the frozen VALIDATION IDs before any SCREEN access."""
    selection_path = Path(selection_path)
    selection_hash = sha256_file(selection_path)
    if selection_hash != FROZEN_VALIDATION_SELECTION_SHA256:
        raise FrozenHashMismatchError(
            f"VALIDATION selection SHA-256 {selection_hash} != "
            f"frozen {FROZEN_VALIDATION_SELECTION_SHA256}"
        )
    document = json.loads(selection_path.read_text(encoding="utf-8"))
    if document.get("selection_rule") != "mean_reduced_qlike_complete_250_validation_targets":
        raise FrozenHashMismatchError("VALIDATION selection rule drifted")
    if document.get("tie_rule") != "lexicographically_smaller_config_id":
        raise FrozenHashMismatchError("VALIDATION tie rule drifted")
    if document.get("secondary_loss") != "squared_frobenius_descriptive_only":
        raise FrozenHashMismatchError("Frobenius was not recorded as descriptive-only")
    text = json.dumps(document)
    if "SCREEN" in text or "CONFIRM" in text:
        raise ScreenLeakError("VALIDATION selection artifact mentions SCREEN or CONFIRM")
    families = document["families"]
    selected: dict[str, str] = {}
    for family, expected in {**FIXED_SELECTED_IDS, **REQUIRED_SELECTED_IDS}.items():
        row = families[family]
        got = row["selected_id"]
        if got != expected:
            raise FrozenHashMismatchError(f"{family} selected {got}, expected {expected}")
        if row.get("n_valid", 0) < 1:
            raise FrozenHashMismatchError(f"{family} lacks a valid VALIDATION candidate")
        selected[family] = expected
    if summary_path is not None:
        table = pd.read_csv(summary_path)
        for family, config_id in REQUIRED_SELECTED_IDS.items():
            match = table[
                (table["family"] == family)
                & (table["configuration_id"] == config_id)
                & (table["complete_support"] == True)  # noqa: E712
            ]
            if match.empty:
                raise FrozenHashMismatchError(f"{config_id} lacks complete VALIDATION support")
        lstm_seeds = table[
            (table["family"] == "LSTM-BEKK")
            & (table["configuration_id"] == "LSTM19")
            & (table["seed_scope"].astype(str).str.startswith("seed"))
        ]
        completed = set(lstm_seeds.loc[lstm_seeds["complete_support"] == True, "seed_scope"])  # noqa: E712
        required_seeds = {f"seed{seed}" for seed in SEEDS}
        if completed != required_seeds:
            raise FrozenHashMismatchError(f"LSTM19 seeds completed {sorted(completed)}")
    return {
        "selected_ids": selected,
        "selection_sha256": selection_hash,
        "selection_rule": document["selection_rule"],
        "tie_rule": document["tie_rule"],
    }


def _find_candidate_row(block: object, config_id: str) -> dict[str, object]:
    rows = block["grid"] if isinstance(block, dict) and "grid" in block else block
    for row in rows:
        if str(row["id"]) == config_id:
            return dict(row)
    raise FrozenHashMismatchError(f"frozen core config is missing {config_id}")


def freeze_screen_configuration(
    *,
    core_path: Path,
    selection_path: Path,
    output_path: Path,
    summary_path: Path | None = None,
    panel_sha256: str | None = None,
) -> str:
    """Write the SCREEN YAML from the frozen core file plus VALIDATION selection."""
    core_path = Path(core_path)
    selection_path = Path(selection_path)
    core_hash = sha256_file(core_path)
    if core_hash != FROZEN_CORE_CONFIG_SHA256:
        raise FrozenHashMismatchError("core configuration hash drifted")
    verified = verify_validation_selection(selection_path, summary_path)
    core = load_core_config(core_path)
    selected = verified["selected_ids"]
    representatives: dict[str, object] = {}
    for family in SCREEN_MODEL_ORDER:
        config_id = selected[family]
        raw = core["candidates"][family]
        if family == "XGBoost-DRD":
            representatives[family] = {
                "id": config_id,
                "fixed": dict(raw["fixed"]),
                "parameters": _find_candidate_row(raw, config_id),
            }
        elif family == "LSTM-BEKK":
            representatives[family] = {
                "id": config_id,
                "fixed": dict(raw["fixed"]),
                "parameters": _find_candidate_row(raw, config_id),
                "seeds": list(SEEDS),
                "seed_aggregation": "mean_ensemble",
                "allow_best_seed": False,
            }
        elif family in REQUIRED_SELECTED_IDS:
            representatives[family] = _find_candidate_row(raw, config_id)
        else:
            representatives[family] = {"id": config_id}
    document = {
        "branch": SCREEN_BRANCH,
        "version": "2026-09-21",
        "source_core_config": _repo_relative(core_path),
        "source_core_config_sha256": core_hash,
        "source_validation_selection": _repo_relative(selection_path),
        "source_validation_selection_sha256": verified["selection_sha256"],
        "production_panel_sha256": panel_sha256 or FROZEN_PANEL_SHA256,
        "assets": list(FROZEN_ASSETS),
        "confirm_locked": True,
        "pack_across_halt": False,
        "m": 250,
        "refit_cadence": 21,
        "screen": {
            "start": SEGMENT_BOUNDS["SCREEN"][0].isoformat(),
            "end": SEGMENT_BOUNDS["SCREEN"][1].isoformat(),
            "n_targets": REQUIRED_SCREEN_TARGETS,
            "first_origin": "2023-11-29",
            "first_window_start": HISTORY_B_START.isoformat(),
            "max_model_observation_date": MAX_MODEL_OBSERVATION_DATE.isoformat(),
            "max_scoring_target_date": MAX_SCORING_TARGET_DATE.isoformat(),
        },
        "primary_screen_loss": "reduced_qlike",
        "secondary_screen_loss": "squared_frobenius",
        "secondary_loss_selects": False,
        "seeds": list(SEEDS),
        "seed_aggregation": "mean_ensemble",
        "allow_best_seed": False,
        "spa": {
            "benchmark": SPA_BENCHMARK,
            "n_boot": DEFAULT_N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "block_length_rule": "max(2, floor(T**(1/3)))",
            "primary_recentering": "consistent",
            "exact_zero_benchmark_differential": "exact_benchmark_tie",
        },
        "mcs": {
            "n_boot": DEFAULT_N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "primary_procedure": PROCEDURE_RANGE,
            "companion_procedure": PROCEDURE_MAX,
            "membership_alpha": MCS_MEMBERSHIP_ALPHA,
            "report_alphas": [0.05, 0.10, 0.25],
        },
        "finalist_rule": {
            "headline_comparison": "one_dl_finalist_versus_one_econ_finalist",
            "dl_eligible": list(DL_ELIGIBLE),
            "econ_eligible": list(ECON_ELIGIBLE),
            "headline_ineligible": list(SHALLOW_ML),
            "method": "median_rank_four_125_qlike_blocks",
            "subblock_size": SUBBLOCK_SIZE,
            "subblock_bounds": [list(bound) for bound in SUBBLOCK_BOUNDS],
            "tie_break": [
                "lowest_median_block_rank",
                "lowest_mean_block_rank",
                "lexicographically_smaller_family_id",
            ],
            "full_sample_mean_qlike": "robustness_only",
            "frobenius_selects_finalist": False,
            "mcs_does_not_replace_finalist": True,
        },
        "roster": list(SCREEN_MODEL_ORDER),
        "representatives": representatives,
        "selected_ids": selected,
        "boundary_notes": {
            "EWMA01": "smallest_lambda_candidate_retained",
            "RIDGE01": "lambda_zero_nests_HAR_retained",
        },
    }
    text = yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True)
    atomic_write_text(output_path, text)
    return sha256_file(output_path)


def load_screen_config(path: Path) -> dict[str, object]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise FrozenHashMismatchError("SCREEN configuration must be a mapping")
    if document.get("branch") != SCREEN_BRANCH:
        raise FrozenHashMismatchError("SCREEN configuration branch drifted")
    if document.get("confirm_locked") is not True:
        raise FrozenHashMismatchError("SCREEN configuration must keep CONFIRM locked")
    if document.get("source_core_config_sha256") != FROZEN_CORE_CONFIG_SHA256:
        raise FrozenHashMismatchError("SCREEN configuration core hash drifted")
    if document.get("source_validation_selection_sha256") != FROZEN_VALIDATION_SELECTION_SHA256:
        raise FrozenHashMismatchError("SCREEN configuration selection hash drifted")
    selected = document["selected_ids"]
    for family, expected in {**FIXED_SELECTED_IDS, **REQUIRED_SELECTED_IDS}.items():
        if selected[family] != expected:
            raise FrozenHashMismatchError(f"SCREEN representative for {family} drifted")
    if list(document["roster"]) != list(SCREEN_MODEL_ORDER):
        raise FrozenHashMismatchError("SCREEN roster drifted")
    return document


def representatives_from_screen_config(document: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return one constructor row per frozen SCREEN family. Grids are not expanded."""
    block = document["representatives"]
    rows: dict[str, dict[str, object]] = {}
    for family in SCREEN_MODEL_ORDER:
        spec = block[family]
        if family in ("XGBoost-DRD", "LSTM-BEKK"):
            row = dict(spec["parameters"])
        else:
            row = dict(spec)
        if str(row["id"]) != document["selected_ids"][family]:
            raise FrozenHashMismatchError(f"{family} representative ID drifted")
        rows[family] = row
    return rows


def request_screen_schedule(protocol: BinanceSegmentedProtocol | None = None):
    """Return the SCREEN schedule only. CONFIRM stays locked."""
    protocol = protocol or build_binance_segmented_protocol()
    try:
        protocol.confirm_targets()
        raise ScreenLeakError("CONFIRM was accessible without unlock")
    except ConfirmLockedError:
        pass
    try:
        protocol.forecast_schedule(BinanceBlock.CONFIRM)
        raise ScreenLeakError("CONFIRM schedule was accessible without unlock")
    except ConfirmLockedError:
        pass
    schedule = protocol.forecast_schedule(BinanceBlock.SCREEN)
    if schedule.block != "SCREEN":
        raise ScreenLeakError("schedule block is not SCREEN")
    if len(schedule.steps) != REQUIRED_SCREEN_TARGETS:
        raise ScreenLeakError(
            f"SCREEN must have {REQUIRED_SCREEN_TARGETS} targets; got {len(schedule.steps)}"
        )
    first = schedule.steps[0]
    last = schedule.steps[-1]
    if first.origin.date() != dt.date(2023, 11, 29):
        raise ScreenLeakError("first SCREEN origin is not 2023-11-29")
    if last.target.date() != MAX_SCORING_TARGET_DATE:
        raise ScreenLeakError("last SCREEN target is not 2025-04-12")
    halt = pd.Timestamp(HALT_DATE)
    refit_positions = [index for index, step in enumerate(schedule.steps) if step.refit]
    if refit_positions != list(range(0, REQUIRED_SCREEN_TARGETS, 21)):
        raise ScreenLeakError("SCREEN refit positions drifted")
    for step in schedule.steps:
        if step.origin == halt or step.target == halt:
            raise ScreenLeakError("halt date entered the SCREEN schedule")
        if step.origin.date() > MAX_MODEL_OBSERVATION_DATE:
            raise ScreenLeakError("origin after 2025-04-11")
        if step.target.date() > MAX_SCORING_TARGET_DATE:
            raise ScreenLeakError("target after 2025-04-12")
        if step.target.date() >= CONFIRM_START or step.origin.date() >= CONFIRM_START:
            raise ScreenLeakError("CONFIRM date entered the SCREEN schedule")
        window = [stamp.date() for stamp in step.estimation_dates(schedule.calendar)]
        if any(day >= CONFIRM_START or day == HALT_DATE for day in window):
            raise ScreenLeakError("CONFIRM or halt date entered a SCREEN window")
        if min(window) < HISTORY_B_START:
            raise ScreenLeakError("SCREEN window crossed 2023-03-24")
    return schedule


def _index_lookup(dates: Sequence[dt.date]) -> dict[dt.date, int]:
    return {day: index for index, day in enumerate(dates)}


def load_screen_view(
    panel_path: Path,
    core_path: Path,
    screen_config_path: Path,
    protocol: BinanceSegmentedProtocol | None = None,
) -> ScreenView:
    """Slice HISTORY_B plus SCREEN from the production NPZ. No CONFIRM rows."""
    panel_hash = sha256_file(panel_path)
    core_hash = sha256_file(core_path)
    if panel_hash != FROZEN_PANEL_SHA256:
        raise FrozenHashMismatchError("production panel hash drifted")
    if core_hash != FROZEN_CORE_CONFIG_SHA256:
        raise FrozenHashMismatchError("core configuration hash drifted")
    screen_hash = sha256_file(screen_config_path)
    load_screen_config(screen_config_path)
    protocol = protocol or build_binance_segmented_protocol()
    schedule = request_screen_schedule(protocol)
    blob = np.load(panel_path)
    assets = tuple(str(name) for name in blob["assets"].tolist())
    if assets != FROZEN_ASSETS:
        raise ScreenLeakError(f"asset order drifted: {assets}")
    npz_dates = tuple(dt.date.fromisoformat(text) for text in blob["dates"].tolist())
    lookup = _index_lookup(npz_dates)
    segments = segment_dates()
    allowed = tuple(day for day in list(segments["HISTORY_B"]) + list(segments["SCREEN"]))
    selected: list[int] = []
    for day in allowed:
        if day >= CONFIRM_START or day == HALT_DATE or day > MAX_SCORING_TARGET_DATE:
            raise ScreenLeakError(f"{day.isoformat()} is outside the SCREEN view")
        selected.append(lookup[day])
    dates = tuple(npz_dates[index] for index in selected)
    if dates[0] != HISTORY_B_START or dates[-1] != MAX_SCORING_TARGET_DATE:
        raise ScreenLeakError("SCREEN view bounds drifted")
    if any(day >= CONFIRM_START for day in dates):
        raise ScreenLeakError("CONFIRM dates leaked into the SCREEN view")
    calendar = pd.DatetimeIndex([pd.Timestamp(day) for day in dates])
    origins = [step.origin.date() for step in schedule.steps]
    if max(origins) != MAX_MODEL_OBSERVATION_DATE:
        raise ScreenLeakError("max origin is not 2025-04-11")
    return ScreenView(
        dates=dates,
        calendar=calendar,
        assets=assets,
        daily_returns=np.array(blob["daily_returns"][selected], dtype=float, copy=True),
        rcov=np.array(blob["rcov_5min"][selected], dtype=float, copy=True),
        rq=np.array(blob["rq_per_asset"][selected], dtype=float, copy=True),
        panel_sha256=panel_hash,
        core_config_sha256=core_hash,
        screen_config_sha256=screen_hash,
        max_model_observation_date=max(origins),
        max_scoring_target_date=max(dates),
    )


def assert_no_future_screen_fit_dates(window_dates: Sequence[dt.date], origin: dt.date) -> None:
    """Reject a fit/update window that reads past the origin or into CONFIRM."""
    if origin > MAX_MODEL_OBSERVATION_DATE:
        raise ScreenLeakError(f"origin {origin.isoformat()} exceeds 2025-04-11")
    if any(day > origin for day in window_dates):
        raise ScreenLeakError("window date after origin")
    if any(day >= CONFIRM_START for day in window_dates):
        raise ScreenLeakError("CONFIRM date entered a model window")
    if HALT_DATE in window_dates:
        raise ScreenLeakError("halt date entered a model window")
    if any(day < HISTORY_B_START for day in window_dates):
        raise ScreenLeakError("pre-HISTORY_B date entered a SCREEN window")


def _proxy_cube(view: ScreenView, schedule) -> NDArray[np.floating]:
    lookup = _index_lookup(view.dates)
    return np.stack([view.rcov[lookup[step.target.date()]] for step in schedule.steps], axis=0)


def run_scheduled_screen_forecasts(
    *,
    model,
    schedule,
    view: ScreenView,
) -> tuple[list[RollingForecastRecord], NDArray[np.bool_], int, dict[str, object]]:
    """Advance one SCREEN representative. Windows cannot enter CONFIRM."""
    records: list[RollingForecastRecord] = []
    repaired_flags: list[bool] = []
    actions: list[str] = []
    min_eigs: list[float] = []
    pd_flags: list[bool] = []
    fit_failures = 0
    for step in schedule.steps:
        window_dates = [stamp.date() for stamp in step.estimation_dates(schedule.calendar)]
        assert_no_future_screen_fit_dates(window_dates, step.origin.date())
        local = _screen_local_step(step, view.calendar)
        payload = build_origin_payload(
            step=local,
            calendar=view.calendar,
            daily_returns=view.daily_returns,
            realized_covariances=view.rcov,
            realized_quarticity=view.rq,
        )
        try:
            action = _fit_at_origin(model, payload) if step.refit else _advance_at_origin(model, payload)
            forecast = model.forecast()
        except Exception as exc:  # noqa: BLE001
            fit_failures += 1
            raise RuntimeError(
                f"model execution failed at origin {step.origin.date()}: {exc}"
            ) from exc
        repaired = bool(forecast.identity.configuration.get("repaired", False))
        repaired_flags.append(repaired)
        actions.append(action.value if hasattr(action, "value") else str(action))
        min_eigs.append(float(forecast.diagnostics.min_eigenvalue))
        pd_flags.append(bool(forecast.diagnostics.positive_definite))
        records.append(
            RollingForecastRecord(
                model_name=forecast.identity.name,
                origin=step.origin,
                target=step.target,
                refit=bool(step.refit),
                action=action,
                covariance=np.array(forecast.matrix, dtype=float, copy=True),
            )
        )
    extras = {
        "actions": tuple(actions),
        "min_eigenvalues": np.array(min_eigs, dtype=float),
        "positive_definite": np.array(pd_flags, dtype=bool),
    }
    return records, np.array(repaired_flags, dtype=bool), fit_failures, extras


def _screen_local_step(step, calendar: pd.DatetimeIndex):
    """Reindex a confirmatory-calendar step onto the HISTORY_B+SCREEN view."""
    origin_index = int(calendar.get_loc(step.origin))
    target_index = int(calendar.get_loc(step.target))
    window_start_index = origin_index - int(step.m) + 1
    return step.__class__(
        origin=step.origin,
        target=step.target,
        window_start=calendar[window_start_index],
        origin_index=origin_index,
        target_index=target_index,
        window_start_index=window_start_index,
        window_end_index=origin_index + 1,
        refit=bool(step.refit),
        m=int(step.m),
    )


def evaluate_screen_records(
    family: str,
    config_id: str,
    seed: int | None,
    records: Sequence[RollingForecastRecord],
    proxies: NDArray[np.floating],
    repaired: NDArray[np.bool_],
    fit_failures: int,
    runtime_seconds: float,
    failure_reason: str | None = None,
    extras: Mapping[str, object] | None = None,
) -> CandidateRunResult:
    n_targets = len(records)
    if n_targets == 0:
        return CandidateRunResult(
            family=family,
            config_id=config_id,
            seed=seed,
            complete_support=False,
            n_targets=0,
            mean_qlike=None,
            mean_frobenius=None,
            repair_count=0,
            qlike_failures=REQUIRED_SCREEN_TARGETS,
            frobenius_failures=REQUIRED_SCREEN_TARGETS,
            nonfinite_forecasts=0,
            fit_failures=fit_failures,
            failure_reason=failure_reason or "no forecasts",
            runtime_seconds=runtime_seconds,
            origins=(),
            targets=(),
            forecasts=None,
            qlike=None,
            frobenius=None,
            repaired=None,
            refit=None,
        )
    forecasts = np.stack([record.covariance for record in records], axis=0)
    qlike, frobenius, qlike_fail, frobenius_fail = score_forecast_cube(forecasts, proxies)
    finite_forecasts = int(np.isfinite(forecasts).all(axis=(1, 2)).sum())
    support = complete_support(qlike, REQUIRED_SCREEN_TARGETS) and fit_failures == 0
    reason = failure_reason
    if not support and reason is None:
        reason = "incomplete primary-loss support"
    extras = extras or {}
    return CandidateRunResult(
        family=family,
        config_id=config_id,
        seed=seed,
        complete_support=support,
        n_targets=n_targets,
        mean_qlike=float(np.mean(qlike)) if support else None,
        mean_frobenius=float(np.mean(frobenius)) if np.isfinite(frobenius).all() else None,
        repair_count=int(repaired.sum()) if repaired is not None else 0,
        qlike_failures=qlike_fail,
        frobenius_failures=frobenius_fail,
        nonfinite_forecasts=n_targets - finite_forecasts,
        fit_failures=fit_failures,
        failure_reason=reason,
        runtime_seconds=runtime_seconds,
        origins=tuple(record.origin.date() for record in records),
        targets=tuple(record.target.date() for record in records),
        forecasts=forecasts,
        qlike=qlike,
        frobenius=frobenius,
        repaired=repaired,
        refit=np.array([record.refit for record in records], dtype=bool),
        actions=tuple(str(item) for item in extras.get("actions", ())),
        min_eigenvalues=np.array(extras["min_eigenvalues"], dtype=float)
        if extras.get("min_eigenvalues") is not None
        else None,
        positive_definite=np.array(extras["positive_definite"], dtype=bool)
        if extras.get("positive_definite") is not None
        else None,
    )


def screen_write_checkpoint(
    directory: Path,
    result: CandidateRunResult,
    *,
    panel_sha256: str,
    config_sha256: str,
) -> Path:
    directory = Path(directory)
    stem = checkpoint_stem(result.family, result.config_id, result.seed)
    complete = bool(
        result.complete_support
        and result.n_targets == REQUIRED_SCREEN_TARGETS
        and result.forecasts is not None
    )
    payload = {
        "panel_sha256": panel_sha256,
        "config_sha256": config_sha256,
        "family": result.family,
        "config_id": result.config_id,
        "seed": result.seed,
        "complete": complete,
        "n_targets": result.n_targets,
        "complete_support": result.complete_support,
        "mean_qlike": result.mean_qlike,
        "mean_frobenius": result.mean_frobenius,
        "repair_count": result.repair_count,
        "qlike_failures": result.qlike_failures,
        "frobenius_failures": result.frobenius_failures,
        "nonfinite_forecasts": result.nonfinite_forecasts,
        "fit_failures": result.fit_failures,
        "failure_reason": result.failure_reason,
        "runtime_seconds": result.runtime_seconds,
        "max_model_observation_date": MAX_MODEL_OBSERVATION_DATE.isoformat(),
        "max_scoring_target_date": MAX_SCORING_TARGET_DATE.isoformat(),
    }
    json_path = directory / f"{stem}.json"
    npz_path = directory / f"{stem}.npz"
    if result.forecasts is not None:
        extras: dict[str, object] = {
            "forecasts": result.forecasts,
            "qlike": result.qlike,
            "frobenius": result.frobenius,
            "repaired": result.repaired,
            "refit": result.refit,
            "origins": np.array([day.isoformat() for day in result.origins], dtype="U10"),
            "targets": np.array([day.isoformat() for day in result.targets], dtype="U10"),
        }
        atomic_save_npz(npz_path, **extras)
        payload["artifact"] = str(npz_path)
        payload["artifact_sha256"] = sha256_file(npz_path)
    atomic_write_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return json_path


def screen_load_complete_checkpoint(
    directory: Path,
    family: str,
    config_id: str,
    seed: int | None,
    *,
    panel_sha256: str,
    config_sha256: str,
    required_targets: Sequence[dt.date],
) -> CandidateRunResult | None:
    json_path = Path(directory) / f"{checkpoint_stem(family, config_id, seed)}.json"
    if not json_path.is_file():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("panel_sha256") != panel_sha256 or payload.get("config_sha256") != config_sha256:
        return None
    if payload.get("complete") is not True:
        return None
    if int(payload.get("n_targets", 0)) != REQUIRED_SCREEN_TARGETS:
        return None
    artifact = payload.get("artifact")
    if not artifact or not Path(artifact).is_file():
        return None
    blob = np.load(artifact)
    targets = tuple(dt.date.fromisoformat(text) for text in blob["targets"].tolist())
    if targets != tuple(required_targets):
        return None
    qlike = np.array(blob["qlike"], dtype=float)
    support = bool(payload.get("complete_support")) and complete_support(
        qlike, REQUIRED_SCREEN_TARGETS
    )
    return CandidateRunResult(
        family=family,
        config_id=config_id,
        seed=seed,
        complete_support=support,
        n_targets=int(payload["n_targets"]),
        mean_qlike=payload.get("mean_qlike"),
        mean_frobenius=payload.get("mean_frobenius"),
        repair_count=int(payload.get("repair_count", 0)),
        qlike_failures=int(payload.get("qlike_failures", 0)),
        frobenius_failures=int(payload.get("frobenius_failures", 0)),
        nonfinite_forecasts=int(payload.get("nonfinite_forecasts", 0)),
        fit_failures=int(payload.get("fit_failures", 0)),
        failure_reason=payload.get("failure_reason"),
        runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
        origins=tuple(dt.date.fromisoformat(text) for text in blob["origins"].tolist()),
        targets=targets,
        forecasts=np.array(blob["forecasts"], dtype=float),
        qlike=qlike,
        frobenius=np.array(blob["frobenius"], dtype=float),
        repaired=np.array(blob["repaired"], dtype=bool),
        refit=np.array(blob["refit"], dtype=bool),
    )


def run_one_screen_configuration(
    *,
    family: str,
    row: Mapping[str, object],
    seed: int | None,
    view: ScreenView,
    schedule,
    checkpoint_dir: Path,
    model_factory: Callable[..., object] | None = None,
) -> CandidateRunResult:
    config_id = str(row["id"])
    required_targets = [step.target.date() for step in schedule.steps]
    reused = screen_load_complete_checkpoint(
        checkpoint_dir,
        family,
        config_id,
        seed,
        panel_sha256=view.panel_sha256,
        config_sha256=view.screen_config_sha256,
        required_targets=required_targets,
    )
    if reused is not None:
        return reused
    factory = model_factory or build_model
    started = time.perf_counter()
    try:
        model = factory(family, row, seed)
        records, repaired, fit_failures, extras = run_scheduled_screen_forecasts(
            model=model, schedule=schedule, view=view
        )
        result = evaluate_screen_records(
            family,
            config_id,
            seed,
            records,
            _proxy_cube(view, schedule),
            repaired,
            fit_failures,
            time.perf_counter() - started,
            extras=extras,
        )
    except Exception as exc:  # noqa: BLE001
        result = CandidateRunResult(
            family=family,
            config_id=config_id,
            seed=seed,
            complete_support=False,
            n_targets=0,
            mean_qlike=None,
            mean_frobenius=None,
            repair_count=0,
            qlike_failures=REQUIRED_SCREEN_TARGETS,
            frobenius_failures=REQUIRED_SCREEN_TARGETS,
            nonfinite_forecasts=0,
            fit_failures=1,
            failure_reason=f"{type(exc).__name__}: {exc}",
            runtime_seconds=time.perf_counter() - started,
            origins=(),
            targets=(),
            forecasts=None,
            qlike=None,
            frobenius=None,
            repaired=None,
            refit=None,
        )
    screen_write_checkpoint(
        checkpoint_dir,
        result,
        panel_sha256=view.panel_sha256,
        config_sha256=view.screen_config_sha256,
    )
    return result


def lstm_screen_ensemble(
    seed_results: Sequence[CandidateRunResult],
    proxies: NDArray[np.floating],
) -> CandidateRunResult:
    if len(seed_results) != 5:
        raise ValueError("LSTM SCREEN ensemble requires five seed results")
    config_id = seed_results[0].config_id
    if any(not item.complete_support or item.forecasts is None for item in seed_results):
        return CandidateRunResult(
            family="LSTM-BEKK",
            config_id=config_id,
            seed=None,
            complete_support=False,
            n_targets=max(item.n_targets for item in seed_results),
            mean_qlike=None,
            mean_frobenius=None,
            repair_count=sum(item.repair_count for item in seed_results),
            qlike_failures=REQUIRED_SCREEN_TARGETS,
            frobenius_failures=0,
            nonfinite_forecasts=0,
            fit_failures=sum(item.fit_failures for item in seed_results),
            failure_reason="failed seed invalidates configuration",
            runtime_seconds=sum(item.runtime_seconds for item in seed_results),
            origins=seed_results[0].origins,
            targets=seed_results[0].targets,
            forecasts=None,
            qlike=None,
            frobenius=None,
            repaired=None,
            refit=seed_results[0].refit,
        )
    ordered = sorted(seed_results, key=lambda item: int(item.seed))
    ensemble = ensemble_covariances([item.forecasts for item in ordered])
    qlike, frobenius, qlike_fail, frobenius_fail = score_forecast_cube(ensemble, proxies)
    support = complete_support(qlike, REQUIRED_SCREEN_TARGETS)
    return CandidateRunResult(
        family="LSTM-BEKK",
        config_id=config_id,
        seed=None,
        complete_support=support,
        n_targets=int(ensemble.shape[0]),
        mean_qlike=float(np.mean(qlike)) if support else None,
        mean_frobenius=float(np.mean(frobenius)) if np.isfinite(frobenius).all() else None,
        repair_count=sum(item.repair_count for item in ordered),
        qlike_failures=qlike_fail,
        frobenius_failures=frobenius_fail,
        nonfinite_forecasts=int((~np.isfinite(ensemble).all(axis=(1, 2))).sum()),
        fit_failures=0,
        failure_reason=None if support else "ensemble incomplete primary-loss support",
        runtime_seconds=sum(item.runtime_seconds for item in ordered),
        origins=ordered[0].origins,
        targets=ordered[0].targets,
        forecasts=ensemble,
        qlike=qlike,
        frobenius=frobenius,
        repaired=np.zeros(ensemble.shape[0], dtype=bool),
        refit=ordered[0].refit,
    )


def average_ranks(values: Sequence[float]) -> NDArray[np.floating]:
    """Lower loss receives rank 1. Exact ties receive the average of occupied ranks."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.shape[0], dtype=float)
    index = 0
    n_items = int(array.shape[0])
    while index < n_items:
        stop = index
        while stop + 1 < n_items and array[order[stop + 1]] == array[order[index]]:
            stop += 1
        assigned = 0.5 * ((index + 1) + (stop + 1))
        ranks[order[index : stop + 1]] = assigned
        index = stop + 1
    return ranks


def econ_median_rank_finalist(
    qlike: NDArray[np.floating],
    labels: Sequence[str],
    *,
    frobenius: NDArray[np.floating] | None = None,
) -> dict[str, object]:
    """Select the econometric finalist from four QLIKE sub-block ranks."""
    if qlike.shape[0] != REQUIRED_SCREEN_TARGETS:
        raise ValueError("QLIKE panel must have 500 rows")
    if qlike.shape[1] != len(labels):
        raise ValueError("QLIKE panel must be (500, n_models)")
    if frobenius is not None and frobenius.shape != qlike.shape:
        raise ValueError("Frobenius panel shape must match QLIKE")
    econ_index = [i for i, name in enumerate(labels) if name in ECON_ELIGIBLE]
    if len(econ_index) < 1:
        raise ValueError("no complete econometric model remains for finalist selection")
    block_means = []
    block_ranks = []
    for start, stop in SUBBLOCK_BOUNDS:
        if stop - start != SUBBLOCK_SIZE:
            raise ValueError("each SCREEN sub-block must contain 125 targets")
        means = qlike[start:stop, econ_index].mean(axis=0)
        block_means.append(means)
        block_ranks.append(average_ranks(means))
    ranks = np.stack(block_ranks, axis=1)
    median_ranks = np.median(ranks, axis=1)
    mean_ranks = ranks.mean(axis=1)
    names = [labels[i] for i in econ_index]
    order = sorted(
        range(len(names)),
        key=lambda i: (float(median_ranks[i]), float(mean_ranks[i]), names[i]),
    )
    winner = names[order[0]]
    full_means = qlike[:, econ_index].mean(axis=0)
    robustness = names[int(np.argmin(full_means))]
    if frobenius is not None:
        _ignored = frobenius[:, econ_index].mean(axis=0)
        del _ignored
    return {
        "econ_finalist": winner,
        "robustness_full_mean_qlike_leader": robustness,
        "rule": "median_rank_four_125_qlike_blocks",
        "tie_break": [
            "lowest_median_block_rank",
            "lowest_mean_block_rank",
            "lexicographically_smaller_family_id",
        ],
        "families": names,
        "block_means": {
            f"block_{index + 1}": {
                name: float(block_means[index][pos]) for pos, name in enumerate(names)
            }
            for index in range(4)
        },
        "block_ranks": {
            f"block_{index + 1}": {
                name: float(block_ranks[index][pos]) for pos, name in enumerate(names)
            }
            for index in range(4)
        },
        "median_ranks": {name: float(median_ranks[pos]) for pos, name in enumerate(names)},
        "mean_ranks": {name: float(mean_ranks[pos]) for pos, name in enumerate(names)},
        "full_screen_mean_qlike": {
            name: float(full_means[pos]) for pos, name in enumerate(names)
        },
    }


def screen_spa(
    losses: NDArray[np.floating],
    labels: Sequence[str],
    *,
    benchmark: str = SPA_BENCHMARK,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Hansen SPA with exact-zero benchmark ties excluded from studentization."""
    labels = list(labels)
    if losses.shape[0] != REQUIRED_SCREEN_TARGETS or losses.shape[1] != len(labels):
        raise ValueError("SPA loss panel must be (500, n_models)")
    bench_index = labels.index(benchmark)
    exact_ties: list[str] = []
    keep = [bench_index]
    for index, label in enumerate(labels):
        if index == bench_index:
            continue
        differential = losses[:, bench_index] - losses[:, index]
        if np.all(differential == 0.0):
            exact_ties.append(label)
            continue
        if np.all(differential == differential[0]) and differential[0] != 0.0:
            raise DegenerateLossDifferentialError(
                "all benchmark-versus-alternative observations are equal. "
                "SPA is undefined. No jitter was added."
            )
        keep.append(index)
    subset = losses[:, keep]
    result = superior_predictive_ability(
        subset,
        benchmark_index=0,
        benchmark_label=benchmark,
        n_boot=n_boot,
        seed=seed,
    )
    alternative_labels = [labels[index] for index in keep if index != bench_index]
    return {
        "benchmark": benchmark,
        "n_models_in_universe": len(labels),
        "n_models_in_spa": int(result.n_models),
        "exact_benchmark_ties": exact_ties,
        "excluded_from_spa_statistic": exact_ties,
        "alternative_labels": alternative_labels,
        "statistic": result.statistic,
        "p_value_lower": result.p_value_lower,
        "p_value_consistent": result.p_value_consistent,
        "p_value_upper": result.p_value_upper,
        "mean_differentials": {
            label: float(value)
            for label, value in zip(alternative_labels, result.mean_differentials)
        },
        "studentized_statistics": {
            label: float(value)
            for label, value in zip(alternative_labels, result.studentized_statistics)
        },
        "block_length": result.block_length,
        "n_boot": result.n_boot,
        "seed": result.seed,
        "alpha": result.alpha,
        "bootstrap_method": result.bootstrap_method,
        "primary_recentering": "consistent",
    }


def _mcs_payload(result, labels: Sequence[str]) -> dict[str, object]:
    membership = {
        f"{alpha:.2f}": [
            labels[index]
            for index, keep in enumerate(result.mcs_pvalues >= float(alpha))
            if keep
        ]
        for alpha in (0.05, 0.10, 0.25)
    }
    return {
        "procedure": result.procedure,
        "alpha": result.alpha,
        "n_models": result.n_models,
        "n_observations": result.n_observations,
        "labels": list(labels),
        "mean_losses": {label: float(value) for label, value in zip(labels, result.mean_losses)},
        "mcs_pvalues": {label: float(value) for label, value in zip(labels, result.mcs_pvalues)},
        "in_mcs_primary_alpha": {
            label: bool(flag) for label, flag in zip(labels, result.in_mcs)
        },
        "membership_by_alpha": membership,
        "elimination_order": [labels[int(index)] for index in result.elimination_order],
        "block_length": result.block_length,
        "n_boot": result.n_boot,
        "seed": result.seed,
        "bootstrap_method": result.bootstrap_method,
    }


def screen_mcs(
    losses: NDArray[np.floating],
    labels: Sequence[str],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    if losses.shape[1] != len(labels):
        raise ValueError("MCS must receive every supplied model column")
    if losses.shape[1] < 2:
        raise ValueError("SCREEN MCS requires at least two loss columns")
    primary = model_confidence_set(
        losses,
        procedure=PROCEDURE_RANGE,
        alpha=MCS_MEMBERSHIP_ALPHA,
        n_boot=n_boot,
        seed=seed,
    )
    companion = model_confidence_set(
        losses,
        procedure=PROCEDURE_MAX,
        alpha=MCS_MEMBERSHIP_ALPHA,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "primary": _mcs_payload(primary, labels),
        "companion": _mcs_payload(companion, labels),
    }


def _result_row(result: CandidateRunResult, seed_scope: str) -> dict[str, object]:
    return {
        "family": result.family,
        "configuration_id": result.config_id,
        "seed_scope": seed_scope,
        "complete_support": result.complete_support,
        "n_targets": result.n_targets,
        "mean_qlike": result.mean_qlike,
        "median_qlike": float(np.median(result.qlike)) if result.qlike is not None else None,
        "mean_frobenius": result.mean_frobenius,
        "median_frobenius": float(np.median(result.frobenius))
        if result.frobenius is not None
        else None,
        "repair_count": result.repair_count,
        "qlike_failures": result.qlike_failures,
        "frobenius_failures": result.frobenius_failures,
        "nonfinite_forecasts": result.nonfinite_forecasts,
        "fit_failures": result.fit_failures,
        "runtime_seconds": result.runtime_seconds,
        "failure_reason": result.failure_reason,
    }


def run_binance_screen(
    *,
    panel_path: Path,
    core_path: Path,
    selection_path: Path,
    screen_config_path: Path,
    output_dir: Path,
    summary_path: Path | None = None,
    model_factory: Callable[..., object] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run SCREEN selection. CONFIRM is not requested."""
    started = time.perf_counter()
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "binance_screen_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda message: print(message, flush=True))
    if not Path(screen_config_path).is_file():
        raise FrozenHashMismatchError("SCREEN configuration artifact is missing")
    verify_validation_selection(selection_path, summary_path)
    protocol = build_binance_segmented_protocol()
    view = load_screen_view(panel_path, core_path, screen_config_path, protocol)
    schedule = request_screen_schedule(protocol)
    document = load_screen_config(screen_config_path)
    rows = representatives_from_screen_config(document)
    proxies = _proxy_cube(view, schedule)
    family_results: dict[str, CandidateRunResult] = {}
    seed_rows: list[CandidateRunResult] = []
    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for family in SCREEN_MODEL_ORDER:
        log(f"starting SCREEN family {family}")
        row = rows[family]
        if family == "LSTM-BEKK":
            per_seed: list[CandidateRunResult] = []
            for seed in SEEDS:
                log(f"  {row['id']} seed {seed}")
                result = run_one_screen_configuration(
                    family=family,
                    row=row,
                    seed=seed,
                    view=view,
                    schedule=schedule,
                    checkpoint_dir=checkpoint_dir,
                    model_factory=model_factory,
                )
                per_seed.append(result)
                seed_rows.append(result)
                summary_rows.append(_result_row(result, f"seed{seed}"))
                if not result.complete_support:
                    failures.append(_result_row(result, f"seed{seed}"))
            ensemble = lstm_screen_ensemble(per_seed, proxies)
            family_results[family] = ensemble
            summary_rows.append(_result_row(ensemble, "ensemble"))
            if not ensemble.complete_support:
                failures.append(_result_row(ensemble, "ensemble"))
        else:
            log(f"  {row['id']}")
            result = run_one_screen_configuration(
                family=family,
                row=row,
                seed=None,
                view=view,
                schedule=schedule,
                checkpoint_dir=checkpoint_dir,
                model_factory=model_factory,
            )
            family_results[family] = result
            summary_rows.append(_result_row(result, "deterministic"))
            if not result.complete_support:
                failures.append(_result_row(result, "deterministic"))

    labels = list(SCREEN_MODEL_ORDER)
    complete_labels = [
        name
        for name in labels
        if family_results[name].complete_support and family_results[name].qlike is not None
    ]
    failed_families = [name for name in labels if name not in complete_labels]
    if failed_families:
        log(f"incomplete SCREEN families {failed_families}")
    if len(complete_labels) < 2:
        raise RuntimeError(f"too few complete SCREEN models for SPA/MCS: {complete_labels}")
    qlike = np.column_stack([family_results[name].qlike for name in complete_labels])
    frobenius = np.column_stack([family_results[name].frobenius for name in complete_labels])
    if SPA_BENCHMARK not in complete_labels:
        raise RuntimeError("SPA benchmark HAR-DRD lacks complete SCREEN support")
    spa_qlike = screen_spa(qlike, complete_labels)
    spa_frobenius = screen_spa(frobenius, complete_labels)
    mcs_qlike = screen_mcs(qlike, complete_labels)
    mcs_frobenius = screen_mcs(frobenius, complete_labels)
    finalist = econ_median_rank_finalist(qlike, complete_labels, frobenius=frobenius)
    dl_finalist = "LSTM-BEKK"
    econ_finalist = str(finalist["econ_finalist"])

    def _mcs_status(payload: dict[str, object], family: str) -> dict[str, object]:
        pvalues = payload["primary"]["mcs_pvalues"]
        if family not in pvalues:
            return {
                "in_mcs_alpha_0.10": False,
                "mcs_pvalue": None,
                "excluded_incomplete": True,
            }
        return {
            "in_mcs_alpha_0.10": family in payload["primary"]["membership_by_alpha"]["0.10"],
            "mcs_pvalue": pvalues[family],
        }

    # Descriptive argsort positions. Exact ties are not averaged and are not distinct performance.
    qlike_rank = {
        complete_labels[index]: int(rank) + 1
        for index, rank in enumerate(np.argsort(np.argsort(qlike.mean(axis=0))))
    }
    frob_rank = {
        complete_labels[index]: int(rank) + 1
        for index, rank in enumerate(np.argsort(np.argsort(frobenius.mean(axis=0))))
    }

    model_table = []
    for family in labels:
        item = family_results[family]
        model_table.append(
            {
                "family": family,
                "configuration": item.config_id,
                "n_targets": item.n_targets,
                "complete_support": item.complete_support,
                "mean_qlike": item.mean_qlike,
                "median_qlike": float(np.median(item.qlike)) if item.qlike is not None else None,
                "mean_frobenius": item.mean_frobenius,
                "median_frobenius": float(np.median(item.frobenius))
                if item.frobenius is not None
                else None,
                "repair_count": item.repair_count,
                "fit_failures": item.fit_failures,
                "runtime_seconds": item.runtime_seconds,
                "descriptive_qlike_rank": qlike_rank.get(family),
                "descriptive_frobenius_rank": frob_rank.get(family),
                "status": "complete" if item.complete_support else "failed",
            }
        )

    lstm_diag = None
    if seed_rows:
        qlike_seeds = [item.mean_qlike for item in seed_rows]
        frob_seeds = [item.mean_frobenius for item in seed_rows]
        lstm_diag = {
            "configuration_id": "LSTM19",
            "seeds_complete": [bool(item.complete_support) for item in seed_rows],
            "seed_mean_qlike": qlike_seeds,
            "seed_mean_frobenius": frob_seeds,
            "seed_qlike_range": [min(qlike_seeds), max(qlike_seeds)],
            "seed_frobenius_range": [min(frob_seeds), max(frob_seeds)],
            "ensemble_mean_qlike": family_results["LSTM-BEKK"].mean_qlike,
            "ensemble_mean_frobenius": family_results["LSTM-BEKK"].mean_frobenius,
        }

    ridge_exact_har_tie = "Ridge-DRD" in spa_qlike["exact_benchmark_ties"]
    target_dates = [step.target.date().isoformat() for step in schedule.steps]
    refit_positions = [index for index, step in enumerate(schedule.steps) if step.refit]
    finalists = {
        "dl_finalist": dl_finalist,
        "failed_families": failed_families,
        "spa_mcs_universe": complete_labels,
        "failed_family_reasons": {
            name: family_results[name].failure_reason for name in failed_families
        },
        "dl_finalist_configuration": family_results[dl_finalist].config_id,
        "dl_finalist_by_construction": True,
        "econ_finalist": econ_finalist,
        "econ_finalist_configuration": family_results[econ_finalist].config_id,
        "finalist_rule": finalist["rule"],
        "subblock_bounds": [list(bound) for bound in SUBBLOCK_BOUNDS],
        "block_ranks": finalist["block_ranks"],
        "block_means": finalist["block_means"],
        "median_ranks": finalist["median_ranks"],
        "mean_ranks": finalist["mean_ranks"],
        "full_screen_mean_qlike_econ": finalist["full_screen_mean_qlike"],
        "robustness_full_mean_qlike_leader": finalist["robustness_full_mean_qlike_leader"],
        "headline_replaced_by_mean_loss": econ_finalist
        != finalist["robustness_full_mean_qlike_leader"],
        "mcs_qlike_finalists": {
            "dl": _mcs_status(mcs_qlike, dl_finalist),
            "econ": _mcs_status(mcs_qlike, econ_finalist),
        },
        "mcs_frobenius_finalists": {
            "dl": _mcs_status(mcs_frobenius, dl_finalist),
            "econ": _mcs_status(mcs_frobenius, econ_finalist),
        },
        "spa_qlike_primary": {
            "benchmark": spa_qlike["benchmark"],
            "statistic": spa_qlike["statistic"],
            "p_value_consistent": spa_qlike["p_value_consistent"],
            "exact_benchmark_ties": spa_qlike["exact_benchmark_ties"],
        },
        "ridge01_exact_har_benchmark_tie": ridge_exact_har_tie,
        "source_screen_config_sha256": view.screen_config_sha256,
        "source_validation_selection_sha256": FROZEN_VALIDATION_SELECTION_SHA256,
        "production_panel_sha256": view.panel_sha256,
        "confirm_locked": True,
        "screen_is_not_confirmation": True,
    }

    manifest = {
        "production_panel_path": str(panel_path),
        "production_panel_sha256": view.panel_sha256,
        "frozen_core_config_path": str(core_path),
        "frozen_core_config_sha256": view.core_config_sha256,
        "screen_config_path": str(screen_config_path),
        "screen_config_sha256": view.screen_config_sha256,
        "validation_selection_sha256": FROZEN_VALIDATION_SELECTION_SHA256,
        "execution_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "assets": list(view.assets),
        "screen_dates": target_dates,
        "target_count": len(schedule.steps),
        "refit_positions": refit_positions,
        "model_families": labels,
        "seed_list": list(SEEDS),
        "max_model_observation_date": view.max_model_observation_date.isoformat(),
        "max_scoring_target_date": view.max_scoring_target_date.isoformat(),
        "runtime_seconds_total": time.perf_counter() - started,
        "runtime_by_family": {
            family: float(family_results[family].runtime_seconds) for family in labels
        },
        "confirm_locked": True,
        "screen_is_selection_data_not_confirmation": True,
        "failed_families": failed_families,
        "spa_mcs_universe": complete_labels,
        "complete_model_count": len(complete_labels),
    }

    manifest_path = output_dir / "binance_screen_manifest.json"
    summary_out = output_dir / "binance_screen_model_summary.csv"
    spa_path = output_dir / "binance_screen_spa.json"
    mcs_q_path = output_dir / "binance_screen_mcs_qlike.json"
    mcs_f_path = output_dir / "binance_screen_mcs_frobenius.json"
    ranks_path = output_dir / "binance_screen_subblock_ranks.csv"
    finalists_path = output_dir / "binance_screen_finalists.json"
    failures_path = output_dir / "binance_screen_failures.json"
    qlike_path = output_dir / "binance_screen_losses_qlike.npz"
    frob_path = output_dir / "binance_screen_losses_frobenius.npz"
    forecasts_path = output_dir / "binance_screen_forecasts.npz"
    diag_path = output_dir / "binance_screen_diagnostics.json"

    rank_rows = []
    for family in ECON_ELIGIBLE:
        if family not in finalist["median_ranks"]:
            continue
        rank_rows.append(
            {
                "family": family,
                "block1_rank": finalist["block_ranks"]["block_1"][family],
                "block2_rank": finalist["block_ranks"]["block_2"][family],
                "block3_rank": finalist["block_ranks"]["block_3"][family],
                "block4_rank": finalist["block_ranks"]["block_4"][family],
                "median_rank": finalist["median_ranks"][family],
                "mean_rank": finalist["mean_ranks"][family],
                "full_screen_mean_qlike": finalist["full_screen_mean_qlike"][family],
            }
        )

    spa_blob = {"qlike": spa_qlike, "frobenius": spa_frobenius}
    atomic_write_text(spa_path, json.dumps(spa_blob, indent=2, sort_keys=True) + "\n")
    atomic_write_text(mcs_q_path, json.dumps(mcs_qlike, indent=2, sort_keys=True) + "\n")
    atomic_write_text(mcs_f_path, json.dumps(mcs_frobenius, indent=2, sort_keys=True) + "\n")
    atomic_write_text(finalists_path, json.dumps(finalists, indent=2, sort_keys=True) + "\n")
    atomic_write_text(failures_path, json.dumps(failures, indent=2, sort_keys=True) + "\n")
    atomic_write_text(
        diag_path,
        json.dumps({"lstm19": lstm_diag, "models": model_table}, indent=2, sort_keys=True) + "\n",
    )
    pd.DataFrame(model_table).to_csv(summary_out, index=False)
    pd.DataFrame(rank_rows).to_csv(ranks_path, index=False)
    n_families = len(labels)
    qlike_full = np.full((REQUIRED_SCREEN_TARGETS, n_families), np.nan)
    frob_full = np.full((REQUIRED_SCREEN_TARGETS, n_families), np.nan)
    for column, name in enumerate(labels):
        item = family_results[name]
        if item.qlike is not None:
            qlike_full[:, column] = item.qlike
        if item.frobenius is not None:
            frob_full[:, column] = item.frobenius
    atomic_save_npz(
        qlike_path,
        losses=qlike_full,
        labels=np.array(labels, dtype="U16"),
        complete_labels=np.array(complete_labels, dtype="U16"),
        dates=np.array(target_dates, dtype="U10"),
    )
    atomic_save_npz(
        frob_path,
        losses=frob_full,
        labels=np.array(labels, dtype="U16"),
        complete_labels=np.array(complete_labels, dtype="U16"),
        dates=np.array(target_dates, dtype="U10"),
    )
    forecast_arrays = {
        "origin_dates": np.array(
            [step.origin.date().isoformat() for step in schedule.steps], dtype="U10"
        ),
        "target_dates": np.array(target_dates, dtype="U10"),
        "proxy_rcov": proxies,
    }
    for family, item in family_results.items():
        if item.forecasts is None:
            continue
        forecast_arrays[f"{family}_forecasts"] = item.forecasts
        forecast_arrays[f"{family}_qlike"] = item.qlike
        forecast_arrays[f"{family}_frobenius"] = item.frobenius
    for item in seed_rows:
        if item.forecasts is None:
            continue
        key = f"LSTM-BEKK_LSTM19_seed{item.seed}"
        forecast_arrays[f"{key}_forecasts"] = item.forecasts
    atomic_save_npz(forecasts_path, **forecast_arrays)

    artifact_hashes = {
        "summary_sha256": sha256_file(summary_out),
        "spa_sha256": sha256_file(spa_path),
        "mcs_qlike_sha256": sha256_file(mcs_q_path),
        "mcs_frobenius_sha256": sha256_file(mcs_f_path),
        "subblock_ranks_sha256": sha256_file(ranks_path),
        "finalists_sha256": sha256_file(finalists_path),
        "failures_sha256": sha256_file(failures_path),
        "losses_qlike_sha256": sha256_file(qlike_path),
        "losses_frobenius_sha256": sha256_file(frob_path),
        "forecasts_sha256": sha256_file(forecasts_path),
        "diagnostics_sha256": sha256_file(diag_path),
    }
    manifest["artifact_hashes"] = artifact_hashes
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifact_hashes["manifest_sha256"] = sha256_file(manifest_path)
    return {
        "manifest": manifest,
        "finalists": finalists,
        "table": model_table,
        "spa": spa_blob,
        "mcs_qlike": mcs_qlike,
        "mcs_frobenius": mcs_frobenius,
        "failures": failures,
        "artifact_hashes": artifact_hashes,
        "lstm": lstm_diag,
        "paths": {
            "manifest": str(manifest_path),
            "summary": str(summary_out),
            "spa": str(spa_path),
            "mcs_qlike": str(mcs_q_path),
            "mcs_frobenius": str(mcs_f_path),
            "subblock_ranks": str(ranks_path),
            "finalists": str(finalists_path),
            "failures": str(failures_path),
            "losses_qlike": str(qlike_path),
            "losses_frobenius": str(frob_path),
            "forecasts": str(forecasts_path),
        },
    }
