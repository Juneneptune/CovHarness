"""First-stage Binance VALIDATION forecasting and within-family selection.

Uses only the VALIDATION block. SCREEN and CONFIRM are not requested.
Model-observable dates stop at 2023-03-22. Scoring targets stop at 2023-03-23.
This module does not change model mathematics or frozen grids.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from covharness.data.binance_calendar import BINANCE_ASSETS, HALT_DATE
from covharness.losses.contracts import (
    ForecastNotPositiveDefiniteError,
    InvalidCovarianceMatrixError,
)
from covharness.losses.frobenius import squared_frobenius_loss
from covharness.losses.qlike import reduced_qlike_loss
from covharness.protocol.binance import (
    CORE_ROSTER,
    PRODUCTION_PANEL_SHA256,
    BinanceBlock,
    BinanceSegmentedProtocol,
    CandidateValidationRecord,
    build_binance_segmented_protocol,
    load_core_config,
    select_validation_configuration,
)
from covharness.protocol.exceptions import ConfirmLockedError
from covharness.protocol.runner import (
    RollingForecastRecord,
    _advance_at_origin,
    _fit_at_origin,
    build_origin_payload,
)

FROZEN_PANEL_SHA256 = PRODUCTION_PANEL_SHA256
FROZEN_CONFIG_SHA256 = (
    "8d63eacacf13a876651f9c4eb5399a8527d8458278974ff069dd0701dd5cbf07"
)
MAX_MODEL_OBSERVATION_DATE = dt.date(2023, 3, 22)
MAX_SCORING_TARGET_DATE = dt.date(2023, 3, 23)
REQUIRED_VALIDATION_TARGETS = 250
FROZEN_ASSETS = BINANCE_ASSETS
FIXED_FAMILIES = (
    "RW",
    "HAR-DRD",
    "HARQ-DRD",
    "LW-linear",
    "LW-NL",
    "DCC",
    "DCC-NL",
)
SEARCHED_FAMILIES = ("EWMA", "Ridge-DRD", "XGBoost-DRD", "LSTM-BEKK")
EXECUTION_ORDER = (
    "RW",
    "HAR-DRD",
    "HARQ-DRD",
    "LW-linear",
    "LW-NL",
    "DCC",
    "DCC-NL",
    "EWMA",
    "Ridge-DRD",
    "XGBoost-DRD",
    "LSTM-BEKK",
)


class FrozenHashMismatchError(RuntimeError):
    """A production panel or configuration hash drifted from the freeze."""


class ValidationLeakError(RuntimeError):
    """A date after the VALIDATION bound was offered to a model or selector."""


class IncompleteCheckpointError(ValueError):
    """A checkpoint is present but is not a complete verified candidate."""


@dataclass(frozen=True)
class ValidationView:
    """Development arrays through 2023-03-23. Later NPZ rows are omitted."""

    dates: tuple[dt.date, ...]
    calendar: pd.DatetimeIndex
    assets: tuple[str, ...]
    daily_returns: NDArray[np.floating]
    rcov: NDArray[np.floating]
    rq: NDArray[np.floating]
    panel_sha256: str
    config_sha256: str
    max_model_observation_date: dt.date
    max_scoring_target_date: dt.date


@dataclass(frozen=True)
class CandidateRunResult:
    """One configuration or LSTM seed after VALIDATION scoring."""

    family: str
    config_id: str
    seed: int | None
    complete_support: bool
    n_targets: int
    mean_qlike: float | None
    mean_frobenius: float | None
    repair_count: int
    qlike_failures: int
    frobenius_failures: int
    nonfinite_forecasts: int
    fit_failures: int
    failure_reason: str | None
    runtime_seconds: float
    origins: tuple[dt.date, ...]
    targets: tuple[dt.date, ...]
    forecasts: NDArray[np.floating] | None
    qlike: NDArray[np.floating] | None
    frobenius: NDArray[np.floating] | None
    repaired: NDArray[np.bool_] | None
    refit: NDArray[np.bool_] | None
    actions: tuple[str, ...] | None = None
    min_eigenvalues: NDArray[np.floating] | None = None
    positive_definite: NDArray[np.bool_] | None = None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_frozen_hashes(panel_path: Path, config_path: Path) -> tuple[str, str]:
    """Stop before fitting if either frozen artifact drifted."""
    panel_hash = sha256_file(panel_path)
    config_hash = sha256_file(config_path)
    if panel_hash != FROZEN_PANEL_SHA256:
        raise FrozenHashMismatchError(
            f"panel SHA-256 {panel_hash} != frozen {FROZEN_PANEL_SHA256}"
        )
    if config_hash != FROZEN_CONFIG_SHA256:
        raise FrozenHashMismatchError(
            f"config SHA-256 {config_hash} != frozen {FROZEN_CONFIG_SHA256}"
        )
    return panel_hash, config_hash


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_save_npz(path: Path, **arrays: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def request_validation_schedule(
    protocol: BinanceSegmentedProtocol | None = None,
) -> object:
    """Return the VALIDATION schedule only. CONFIRM stays locked."""
    protocol = protocol or build_binance_segmented_protocol()
    try:
        protocol.confirm_targets()
        raise ValidationLeakError("CONFIRM was accessible without unlock")
    except ConfirmLockedError:
        pass
    schedule = protocol.forecast_schedule(BinanceBlock.VALIDATION)
    if schedule.block != "VALIDATION":
        raise ValidationLeakError("schedule block is not VALIDATION")
    if len(schedule.steps) != REQUIRED_VALIDATION_TARGETS:
        raise ValidationLeakError(
            f"VALIDATION must have {REQUIRED_VALIDATION_TARGETS} targets; "
            f"got {len(schedule.steps)}"
        )
    first = schedule.steps[0]
    last = schedule.steps[-1]
    if first.origin.date() != dt.date(2022, 7, 16):
        raise ValidationLeakError("first VALIDATION origin is not 2022-07-16")
    if last.target.date() != MAX_SCORING_TARGET_DATE:
        raise ValidationLeakError("last VALIDATION target is not 2023-03-23")
    halt = pd.Timestamp(HALT_DATE)
    for step in schedule.steps:
        if step.origin == halt or step.target == halt:
            raise ValidationLeakError("halt date entered the VALIDATION schedule")
        if step.origin.date() > MAX_MODEL_OBSERVATION_DATE:
            raise ValidationLeakError("origin after 2023-03-22")
        if step.target.date() > MAX_SCORING_TARGET_DATE:
            raise ValidationLeakError("target after 2023-03-23")
    return schedule


def _index_lookup(dates: Sequence[dt.date]) -> dict[dt.date, int]:
    return {day: index for index, day in enumerate(dates)}


def load_validation_view(
    panel_path: Path,
    config_path: Path,
    protocol: BinanceSegmentedProtocol | None = None,
) -> ValidationView:
    """Slice HISTORY_A plus VALIDATION from the production NPZ. No later rows."""
    panel_hash, config_hash = require_frozen_hashes(panel_path, config_path)
    protocol = protocol or build_binance_segmented_protocol()
    schedule = request_validation_schedule(protocol)
    blob = np.load(panel_path)
    assets = tuple(str(name) for name in blob["assets"].tolist())
    if assets != FROZEN_ASSETS:
        raise ValidationLeakError(f"asset order drifted: {assets}")
    npz_dates = tuple(dt.date.fromisoformat(text) for text in blob["dates"].tolist())
    lookup = _index_lookup(npz_dates)
    calendar = protocol.calendar_for_block(BinanceBlock.VALIDATION)
    selected: list[int] = []
    for stamp in calendar:
        day = stamp.date()
        if day > MAX_SCORING_TARGET_DATE or day == HALT_DATE:
            raise ValidationLeakError(f"{day.isoformat()} is outside VALIDATION view")
        selected.append(lookup[day])
    dates = tuple(npz_dates[index] for index in selected)
    if max(dates) != MAX_SCORING_TARGET_DATE:
        raise ValidationLeakError("VALIDATION view does not end on 2023-03-23")
    if any(day > MAX_SCORING_TARGET_DATE for day in dates):
        raise ValidationLeakError("post-VALIDATION dates leaked into the view")
    daily = np.array(blob["daily_returns"][selected], dtype=float, copy=True)
    rcov = np.array(blob["rcov_5min"][selected], dtype=float, copy=True)
    rq = np.array(blob["rq_per_asset"][selected], dtype=float, copy=True)
    origins = [step.origin.date() for step in schedule.steps]
    if max(origins) != MAX_MODEL_OBSERVATION_DATE:
        raise ValidationLeakError("max origin is not 2023-03-22")
    return ValidationView(
        dates=dates,
        calendar=calendar,
        assets=assets,
        daily_returns=daily,
        rcov=rcov,
        rq=rq,
        panel_sha256=panel_hash,
        config_sha256=config_hash,
        max_model_observation_date=max(origins),
        max_scoring_target_date=max(dates),
    )


def assert_no_future_fit_dates(window_dates: Sequence[dt.date], origin: dt.date) -> None:
    """Reject a fit/update window that reads past the origin or past 2023-03-22."""
    if origin > MAX_MODEL_OBSERVATION_DATE:
        raise ValidationLeakError(f"origin {origin.isoformat()} exceeds 2023-03-22")
    if any(day > origin for day in window_dates):
        raise ValidationLeakError("window date after origin")
    if any(day > MAX_MODEL_OBSERVATION_DATE for day in window_dates):
        raise ValidationLeakError("window date after 2023-03-22")
    if HALT_DATE in window_dates:
        raise ValidationLeakError("halt date entered a model window")


def ensemble_covariances(seed_forecasts: Sequence[NDArray[np.floating]]) -> NDArray[np.floating]:
    """Equal-weight mean of seed covariance cubes. Seeds are not dropped."""
    if len(seed_forecasts) != 5:
        raise ValueError("LSTM ensemble requires exactly five seed cubes")
    stacked = np.stack([np.asarray(cube, dtype=float) for cube in seed_forecasts], axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError("LSTM ensemble requires finite seed forecasts")
    return np.mean(stacked, axis=0)


def score_forecast_cube(
    forecasts: NDArray[np.floating],
    proxies: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], int, int]:
    """Score each target. Evaluation does not repair forecasts."""
    n_targets = int(forecasts.shape[0])
    qlike = np.full(n_targets, np.nan, dtype=float)
    frobenius = np.full(n_targets, np.nan, dtype=float)
    qlike_fail = 0
    frobenius_fail = 0
    for time_index in range(n_targets):
        try:
            qlike[time_index] = float(
                reduced_qlike_loss(proxies[time_index], forecasts[time_index])
            )
        except (ForecastNotPositiveDefiniteError, InvalidCovarianceMatrixError):
            qlike_fail += 1
        try:
            frobenius[time_index] = float(
                squared_frobenius_loss(proxies[time_index], forecasts[time_index])
            )
        except InvalidCovarianceMatrixError:
            frobenius_fail += 1
    return qlike, frobenius, qlike_fail, frobenius_fail


def complete_support(qlike: NDArray[np.floating], n_required: int) -> bool:
    """True iff every required target has a finite primary loss."""
    return (
        int(qlike.shape[0]) == n_required
        and bool(np.isfinite(qlike).all())
        and int(np.isfinite(qlike).sum()) == n_required
    )


def candidates_from_config(document: Mapping[str, object], family: str) -> list[dict[str, object]]:
    """Read candidate rows from the frozen YAML mapping. No local grid."""
    block = document["candidates"][family]
    if family in ("XGBoost-DRD", "LSTM-BEKK"):
        rows = block["grid"]
    else:
        rows = block
    if not isinstance(rows, list) or len(rows) < 1:
        raise ValueError(f"{family} has no frozen candidates")
    return [dict(row) for row in rows]


def build_model(family: str, row: Mapping[str, object], seed: int | None = None):
    """Construct one frozen configuration. Grids are not invented here."""
    from covharness.models import (
        DCCCovariance,
        DCCNonlinearCovariance,
        EWMARealizedCovariance,
        HARDRDRealizedCovariance,
        HARQDRDRealizedCovariance,
        LSTMBEKKCovariance,
        LedoitWolfLinearCovariance,
        LedoitWolfNonlinearCovariance,
        RandomWalkRealizedCovariance,
        RidgeDRDRealizedCovariance,
        XGBoostDRDRealizedCovariance,
    )

    if family == "RW":
        return RandomWalkRealizedCovariance()
    if family == "HAR-DRD":
        return HARDRDRealizedCovariance()
    if family == "HARQ-DRD":
        return HARQDRDRealizedCovariance()
    if family == "LW-linear":
        return LedoitWolfLinearCovariance()
    if family == "LW-NL":
        return LedoitWolfNonlinearCovariance()
    if family == "DCC":
        return DCCCovariance()
    if family == "DCC-NL":
        return DCCNonlinearCovariance()
    if family == "EWMA":
        return EWMARealizedCovariance(decay=float(row["decay"]))
    if family == "Ridge-DRD":
        return RidgeDRDRealizedCovariance(lambda_=float(row["lambda"]))
    if family == "XGBoost-DRD":
        return XGBoostDRDRealizedCovariance(
            n_estimators=int(row["n_estimators"]),
            max_depth=int(row["max_depth"]),
            learning_rate=float(row["learning_rate"]),
            min_child_weight=float(row["min_child_weight"]),
            reg_lambda=float(row["reg_lambda"]),
            reg_alpha=float(row["reg_alpha"]),
            gamma=float(row["gamma"]),
        )
    if family == "LSTM-BEKK":
        if seed is None:
            raise ValueError("LSTM-BEKK requires an experiment-layer seed")
        return LSTMBEKKCovariance(
            seed=int(seed),
            num_layers=int(row["num_layers"]),
            dropout=float(row["dropout"]),
            learning_rate=float(row["learning_rate"]),
            gradient_clip_norm=float(row["gradient_clip_norm"]),
            max_epochs=int(row["max_epochs"]),
        )
    raise ValueError(f"unknown family {family}")


def _proxy_cube(view: ValidationView, schedule) -> NDArray[np.floating]:
    lookup = _index_lookup(view.dates)
    return np.stack(
        [view.rcov[lookup[step.target.date()]] for step in schedule.steps],
        axis=0,
    )


def run_scheduled_forecasts(
    *,
    model,
    schedule,
    view: ValidationView,
) -> tuple[list[RollingForecastRecord], NDArray[np.bool_], int, dict[str, object]]:
    """Advance one model on VALIDATION. Fit windows cannot pass 2023-03-22."""
    records: list[RollingForecastRecord] = []
    repaired_flags: list[bool] = []
    actions: list[str] = []
    min_eigs: list[float] = []
    pd_flags: list[bool] = []
    fit_failures = 0
    for step in schedule.steps:
        window_dates = [stamp.date() for stamp in step.estimation_dates(schedule.calendar)]
        assert_no_future_fit_dates(window_dates, step.origin.date())
        payload = build_origin_payload(
            step=step,
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


def evaluate_records(
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
    """Score a completed or failed candidate without shortening support."""
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
            qlike_failures=REQUIRED_VALIDATION_TARGETS,
            frobenius_failures=REQUIRED_VALIDATION_TARGETS,
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
    support = complete_support(qlike, REQUIRED_VALIDATION_TARGETS) and fit_failures == 0
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


def checkpoint_stem(family: str, config_id: str, seed: int | None) -> str:
    if seed is None:
        return f"{family}_{config_id}"
    return f"{family}_{config_id}_seed{seed}"


def write_checkpoint(
    directory: Path,
    result: CandidateRunResult,
    *,
    panel_sha256: str,
    config_sha256: str,
) -> Path:
    """Atomically persist a complete or failed candidate. Partial runs are not complete."""
    directory = Path(directory)
    stem = checkpoint_stem(result.family, result.config_id, result.seed)
    complete = bool(
        result.complete_support
        and result.n_targets == REQUIRED_VALIDATION_TARGETS
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
        if result.min_eigenvalues is not None:
            extras["min_eigenvalues"] = result.min_eigenvalues
        if result.positive_definite is not None:
            extras["positive_definite"] = result.positive_definite
        if result.actions is not None:
            extras["actions"] = np.array(list(result.actions), dtype="U24")
        atomic_save_npz(npz_path, **extras)
        payload["artifact"] = str(npz_path)
        payload["artifact_sha256"] = sha256_file(npz_path)
    atomic_write_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return json_path


def load_complete_checkpoint(
    directory: Path,
    family: str,
    config_id: str,
    seed: int | None,
    *,
    panel_sha256: str,
    config_sha256: str,
    required_targets: Sequence[dt.date],
) -> CandidateRunResult | None:
    """Reuse a complete verified checkpoint. Partial files are ignored, not completed."""
    json_path = Path(directory) / f"{checkpoint_stem(family, config_id, seed)}.json"
    if not json_path.is_file():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("panel_sha256") != panel_sha256 or payload.get("config_sha256") != config_sha256:
        return None
    if payload.get("complete") is not True:
        return None
    if int(payload.get("n_targets", 0)) != REQUIRED_VALIDATION_TARGETS:
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
        qlike, REQUIRED_VALIDATION_TARGETS
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
        actions=tuple(str(item) for item in blob["actions"].tolist())
        if "actions" in blob.files
        else None,
        min_eigenvalues=np.array(blob["min_eigenvalues"], dtype=float)
        if "min_eigenvalues" in blob.files
        else None,
        positive_definite=np.array(blob["positive_definite"], dtype=bool)
        if "positive_definite" in blob.files
        else None,
    )


def run_one_configuration(
    *,
    family: str,
    row: Mapping[str, object],
    seed: int | None,
    view: ValidationView,
    schedule,
    checkpoint_dir: Path,
    model_factory: Callable[..., object] | None = None,
) -> CandidateRunResult:
    """Fit one configuration or LSTM seed, or reuse a verified checkpoint."""
    config_id = str(row["id"])
    required_targets = [step.target.date() for step in schedule.steps]
    reused = load_complete_checkpoint(
        checkpoint_dir,
        family,
        config_id,
        seed,
        panel_sha256=view.panel_sha256,
        config_sha256=view.config_sha256,
        required_targets=required_targets,
    )
    if reused is not None:
        return reused
    factory = model_factory or build_model
    started = time.perf_counter()
    try:
        model = factory(family, row, seed)
        records, repaired, fit_failures, extras = run_scheduled_forecasts(
            model=model, schedule=schedule, view=view
        )
        result = evaluate_records(
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
            qlike_failures=REQUIRED_VALIDATION_TARGETS,
            frobenius_failures=REQUIRED_VALIDATION_TARGETS,
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
    write_checkpoint(
        checkpoint_dir,
        result,
        panel_sha256=view.panel_sha256,
        config_sha256=view.config_sha256,
    )
    return result


def lstm_ensemble_result(seed_results: Sequence[CandidateRunResult], proxies: NDArray[np.floating]) -> CandidateRunResult:
    """Average seed covariances, then score. A failed seed invalidates the candidate."""
    if len(seed_results) != 5:
        raise ValueError("LSTM ensemble requires five seed results")
    config_id = seed_results[0].config_id
    if any(not item.complete_support or item.forecasts is None for item in seed_results):
        reason = "failed seed invalidates configuration"
        return CandidateRunResult(
            family="LSTM-BEKK",
            config_id=config_id,
            seed=None,
            complete_support=False,
            n_targets=max(item.n_targets for item in seed_results),
            mean_qlike=None,
            mean_frobenius=None,
            repair_count=sum(item.repair_count for item in seed_results),
            qlike_failures=REQUIRED_VALIDATION_TARGETS,
            frobenius_failures=0,
            nonfinite_forecasts=0,
            fit_failures=sum(item.fit_failures for item in seed_results),
            failure_reason=reason,
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
    support = complete_support(qlike, REQUIRED_VALIDATION_TARGETS)
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


def select_family_winner(
    family: str,
    results: Sequence[CandidateRunResult],
) -> dict[str, object]:
    """Apply the frozen helper. Frobenius does not select."""
    if family in FIXED_FAMILIES:
        row = results[0]
        return {
            "family": family,
            "status": "fixed" if row.complete_support else "failed",
            "selected_id": row.config_id if row.complete_support else None,
            "tuned": False,
            "n_valid": int(row.complete_support),
            "n_candidates": 1,
            "mean_qlike": row.mean_qlike,
            "mean_frobenius": row.mean_frobenius,
            "repair_count": row.repair_count,
            "failure_reason": row.failure_reason,
        }
    records = [
        CandidateValidationRecord(item.config_id, item.complete_support, item.mean_qlike)
        for item in results
    ]
    n_valid = sum(1 for item in results if item.complete_support)
    try:
        winner_id = select_validation_configuration(records)
    except Exception as exc:  # noqa: BLE001
        return {
            "family": family,
            "status": "failed",
            "selected_id": None,
            "tuned": True,
            "n_valid": n_valid,
            "n_candidates": len(results),
            "mean_qlike": None,
            "mean_frobenius": None,
            "repair_count": 0,
            "failure_reason": str(exc),
        }
    winner = next(item for item in results if item.config_id == winner_id)
    return {
        "family": family,
        "status": "selected",
        "selected_id": winner_id,
        "tuned": True,
        "n_valid": n_valid,
        "n_candidates": len(results),
        "mean_qlike": winner.mean_qlike,
        "mean_frobenius": winner.mean_frobenius,
        "repair_count": winner.repair_count,
        "failure_reason": None,
    }


def _result_row(result: CandidateRunResult, seed_scope: str) -> dict[str, object]:
    return {
        "family": result.family,
        "configuration_id": result.config_id,
        "seed_scope": seed_scope,
        "complete_support": result.complete_support,
        "n_targets": result.n_targets,
        "mean_qlike": result.mean_qlike,
        "mean_frobenius": result.mean_frobenius,
        "repair_count": result.repair_count,
        "qlike_failures": result.qlike_failures,
        "frobenius_failures": result.frobenius_failures,
        "nonfinite_forecasts": result.nonfinite_forecasts,
        "fit_failures": result.fit_failures,
        "runtime_seconds": result.runtime_seconds,
        "failure_reason": result.failure_reason,
    }


def run_binance_validation(
    *,
    panel_path: Path,
    config_path: Path,
    output_dir: Path,
    model_factory: Callable[..., object] | None = None,
    families: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run VALIDATION selection. SCREEN and CONFIRM are not touched."""
    started = time.perf_counter()
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "binance_validation_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda message: print(message, flush=True))
    protocol = build_binance_segmented_protocol()
    view = load_validation_view(panel_path, config_path, protocol)
    schedule = request_validation_schedule(protocol)
    document = load_core_config(config_path)
    seeds = tuple(int(seed) for seed in document["seeds"])
    if tuple(seeds) != (0, 1, 2, 3, 4):
        raise FrozenHashMismatchError("frozen seed list drifted")
    proxies = _proxy_cube(view, schedule)
    roster = tuple(families) if families is not None else EXECUTION_ORDER
    family_results: dict[str, list[CandidateRunResult]] = {}
    seed_results: dict[str, list[CandidateRunResult]] = {}
    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for family in roster:
        log(f"starting family {family}")
        rows = candidates_from_config(document, family)
        collected: list[CandidateRunResult] = []
        if family == "LSTM-BEKK":
            for row in rows:
                per_seed: list[CandidateRunResult] = []
                for seed in seeds:
                    log(f"  {row['id']} seed {seed}")
                    result = run_one_configuration(
                        family=family,
                        row=row,
                        seed=seed,
                        view=view,
                        schedule=schedule,
                        checkpoint_dir=checkpoint_dir,
                        model_factory=model_factory,
                    )
                    per_seed.append(result)
                    seed_results.setdefault(str(row["id"]), []).append(result)
                    summary_rows.append(_result_row(result, f"seed{seed}"))
                    if not result.complete_support:
                        failures.append(_result_row(result, f"seed{seed}"))
                ensemble = lstm_ensemble_result(per_seed, proxies)
                collected.append(ensemble)
                summary_rows.append(_result_row(ensemble, "ensemble"))
                if not ensemble.complete_support:
                    failures.append(_result_row(ensemble, "ensemble"))
        else:
            for row in rows:
                log(f"  {row['id']}")
                result = run_one_configuration(
                    family=family,
                    row=row,
                    seed=None,
                    view=view,
                    schedule=schedule,
                    checkpoint_dir=checkpoint_dir,
                    model_factory=model_factory,
                )
                collected.append(result)
                summary_rows.append(_result_row(result, "deterministic"))
                if not result.complete_support:
                    failures.append(_result_row(result, "deterministic"))
        family_results[family] = collected

    selection = {
        "source_panel_sha256": view.panel_sha256,
        "source_config_sha256": view.config_sha256,
        "selection_rule": "mean_reduced_qlike_complete_250_validation_targets",
        "tie_rule": "lexicographically_smaller_config_id",
        "secondary_loss": "squared_frobenius_descriptive_only",
        "any_candidate_failed": len(failures) > 0,
        "families": {
            family: select_family_winner(family, family_results[family])
            for family in roster
        },
    }
    refit_positions = [index for index, step in enumerate(schedule.steps) if step.refit]
    manifest = {
        "production_panel_path": str(panel_path),
        "production_panel_sha256": view.panel_sha256,
        "frozen_config_path": str(config_path),
        "frozen_config_sha256": view.config_sha256,
        "execution_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "assets": list(view.assets),
        "validation_dates": [step.target.date().isoformat() for step in schedule.steps],
        "target_count": len(schedule.steps),
        "refit_positions": refit_positions,
        "model_families": list(roster),
        "seed_list": list(seeds),
        "max_model_observation_date": view.max_model_observation_date.isoformat(),
        "max_scoring_target_date": view.max_scoring_target_date.isoformat(),
        "runtime_seconds_total": time.perf_counter() - started,
        "runtime_by_family": {
            family: float(sum(item.runtime_seconds for item in family_results[family]))
            for family in roster
        },
    }
    table_rows = []
    for family in roster:
        info = selection["families"][family]
        table_rows.append(
            {
                "family": family,
                "configuration": info["selected_id"],
                "mean_reduced_qlike": info["mean_qlike"],
                "mean_squared_frobenius": info["mean_frobenius"],
                "target_count": REQUIRED_VALIDATION_TARGETS if info["selected_id"] else 0,
                "repair_fallback_count": info["repair_count"],
                "status": info["status"],
            }
        )

    forecast_arrays: dict[str, object] = {
        "origin_dates": np.array(
            [step.origin.date().isoformat() for step in schedule.steps], dtype="U10"
        ),
        "target_dates": np.array(
            [step.target.date().isoformat() for step in schedule.steps], dtype="U10"
        ),
        "refit": np.array([bool(step.refit) for step in schedule.steps], dtype=bool),
        "proxy_rcov": proxies,
    }
    for family, results in family_results.items():
        for item in results:
            if item.forecasts is None:
                continue
            key = f"{family}_{item.config_id}"
            forecast_arrays[f"{key}_forecasts"] = item.forecasts
            forecast_arrays[f"{key}_qlike"] = item.qlike
            forecast_arrays[f"{key}_frobenius"] = item.frobenius
            if item.repaired is not None:
                forecast_arrays[f"{key}_repaired"] = item.repaired
    for config_id, seeds_for_id in seed_results.items():
        for item in seeds_for_id:
            if item.forecasts is None:
                continue
            key = f"LSTM-BEKK_{config_id}_seed{item.seed}"
            forecast_arrays[f"{key}_forecasts"] = item.forecasts
            forecast_arrays[f"{key}_qlike"] = item.qlike
            forecast_arrays[f"{key}_frobenius"] = item.frobenius

    diagnostics = {
        "max_model_observation_date": view.max_model_observation_date.isoformat(),
        "max_scoring_target_date": view.max_scoring_target_date.isoformat(),
        "candidates": summary_rows,
        "lstm_seed_completion": {
            config_id: {
                "seeds_complete": [bool(item.complete_support) for item in rows],
                "per_seed_runtime_seconds": [item.runtime_seconds for item in rows],
                "per_seed_mean_qlike": [item.mean_qlike for item in rows],
                "per_seed_mean_frobenius": [item.mean_frobenius for item in rows],
            }
            for config_id, rows in seed_results.items()
        },
        "lstm_ensemble": {
            item.config_id: {
                "mean_qlike": item.mean_qlike,
                "mean_frobenius": item.mean_frobenius,
                "complete_support": item.complete_support,
                "runtime_seconds": item.runtime_seconds,
            }
            for item in family_results.get("LSTM-BEKK", [])
        },
    }

    manifest_path = output_dir / "binance_validation_manifest.json"
    selection_path = output_dir / "binance_validation_selection.json"
    failures_path = output_dir / "binance_validation_failures.json"
    summary_path = output_dir / "binance_validation_candidate_summary.csv"
    table_path = output_dir / "binance_validation_development_table.csv"
    diagnostics_path = output_dir / "binance_validation_diagnostics.json"
    forecasts_path = output_dir / "binance_validation_forecasts.npz"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    atomic_write_text(selection_path, json.dumps(selection, indent=2, sort_keys=True) + "\n")
    atomic_write_text(failures_path, json.dumps(failures, indent=2, sort_keys=True) + "\n")
    atomic_write_text(diagnostics_path, json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(table_rows).to_csv(table_path, index=False)
    if forecast_arrays:
        atomic_save_npz(forecasts_path, **forecast_arrays)
    artifact_hashes = {
        "selection_sha256": sha256_file(selection_path),
        "failures_sha256": sha256_file(failures_path),
        "summary_sha256": sha256_file(summary_path),
        "development_table_sha256": sha256_file(table_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
    }
    if forecasts_path.is_file():
        artifact_hashes["forecasts_sha256"] = sha256_file(forecasts_path)
        manifest["forecast_artifact"] = str(forecasts_path)
        manifest["forecast_artifact_sha256"] = artifact_hashes["forecasts_sha256"]
    manifest["artifact_hashes"] = artifact_hashes
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifact_hashes["manifest_sha256"] = sha256_file(manifest_path)
    return {
        "manifest": manifest,
        "selection": selection,
        "table": table_rows,
        "failures": failures,
        "artifact_hashes": artifact_hashes,
        "paths": {
            "manifest": str(manifest_path),
            "selection": str(selection_path),
            "failures": str(failures_path),
            "summary": str(summary_path),
            "development_table": str(table_path),
            "diagnostics": str(diagnostics_path),
            "forecasts": str(forecasts_path) if forecasts_path.is_file() else None,
        },
    }
