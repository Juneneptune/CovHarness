"""Branch-specific Binance segmented schedule and core configuration.

This module does not modify ``TemporalProtocol``. Development and confirmatory
calendars are never packed across 2023-03-24. Schedule construction uses
dates only. It does not read realized covariance, quarticity, or returns.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml

from covharness.data.binance_calendar import (
    BINANCE_ASSETS,
    EXPECTED_SEGMENT_COUNTS,
    HALT_DATE,
    SAMPLE_END,
    SEGMENT_BOUNDS,
    segment_dates,
    verify_frozen_segment_counts,
)
from covharness.protocol.constants import (
    DEFAULT_REFIT_CADENCE,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_SEEDS,
    LOCKED,
    SEED_AGGREGATION,
)
from covharness.protocol.exceptions import ConfirmLockedError, LookaheadError
from covharness.protocol.rolling import ForecastStep, build_schedule

BRANCH_IDENTITY = "binance_open_data_core"
BRANCH_VERSION = "2026-09-21"
PRODUCTION_PANEL_SHA256 = (
    "2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a"
)
CORE_CONFIG_NAME = "binance_open_data_core.yaml"
PRIMARY_VALIDATION_LOSS = "reduced_qlike"
SECONDARY_VALIDATION_LOSS = "squared_frobenius"
TIE_RULE = "lexicographically_smaller_config_id"
FAILURE_RULE = "complete_validation_support_required"

EVALUATION_BLOCKS = ("VALIDATION", "SCREEN", "CONFIRM")
CORE_ROSTER = (
    "RW",
    "EWMA",
    "HAR-DRD",
    "HARQ-DRD",
    "LW-linear",
    "LW-NL",
    "DCC",
    "DCC-NL",
    "Ridge-DRD",
    "XGBoost-DRD",
    "LSTM-BEKK",
)

EWMA_DECAYS: tuple[tuple[str, float], ...] = (
    ("EWMA01", 0.8000),
    ("EWMA02", 0.8500),
    ("EWMA03", 0.8800),
    ("EWMA04", 0.9000),
    ("EWMA05", 0.9200),
    ("EWMA06", 0.9300),
    ("EWMA07", 0.9400),
    ("EWMA08", 0.9500),
    ("EWMA09", 0.9600),
    ("EWMA10", 0.9650),
    ("EWMA11", 0.9700),
    ("EWMA12", 0.9750),
    ("EWMA13", 0.9800),
    ("EWMA14", 0.9825),
    ("EWMA15", 0.9850),
    ("EWMA16", 0.9875),
    ("EWMA17", 0.9900),
    ("EWMA18", 0.9925),
    ("EWMA19", 0.9950),
    ("EWMA20", 0.9975),
)

RIDGE_LAMBDAS: tuple[tuple[str, float], ...] = (
    ("RIDGE01", 0.0),
    ("RIDGE02", 0.0001),
    ("RIDGE03", 0.0003),
    ("RIDGE04", 0.001),
    ("RIDGE05", 0.003),
    ("RIDGE06", 0.01),
    ("RIDGE07", 0.03),
    ("RIDGE08", 0.1),
    ("RIDGE09", 0.3),
    ("RIDGE10", 1.0),
    ("RIDGE11", 3.0),
    ("RIDGE12", 10.0),
    ("RIDGE13", 30.0),
    ("RIDGE14", 100.0),
    ("RIDGE15", 300.0),
    ("RIDGE16", 1000.0),
    ("RIDGE17", 3000.0),
    ("RIDGE18", 10000.0),
    ("RIDGE19", 30000.0),
    ("RIDGE20", 100000.0),
)

XGB_CAPACITY: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("C1", {"n_estimators": 50, "max_depth": 1, "learning_rate": 0.10}),
    ("C2", {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05}),
    ("C3", {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.025}),
    ("C4", {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.02}),
)

XGB_REGULARIZATION: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("R1", {"min_child_weight": 1, "reg_lambda": 1, "reg_alpha": 0, "gamma": 0}),
    ("R2", {"min_child_weight": 5, "reg_lambda": 1, "reg_alpha": 0, "gamma": 0}),
    ("R3", {"min_child_weight": 1, "reg_lambda": 10, "reg_alpha": 0, "gamma": 0}),
    ("R4", {"min_child_weight": 1, "reg_lambda": 1, "reg_alpha": 0.1, "gamma": 0}),
    ("R5", {"min_child_weight": 1, "reg_lambda": 1, "reg_alpha": 0, "gamma": 0.01}),
)

LSTM_ARCHITECTURE: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("A1", {"num_layers": 3, "dropout": 0.0, "max_epochs": 25}),
    ("A2", {"num_layers": 3, "dropout": 0.1, "max_epochs": 50}),
    ("A3", {"num_layers": 4, "dropout": 0.1, "max_epochs": 50}),
    ("A4", {"num_layers": 4, "dropout": 0.2, "max_epochs": 75}),
    ("A5", {"num_layers": 5, "dropout": 0.2, "max_epochs": 75}),
)

LSTM_OPTIMIZER: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("O1", {"learning_rate": 0.0001, "gradient_clip_norm": 1}),
    ("O2", {"learning_rate": 0.0003, "gradient_clip_norm": 1}),
    ("O3", {"learning_rate": 0.001, "gradient_clip_norm": 1}),
    ("O4", {"learning_rate": 0.0003, "gradient_clip_norm": 5}),
)


class BinanceBlock(str, Enum):
    """Binance evaluation blocks. HISTORY_A and HISTORY_B are burn-in only."""

    VALIDATION = "VALIDATION"
    SCREEN = "SCREEN"
    CONFIRM = "CONFIRM"


class InvalidBinanceProtocolError(ValueError):
    """A frozen Binance schedule or configuration identity failed a check."""


class BinanceSelectionError(ValueError):
    """No admissible VALIDATION candidate remained after the frozen filters."""


@dataclass(frozen=True)
class BinanceBlockSchedule:
    """One evaluation-block schedule on a single uncrossed segment calendar."""

    block: str
    calendar: pd.DatetimeIndex
    steps: tuple[ForecastStep, ...]

    def estimation_dates(self, step: ForecastStep) -> pd.DatetimeIndex:
        return step.estimation_dates(self.calendar)


@dataclass(frozen=True)
class CandidateValidationRecord:
    """Already-computed VALIDATION support and primary score. No market data."""

    config_id: str
    complete_support: bool
    primary_score: float | None


def _utc_index(dates: Sequence[dt.date]) -> pd.DatetimeIndex:
    """Build a tz-naive midnight index that represents UTC statistical dates."""
    return pd.DatetimeIndex([pd.Timestamp(day) for day in dates])


def _core_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / CORE_CONFIG_NAME


def _xgboost_candidates() -> list[dict[str, object]]:
    """Cartesian product of capacity then regularization in frozen ID order."""
    candidates: list[dict[str, object]] = []
    index = 1
    for capacity_id, capacity in XGB_CAPACITY:
        for reg_id, regularization in XGB_REGULARIZATION:
            payload: dict[str, object] = {
                "id": f"XGB{index:02d}",
                "capacity": capacity_id,
                "regularization": reg_id,
            }
            payload.update(capacity)
            payload.update(regularization)
            candidates.append(payload)
            index += 1
    return candidates


def _lstm_candidates() -> list[dict[str, object]]:
    """Cartesian product of architecture then optimizer. Seed is not a field."""
    candidates: list[dict[str, object]] = []
    index = 1
    for architecture_id, architecture in LSTM_ARCHITECTURE:
        for optimizer_id, optimizer in LSTM_OPTIMIZER:
            payload: dict[str, object] = {
                "id": f"LSTM{index:02d}",
                "architecture": architecture_id,
                "optimizer": optimizer_id,
            }
            payload.update(architecture)
            payload.update(optimizer)
            candidates.append(payload)
            index += 1
    return candidates


def frozen_core_config() -> dict[str, object]:
    """Return the auditable first-stage Binance core configuration."""
    verify_frozen_segment_counts()
    segments = {
        name: {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "n_dates": EXPECTED_SEGMENT_COUNTS[name],
        }
        for name, (start, end) in SEGMENT_BOUNDS.items()
    }
    return {
        "branch": BRANCH_IDENTITY,
        "version": BRANCH_VERSION,
        "production_panel_sha256": PRODUCTION_PANEL_SHA256,
        "assets": list(BINANCE_ASSETS),
        "halt_date": HALT_DATE.isoformat(),
        "sample_end": SAMPLE_END.isoformat(),
        "segments": segments,
        "m": DEFAULT_ROLLING_WINDOW,
        "refit_cadence": DEFAULT_REFIT_CADENCE,
        "first_origin_is_refit": True,
        "block_local_refit_reset": True,
        "pack_across_halt": False,
        "roster": list(CORE_ROSTER),
        "primary_validation_loss": PRIMARY_VALIDATION_LOSS,
        "secondary_validation_loss": SECONDARY_VALIDATION_LOSS,
        "secondary_loss_selects": False,
        "seeds": list(DEFAULT_SEEDS),
        "seed_aggregation": SEED_AGGREGATION,
        "allow_best_seed": False,
        "tie_rule": TIE_RULE,
        "candidate_failure_rule": FAILURE_RULE,
        "confirm_locked": True,
        "candidates": {
            "RW": [{"id": "RW01"}],
            "HAR-DRD": [{"id": "HARDRD01"}],
            "HARQ-DRD": [{"id": "HARQDRD01"}],
            "LW-linear": [{"id": "LWLIN01"}],
            "LW-NL": [{"id": "LWNL01"}],
            "DCC": [{"id": "DCC01"}],
            "DCC-NL": [{"id": "DCCNL01"}],
            "EWMA": [{"id": name, "decay": decay} for name, decay in EWMA_DECAYS],
            "Ridge-DRD": [
                {"id": name, "lambda": value} for name, value in RIDGE_LAMBDAS
            ],
            "XGBoost-DRD": {
                "fixed": {
                    "objective": "reg:squarederror",
                    "booster": "gbtree",
                    "tree_method": "hist",
                    "device": "cpu",
                    "n_jobs": 1,
                    "subsample": 1,
                    "colsample_bytree": 1,
                    "colsample_bylevel": 1,
                    "colsample_bynode": 1,
                    "grow_policy": "depthwise",
                    "max_bin": 256,
                    "base_score": 0,
                    "random_state": 0,
                    "early_stopping": False,
                },
                "grid": _xgboost_candidates(),
            },
            "LSTM-BEKK": {
                "fixed": {
                    "hidden_size": "N",
                    "dtype": "float64",
                    "device": "cpu",
                    "full_bptt": True,
                    "early_stopping": False,
                    "inner_validation_split": False,
                    "learning_rate_scheduler": False,
                    "optimizer": "RMSprop",
                    "rmsprop_alpha": 0.99,
                    "rmsprop_eps": 1.0e-8,
                    "rmsprop_momentum": 0.0,
                    "rmsprop_centered": False,
                    "weight_decay": 0.0,
                    "seeds_are_not_configurations": True,
                },
                "grid": _lstm_candidates(),
            },
        },
    }


def serialize_core_config(document: Mapping[str, object] | None = None) -> str:
    """Serialize the frozen configuration with a stable YAML layout."""
    payload = frozen_core_config() if document is None else dict(document)
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=88,
    )


def write_core_config(path: Path | None = None) -> Path:
    """Write ``configs/binance_open_data_core.yaml``. Results are not stored."""
    target = Path(path) if path is not None else _core_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialize_core_config(), encoding="utf-8")
    return target


def load_core_config(path: Path | None = None) -> dict[str, object]:
    """Load and validate the frozen configuration artifact."""
    target = Path(path) if path is not None else _core_config_path()
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise InvalidBinanceProtocolError("core configuration must be a mapping")
    _require_config_identity(document)
    return document


def core_config_sha256(path: Path | None = None) -> str:
    """SHA-256 of the serialized configuration file bytes."""
    target = Path(path) if path is not None else _core_config_path()
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _require_config_identity(document: Mapping[str, object]) -> None:
    """Reject a configuration that drifted from the frozen first-stage identity."""
    if document.get("branch") != BRANCH_IDENTITY:
        raise InvalidBinanceProtocolError("configuration branch identity mismatch")
    if document.get("production_panel_sha256") != PRODUCTION_PANEL_SHA256:
        raise InvalidBinanceProtocolError("configuration panel SHA-256 mismatch")
    if list(document.get("roster", [])) != list(CORE_ROSTER):
        raise InvalidBinanceProtocolError("core roster is not the frozen eleven-model list")
    if list(document.get("seeds", [])) != list(DEFAULT_SEEDS):
        raise InvalidBinanceProtocolError("seed list is not the frozen (0,1,2,3,4)")
    if document.get("allow_best_seed") is not False:
        raise InvalidBinanceProtocolError("best-seed selection is forbidden")
    if document.get("confirm_locked") is not True:
        raise InvalidBinanceProtocolError("CONFIRM must be locked in configuration")
    if document.get("pack_across_halt") is not False:
        raise InvalidBinanceProtocolError("packed lags across the halt are forbidden")
    candidates = document["candidates"]
    if not isinstance(candidates, dict):
        raise InvalidBinanceProtocolError("candidates must be a mapping")
    for name in ("RW", "HAR-DRD", "HARQ-DRD", "LW-linear", "LW-NL", "DCC", "DCC-NL"):
        rows = candidates[name]
        if not isinstance(rows, list) or len(rows) != 1:
            raise InvalidBinanceProtocolError(f"{name} must have exactly one configuration")
    if len(candidates["EWMA"]) != 20 or len(candidates["Ridge-DRD"]) != 20:
        raise InvalidBinanceProtocolError("EWMA and Ridge-DRD must have 20 configurations")
    xgb = candidates["XGBoost-DRD"]["grid"]
    lstm = candidates["LSTM-BEKK"]["grid"]
    if len(xgb) != 20 or len(lstm) != 20:
        raise InvalidBinanceProtocolError("XGBoost-DRD and LSTM-BEKK must have 20 configurations")
    for row in lstm:
        if "seed" in row or "seeds" in row:
            raise InvalidBinanceProtocolError("LSTM-BEKK candidates must not encode a seed")


def select_validation_configuration(
    records: Sequence[CandidateValidationRecord | Mapping[str, object]],
) -> str:
    """Return the admissible finite minimum-QLIKE ID. Exact ties use ID order.

    Incomplete support is not scored. SCREEN and CONFIRM fields are ignored.
    """
    admissible: list[tuple[float, str]] = []
    for item in records:
        if isinstance(item, CandidateValidationRecord):
            config_id = item.config_id
            complete = bool(item.complete_support)
            score = item.primary_score
        else:
            config_id = str(item["config_id"])
            complete = bool(item["complete_support"])
            score = item.get("primary_score")
        if not complete:
            continue
        if score is None:
            continue
        value = float(score)
        if value != value or value == float("inf") or value == float("-inf"):
            continue
        admissible.append((value, config_id))
    if not admissible:
        raise BinanceSelectionError(
            "every candidate is invalid. The family is failed rather than removed."
        )
    admissible.sort(key=lambda pair: (pair[0], pair[1]))
    return admissible[0][1]


class BinanceSegmentedProtocol:
    """Leak-free Binance schedule with a hard halt break and a CONFIRM lock.

    VALIDATION uses HISTORY_A plus VALIDATION dates only. SCREEN and CONFIRM
    use HISTORY_B plus SCREEN plus CONFIRM dates only. The two calendars are
    never concatenated.
    """

    def __init__(self, *, confirm_locked: bool = LOCKED) -> None:
        verify_frozen_segment_counts()
        segments = segment_dates()
        self._history_a = _utc_index(segments["HISTORY_A"])
        self._validation = _utc_index(segments["VALIDATION"])
        self._history_b = _utc_index(segments["HISTORY_B"])
        self._screen = _utc_index(segments["SCREEN"])
        self._confirm = _utc_index(segments["CONFIRM"])
        self._development = self._history_a.append(self._validation)
        self._confirmatory = self._history_b.append(self._screen).append(self._confirm)
        self.m = DEFAULT_ROLLING_WINDOW
        self.refit_cadence = DEFAULT_REFIT_CADENCE
        self.confirm_locked = bool(confirm_locked)
        if not self.confirm_locked:
            raise ConfirmLockedError(
                "BinanceSegmentedProtocol must be constructed locked. Pass "
                "unlock_confirm=True to confirm_targets or forecast_schedule."
            )
        self._frozen = True
        self._assert_calendars()

    def __setattr__(self, name: str, value: object) -> None:
        if name != "_frozen" and self.__dict__.get("_frozen"):
            raise AttributeError("BinanceSegmentedProtocol is frozen after construction")
        object.__setattr__(self, name, value)

    def _assert_calendars(self) -> None:
        """Reject packing, halt leakage, or a length drift from the freeze."""
        if HALT_DATE in {stamp.date() for stamp in self._development}:
            raise InvalidBinanceProtocolError("halt date entered the development calendar")
        if HALT_DATE in {stamp.date() for stamp in self._confirmatory}:
            raise InvalidBinanceProtocolError("halt date entered the confirmatory calendar")
        if self._development[-1].date() >= self._confirmatory[0].date():
            raise InvalidBinanceProtocolError("development and confirmatory calendars overlap")
        if (self._confirmatory[0].date() - self._development[-1].date()).days != 2:
            raise InvalidBinanceProtocolError("hard break is not a two-calendar-day gap")
        if len(self._history_a) != 250 or len(self._validation) != 250:
            raise InvalidBinanceProtocolError("development block lengths drifted")
        if len(self._history_b) != 250 or len(self._screen) != 500 or len(self._confirm) != 500:
            raise InvalidBinanceProtocolError("confirmatory block lengths drifted")

    def history_a_dates(self) -> pd.DatetimeIndex:
        return self._history_a.copy()

    def history_b_dates(self) -> pd.DatetimeIndex:
        return self._history_b.copy()

    def validation_targets(self) -> pd.DatetimeIndex:
        return self._validation.copy()

    def screen_targets(self) -> pd.DatetimeIndex:
        return self._screen.copy()

    def confirm_targets(self, *, unlock_confirm: bool = False) -> pd.DatetimeIndex:
        """Return CONFIRM target dates. Locked by default."""
        self._require_confirm_unlock(unlock_confirm)
        return self._confirm.copy()

    def confirm_bounds(self) -> tuple[dt.date, dt.date]:
        """Frozen CONFIRM calendar endpoints. Not target observations."""
        start, end = SEGMENT_BOUNDS["CONFIRM"]
        return start, end

    def calendar_for_block(self, block: BinanceBlock | str) -> pd.DatetimeIndex:
        """Return the uncrossed segment calendar that owns ``block``."""
        name = BinanceBlock(block)
        if name is BinanceBlock.VALIDATION:
            return self._development.copy()
        return self._confirmatory.copy()

    def forecast_schedule(
        self, block: BinanceBlock | str, *, unlock_confirm: bool = False
    ) -> BinanceBlockSchedule:
        """Block-local 21-origin schedule. The refit counter starts at 0."""
        name = BinanceBlock(block)
        targets = self._block_targets(name, unlock_confirm=unlock_confirm)
        calendar = self.calendar_for_block(name)
        steps = build_schedule(
            calendar,
            targets,
            m=self.m,
            refit_cadence=self.refit_cadence,
        )
        self._assert_schedule(name, calendar, steps)
        return BinanceBlockSchedule(block=name.value, calendar=calendar, steps=steps)

    def _block_targets(
        self, block: BinanceBlock, *, unlock_confirm: bool
    ) -> pd.DatetimeIndex:
        if block is BinanceBlock.VALIDATION:
            return self.validation_targets()
        if block is BinanceBlock.SCREEN:
            return self.screen_targets()
        if block is BinanceBlock.CONFIRM:
            return self.confirm_targets(unlock_confirm=unlock_confirm)
        raise InvalidBinanceProtocolError(f"unknown evaluation block {block}")

    def _assert_schedule(
        self,
        block: BinanceBlock,
        calendar: pd.DatetimeIndex,
        steps: tuple[ForecastStep, ...],
    ) -> None:
        """Check halt exclusion, one-day origin/target pairs, and window length."""
        halt = pd.Timestamp(HALT_DATE)
        for position, step in enumerate(steps):
            if step.origin == halt or step.target == halt:
                raise InvalidBinanceProtocolError("halt date is an origin or target")
            window = step.estimation_dates(calendar)
            if halt in window:
                raise InvalidBinanceProtocolError("estimation window contains the halt")
            if len(window) != self.m:
                raise LookaheadError("estimation window is not length m=250")
            if (step.target.normalize() - step.origin.normalize()).days != 1:
                raise InvalidBinanceProtocolError(
                    f"{block.value} origin {step.origin.date()} does not precede "
                    f"target {step.target.date()} by one UTC day"
                )
            if step.target in window:
                raise LookaheadError("target entered the estimation window")
            expected_refit = position % self.refit_cadence == 0
            if bool(step.refit) != expected_refit:
                raise InvalidBinanceProtocolError("block-local refit cadence drifted")

    def _require_confirm_unlock(self, unlock_confirm: bool) -> None:
        if not unlock_confirm:
            raise ConfirmLockedError(
                "CONFIRM is locked. Pass unlock_confirm=True at this protocol "
                "boundary. The default is locked."
            )

    def __getattr__(self, name: str) -> object:
        if "confirm" in name.lower():
            raise ConfirmLockedError(
                f"{name!r} is not a public CONFIRM accessor. Use "
                "confirm_targets(unlock_confirm=True) or "
                "forecast_schedule('CONFIRM', unlock_confirm=True)."
            )
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")


def build_binance_segmented_protocol() -> BinanceSegmentedProtocol:
    """Construct the locked first-stage Binance schedule object."""
    return BinanceSegmentedProtocol(confirm_locked=True)
