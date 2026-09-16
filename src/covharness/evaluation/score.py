"""Align rolling forecasts to target-day realized covariance and score them.

This adapter does not reimplement reduced QLIKE, squared Frobenius, or
Block 3 inference. Point losses are computed by the existing one-date
functions ``reduced_qlike_loss`` and ``squared_frobenius_loss``. Forecast
generation remains in the serial runner. Evaluation never sees origin-day
``S_t`` as the scoring target.

The target of record ``(origin t, target t+1, H_{t+1|t})`` is ``S_{t+1}``,
looked up by the target calendar key.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from covharness.evaluation.exceptions import EvaluationAlignmentError
from covharness.inference.differentials import loss_differential
from covharness.losses.frobenius import squared_frobenius_loss
from covharness.losses.qlike import reduced_qlike_loss
from covharness.protocol.runner import RollingAction, RollingForecastRecord

LOSS_REDUCED_QLIKE = "reduced_qlike"
LOSS_SQUARED_FROBENIUS = "squared_frobenius"
DEFAULT_LOSSES = (LOSS_REDUCED_QLIKE, LOSS_SQUARED_FROBENIUS)
MISMATCH_ERROR = "error"
MISMATCH_INTERSECTION = "intersection"
VALID_MISMATCH_POLICIES = (MISMATCH_ERROR, MISMATCH_INTERSECTION)


class EvaluationLoss(str, Enum):
    """Named covariance-space losses scored by this adapter."""

    REDUCED_QLIKE = LOSS_REDUCED_QLIKE
    SQUARED_FROBENIUS = LOSS_SQUARED_FROBENIUS


_LOSS_FUNCTIONS = {
    LOSS_REDUCED_QLIKE: reduced_qlike_loss,
    LOSS_SQUARED_FROBENIUS: squared_frobenius_loss,
}


@dataclass(frozen=True)
class TargetCovariancePanel:
    """Calendar-keyed realized-covariance cube used as the evaluation proxy.

    ``realized_covariances[i]`` is the proxy for ``calendar[i]``. Lookup is
    by timestamp, not by positional alignment with forecast records.
    """

    calendar: pd.DatetimeIndex
    realized_covariances: NDArray[np.floating]

    def __post_init__(self) -> None:
        calendar = pd.DatetimeIndex(self.calendar)
        cube = np.array(self.realized_covariances, dtype=float, copy=True)
        if not calendar.is_unique:
            raise EvaluationAlignmentError("target calendar dates must be unique")
        if cube.ndim != 3 or cube.shape[1] != cube.shape[2]:
            raise EvaluationAlignmentError(
                "target realized covariances must have shape (T, N, N); "
                f"got {cube.shape}"
            )
        if cube.shape[0] != len(calendar):
            raise EvaluationAlignmentError(
                "target calendar length must equal the realized-covariance "
                f"time dimension; got {len(calendar)} and {cube.shape[0]}"
            )
        if cube.shape[1] < 1:
            raise EvaluationAlignmentError("target covariances require N >= 1")
        object.__setattr__(self, "calendar", calendar.copy())
        object.__setattr__(self, "realized_covariances", cube)

    @property
    def n_assets(self) -> int:
        return int(self.realized_covariances.shape[1])

    def index_of(self, stamp: pd.Timestamp, *, role: str) -> int:
        """Return the calendar position of ``stamp``. Missing dates fail."""
        try:
            located = self.calendar.get_loc(pd.Timestamp(stamp))
        except KeyError as exc:
            raise EvaluationAlignmentError(
                f"{role} {pd.Timestamp(stamp).date()} is not on the target calendar"
            ) from exc
        if not isinstance(located, (int, np.integer)):
            raise EvaluationAlignmentError(
                f"{role} {pd.Timestamp(stamp).date()} must match exactly one calendar row"
            )
        return int(located)

    def covariance_at(self, stamp: pd.Timestamp, *, role: str) -> NDArray[np.floating]:
        """Copy the covariance stored under ``stamp``."""
        index = self.index_of(stamp, role=role)
        return np.array(self.realized_covariances[index], dtype=float, copy=True)

    def is_next_date(self, origin: pd.Timestamp, target: pd.Timestamp) -> bool:
        """True iff ``target`` is the next calendar key after ``origin``.

        Storage order is ignored. Chronological rank uses the unique date keys.
        """
        ordered = self.calendar.sort_values()
        origin_rank = int(ordered.get_loc(pd.Timestamp(origin)))
        target_rank = int(ordered.get_loc(pd.Timestamp(target)))
        return target_rank == origin_rank + 1


def target_covariance_panel(
    calendar: pd.DatetimeIndex,
    realized_covariances: ArrayLike,
) -> TargetCovariancePanel:
    """Copy a calendar-aligned realized-covariance cube."""
    return TargetCovariancePanel(
        calendar=pd.DatetimeIndex(calendar),
        realized_covariances=np.asarray(realized_covariances, dtype=float),
    )


@dataclass(frozen=True)
class AlignedForecastTarget:
    """One forecast paired with the target-day realized covariance ``S_{t+1}``."""

    model_name: str
    origin: pd.Timestamp
    target: pd.Timestamp
    forecast: NDArray[np.floating]
    proxy: NDArray[np.floating]
    refit: bool
    action: RollingAction


@dataclass(frozen=True)
class LossRecord:
    """One point loss. Covariance matrices are not stored here."""

    model_name: str
    origin: pd.Timestamp
    target: pd.Timestamp
    loss_name: str
    loss_value: float
    refit: bool
    action: RollingAction


@dataclass(frozen=True)
class LossPanel:
    """Aligned model-by-date losses for one named loss.

    ``values`` has shape ``(n_targets, n_models)``. Rows follow chronological
    target order. Columns follow ``model_names``.
    """

    model_names: tuple[str, ...]
    targets: pd.DatetimeIndex
    loss_name: str
    values: NDArray[np.floating]

    def column(self, model_name: str) -> NDArray[np.floating]:
        """Return the loss series for one model. The array is copied."""
        if model_name not in self.model_names:
            raise EvaluationAlignmentError(
                f"model {model_name!r} is not in this loss panel"
            )
        index = self.model_names.index(model_name)
        return np.array(self.values[:, index], dtype=float, copy=True)


@dataclass(frozen=True)
class ModelLossSummary:
    """Descriptive statistics for one model and one loss. Not a ranking."""

    model_name: str
    loss_name: str
    n_forecasts: int
    mean_loss: float
    median_loss: float
    std_loss: float


def align_forecast_records(
    records: Sequence[RollingForecastRecord],
    targets: TargetCovariancePanel,
) -> tuple[AlignedForecastTarget, ...]:
    """Pair each forecast with ``S`` on its target date, never the origin date."""
    if len(records) < 1:
        raise EvaluationAlignmentError("forecast records must be non-empty")
    seen: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}
    aligned: list[AlignedForecastTarget] = []
    for record in records:
        origin = pd.Timestamp(record.origin)
        target = pd.Timestamp(record.target)
        key = (str(record.model_name), target)
        if key in seen:
            raise EvaluationAlignmentError(
                f"duplicate target {target.date()} for model {record.model_name!r}"
            )
        seen[key] = origin
        targets.index_of(origin, role="origin")
        targets.index_of(target, role="target")
        if not targets.is_next_date(origin, target):
            raise EvaluationAlignmentError(
                "evaluation target must be the next calendar date after the origin; "
                f"got origin {origin.date()} and target {target.date()}"
            )
        forecast = np.array(record.covariance, dtype=float, copy=True)
        proxy = targets.covariance_at(target, role="target")
        if forecast.shape != proxy.shape:
            raise EvaluationAlignmentError(
                "forecast and target covariance dimensions must match; "
                f"got {forecast.shape} and {proxy.shape} at {target.date()}"
            )
        if forecast.shape != (targets.n_assets, targets.n_assets):
            raise EvaluationAlignmentError(
                "forecast width must equal the target panel N="
                f"{targets.n_assets}; got {forecast.shape} at {target.date()}"
            )
        aligned.append(
            AlignedForecastTarget(
                model_name=str(record.model_name),
                origin=origin,
                target=target,
                forecast=forecast,
                proxy=proxy,
                refit=bool(record.refit),
                action=record.action,
            )
        )
    return tuple(aligned)


def score_aligned_pairs(
    pairs: Sequence[AlignedForecastTarget],
    *,
    losses: Sequence[str] = DEFAULT_LOSSES,
) -> tuple[LossRecord, ...]:
    """Score each aligned pair with existing one-date loss functions."""
    names = _require_loss_names(losses)
    scored: list[LossRecord] = []
    for pair in pairs:
        for loss_name in names:
            value = float(_LOSS_FUNCTIONS[loss_name](pair.proxy, pair.forecast))
            scored.append(
                LossRecord(
                    model_name=pair.model_name,
                    origin=pair.origin,
                    target=pair.target,
                    loss_name=loss_name,
                    loss_value=value,
                    refit=pair.refit,
                    action=pair.action,
                )
            )
    return tuple(scored)


def score_forecast_records(
    records: Sequence[RollingForecastRecord],
    targets: TargetCovariancePanel,
    *,
    losses: Sequence[str] = DEFAULT_LOSSES,
) -> tuple[LossRecord, ...]:
    """Align rolling forecasts to ``S_{t+1}`` and compute point losses."""
    pairs = align_forecast_records(records, targets)
    return score_aligned_pairs(pairs, losses=losses)


def build_loss_panel(
    loss_records: Sequence[LossRecord],
    *,
    loss_name: str,
    model_order: Sequence[str] | None = None,
    date_support: str = MISMATCH_ERROR,
) -> LossPanel:
    """Assemble a ``(T, M)`` loss matrix for one named loss.

    Default ``date_support='error'`` requires identical target dates across
    models. ``date_support='intersection'`` keeps only dates present for every
    requested model. Dates are never dropped silently under the default.
    """
    name = _require_single_loss(loss_name)
    policy = str(date_support)
    if policy not in VALID_MISMATCH_POLICIES:
        raise EvaluationAlignmentError(
            f"date_support must be one of {VALID_MISMATCH_POLICIES}; got {date_support!r}"
        )
    selected = [record for record in loss_records if record.loss_name == name]
    if len(selected) < 1:
        raise EvaluationAlignmentError(f"no loss records named {name!r}")
    by_model: dict[str, dict[pd.Timestamp, LossRecord]] = {}
    appearance: list[str] = []
    for record in selected:
        model = record.model_name
        if model not in by_model:
            by_model[model] = {}
            appearance.append(model)
        target = pd.Timestamp(record.target)
        if target in by_model[model]:
            raise EvaluationAlignmentError(
                f"duplicate target {target.date()} for model {model!r} and loss {name!r}"
            )
        by_model[model][target] = record
    names = _resolve_model_order(appearance, model_order)
    target_sets = [set(by_model[model]) for model in names]
    if policy == MISMATCH_ERROR:
        first = target_sets[0]
        for model, dates in zip(names, target_sets, strict=True):
            if dates != first:
                raise EvaluationAlignmentError(
                    "compared models must share identical target dates; "
                    f"{names[0]!r} has {len(first)} dates and {model!r} has {len(dates)}"
                )
        common = first
    else:
        common = set.intersection(*target_sets) if target_sets else set()
        if len(common) < 1:
            raise EvaluationAlignmentError(
                "model target-date intersection is empty"
            )
    ordered_targets = pd.DatetimeIndex(sorted(common))
    values = np.empty((len(ordered_targets), len(names)), dtype=float)
    for column, model in enumerate(names):
        for row, target in enumerate(ordered_targets):
            values[row, column] = float(by_model[model][target].loss_value)
    return LossPanel(
        model_names=tuple(names),
        targets=ordered_targets,
        loss_name=name,
        values=np.array(values, dtype=float, copy=True),
    )


def summarize_loss_panel(panel: LossPanel) -> tuple[ModelLossSummary, ...]:
    """Return roster-ordered descriptive statistics. This is not a ranking."""
    summaries: list[ModelLossSummary] = []
    for column, model_name in enumerate(panel.model_names):
        series = np.array(panel.values[:, column], dtype=float, copy=True)
        n_obs = int(series.size)
        std_loss = float(series.std(ddof=1)) if n_obs >= 2 else 0.0
        summaries.append(
            ModelLossSummary(
                model_name=model_name,
                loss_name=panel.loss_name,
                n_forecasts=n_obs,
                mean_loss=float(series.mean()),
                median_loss=float(np.median(series)),
                std_loss=std_loss,
            )
        )
    return tuple(summaries)


def loss_differential_from_panel(
    panel: LossPanel,
    model_a: str,
    model_b: str,
) -> NDArray[np.floating]:
    """Return ``d_t = L_{A,t} - L_{B,t}`` using the existing Block 3A convention.

    Negative entries are dates on which A has lower loss. Positive entries
    are dates on which B has lower loss. This adapter does not run DM, SPA,
    or MCS.
    """
    return loss_differential(panel.column(model_a), panel.column(model_b))


def _require_loss_names(losses: Sequence[str]) -> tuple[str, ...]:
    """Reject an empty or unknown loss roster."""
    if len(losses) < 1:
        raise EvaluationAlignmentError("at least one evaluation loss is required")
    names = tuple(str(name) for name in losses)
    unknown = [name for name in names if name not in _LOSS_FUNCTIONS]
    if unknown:
        raise EvaluationAlignmentError(
            f"unsupported evaluation loss {unknown[0]!r}. "
            f"Supported names are {tuple(_LOSS_FUNCTIONS)}"
        )
    return names


def _require_single_loss(loss_name: str) -> str:
    """Reject a loss name the adapter does not score."""
    name = str(loss_name)
    if name not in _LOSS_FUNCTIONS:
        raise EvaluationAlignmentError(
            f"unsupported evaluation loss {name!r}. "
            f"Supported names are {tuple(_LOSS_FUNCTIONS)}"
        )
    return name


def _resolve_model_order(
    appearance: Sequence[str], requested: Sequence[str] | None
) -> tuple[str, ...]:
    """Keep caller roster order when supplied, otherwise first appearance."""
    if requested is None:
        return tuple(appearance)
    names = tuple(str(name) for name in requested)
    if len(names) != len(set(names)):
        raise EvaluationAlignmentError("model_order must not contain duplicates")
    missing = [name for name in names if name not in appearance]
    extra = [name for name in appearance if name not in names]
    if missing or extra:
        raise EvaluationAlignmentError(
            "model_order must list exactly the models present in the loss records"
        )
    return names
