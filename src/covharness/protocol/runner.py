"""Serial rolling forecast generation along a leak-free schedule.

Parameter refit, daily observable-state update, and forecast formation are
distinct. The 21-origin cadence is estimator refit, not forecast cadence.
This runner does not compute losses or inference. It does not inspect
VALIDATION, SCREEN, or CONFIRM labels except when a caller requests a
block schedule through :func:`run_block_forecasts`. CONFIRM remains locked
unless that caller passes ``unlock_confirm=True``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from covharness.models.base import CovarianceModel
from covharness.models.capabilities import (
    FitInput,
    RollingCadence,
    UpdateObservable,
    capabilities_of,
)
from covharness.models.exceptions import InvalidModelInputError
from covharness.protocol.constants import BlockName
from covharness.protocol.exceptions import LookaheadError
from covharness.protocol.rolling import ForecastStep
from covharness.protocol.splits import TemporalProtocol


class RollingAction(str, Enum):
    """Dispatch performed at one origin before ``forecast``."""

    FIT = "fit"
    UPDATE = "update"
    UPDATE_WINDOW = "update_window"
    HOLD = "hold"


@dataclass(frozen=True)
class OriginPayload:
    """Origin-t information supplied to one model step.

    Optional arrays are present only when the caller provided the matching
    panel. Every window ends at origin ``t``. Target ``t+1`` values are not
    included.
    """

    origin: pd.Timestamp
    target: pd.Timestamp
    origin_index: int
    target_index: int
    refit: bool
    m: int
    return_window: NDArray[np.floating] | None = None
    current_return: NDArray[np.floating] | None = None
    rcov_window: NDArray[np.floating] | None = None
    current_rcov: NDArray[np.floating] | None = None
    rq_window: NDArray[np.floating] | None = None
    current_rq: NDArray[np.floating] | None = None


@dataclass(frozen=True)
class RollingForecastRecord:
    """One stored one-day-ahead covariance forecast.

    ``covariance`` is an independent copy. Later model-state mutation does
    not change this record.
    """

    model_name: str
    origin: pd.Timestamp
    target: pd.Timestamp
    refit: bool
    action: RollingAction
    covariance: NDArray[np.floating]


def run_block_forecasts(
    *,
    model: CovarianceModel,
    protocol: TemporalProtocol,
    block: BlockName,
    daily_returns: ArrayLike | None = None,
    realized_covariances: ArrayLike | None = None,
    realized_quarticity: ArrayLike | None = None,
    unlock_confirm: bool = False,
) -> tuple[RollingForecastRecord, ...]:
    """Run one model on a protocol evaluation block.

    CONFIRM raises :class:`~covharness.protocol.exceptions.ConfirmLockedError`
    unless ``unlock_confirm`` is true. The model never receives the block
    label.
    """
    schedule = protocol.forecast_schedule(block, unlock_confirm=unlock_confirm)
    return run_rolling_forecasts(
        model=model,
        schedule=schedule,
        calendar=protocol.calendar,
        daily_returns=daily_returns,
        realized_covariances=realized_covariances,
        realized_quarticity=realized_quarticity,
    )


def run_rolling_forecasts(
    *,
    model: CovarianceModel,
    schedule: Sequence[ForecastStep],
    calendar: pd.DatetimeIndex,
    daily_returns: ArrayLike | None = None,
    realized_covariances: ArrayLike | None = None,
    realized_quarticity: ArrayLike | None = None,
) -> tuple[RollingForecastRecord, ...]:
    """Advance one model along ``schedule`` and store a forecast at each origin.

    Execution is serial. Forecast order matches the schedule. The same origin
    never receives both ``fit`` and a subsequent state update.
    """
    returns = _optional_array(daily_returns, "daily_returns")
    rcov = _optional_array(realized_covariances, "realized_covariances")
    rq = _optional_array(realized_quarticity, "realized_quarticity")
    records: list[RollingForecastRecord] = []
    for step in schedule:
        # Build the origin-t payload. Target t+1 is never sliced in.
        payload = build_origin_payload(
            step=step,
            calendar=calendar,
            daily_returns=returns,
            realized_covariances=rcov,
            realized_quarticity=rq,
        )
        if step.refit:
            action = _fit_at_origin(model, payload)
        else:
            action = _advance_at_origin(model, payload)
        forecast = model.forecast()
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
    return tuple(records)


def build_origin_payload(
    *,
    step: ForecastStep,
    calendar: pd.DatetimeIndex,
    daily_returns: NDArray[np.floating] | None = None,
    realized_covariances: NDArray[np.floating] | None = None,
    realized_quarticity: NDArray[np.floating] | None = None,
) -> OriginPayload:
    """Slice origin-t arrays from aligned calendar panels."""
    _reject_leaking_step(step, calendar)
    return OriginPayload(
        origin=step.origin,
        target=step.target,
        origin_index=step.origin_index,
        target_index=step.target_index,
        refit=bool(step.refit),
        m=int(step.m),
        return_window=_slice_window(daily_returns, step, "daily_returns"),
        current_return=_slice_current(daily_returns, step, "daily_returns"),
        rcov_window=_slice_window(realized_covariances, step, "realized_covariances"),
        current_rcov=_slice_current(realized_covariances, step, "realized_covariances"),
        rq_window=_slice_window(realized_quarticity, step, "realized_quarticity"),
        current_rq=_slice_current(realized_quarticity, step, "realized_quarticity"),
    )


def _fit_at_origin(model: CovarianceModel, payload: OriginPayload) -> RollingAction:
    """Re-estimate fitted quantities from the current m-day window."""
    caps = capabilities_of(model)
    if caps.fit_input is FitInput.REALIZED_COVARIANCE:
        model.fit(_require_field(payload.rcov_window, "rcov_window", payload))
    elif caps.fit_input is FitInput.REALIZED_COVARIANCE_AND_RQ:
        model.fit(
            _require_field(payload.rcov_window, "rcov_window", payload),
            _require_field(payload.rq_window, "rq_window", payload),
        )
    elif caps.fit_input is FitInput.DAILY_RETURN:
        model.fit(_require_field(payload.return_window, "return_window", payload))
    else:
        raise InvalidModelInputError(
            f"unsupported fit input {caps.fit_input} at origin {payload.origin.date()}"
        )
    return RollingAction.FIT


def _advance_at_origin(model: CovarianceModel, payload: OriginPayload) -> RollingAction:
    """Ingest origin-t information once with fitted quantities frozen."""
    caps = capabilities_of(model)
    cadence = caps.rolling_cadence
    if cadence is RollingCadence.REFIT_HOLD:
        return RollingAction.HOLD
    if cadence is RollingCadence.ORIGIN_MAP:
        model.update(_require_field(payload.current_rcov, "current_rcov", payload))
        return RollingAction.UPDATE
    if cadence is RollingCadence.WINDOW_STATE:
        rcov_window = _require_field(payload.rcov_window, "rcov_window", payload)
        if caps.update_observable is UpdateObservable.REALIZED_COVARIANCE_AND_RQ_WINDOW:
            model.update_window(
                rcov_window,
                _require_field(payload.rq_window, "rq_window", payload),
            )
        else:
            model.update_window(rcov_window)
        return RollingAction.UPDATE_WINDOW
    if cadence is RollingCadence.RECURSIVE_STATE:
        if caps.update_observable is UpdateObservable.REALIZED_COVARIANCE:
            model.update(_require_field(payload.current_rcov, "current_rcov", payload))
        elif caps.update_observable is UpdateObservable.DAILY_RETURN:
            model.update(
                _require_field(payload.current_return, "current_return", payload)
            )
        else:
            raise InvalidModelInputError(
                f"unsupported recursive observable {caps.update_observable} "
                f"at origin {payload.origin.date()}"
            )
        return RollingAction.UPDATE
    raise InvalidModelInputError(
        f"unsupported rolling cadence {cadence} at origin {payload.origin.date()}"
    )


def _reject_leaking_step(step: ForecastStep, calendar: pd.DatetimeIndex) -> None:
    """Reject a step whose window would include the target date."""
    if step.window_end_index != step.origin_index + 1:
        raise LookaheadError(
            "estimation window must end at the origin exclusive of the target"
        )
    if step.target_index != step.origin_index + 1:
        raise LookaheadError("target index must be origin index plus one")
    if step.window_end_index - step.window_start_index != step.m:
        raise LookaheadError("estimation window length must equal m")
    if step.origin_index >= len(calendar) or step.target_index >= len(calendar):
        raise LookaheadError("origin or target is outside the supplied calendar")
    if calendar[step.origin_index] != step.origin:
        raise LookaheadError("origin does not match calendar[origin_index]")
    if calendar[step.target_index] != step.target:
        raise LookaheadError("target does not match calendar[target_index]")


def _optional_array(values: ArrayLike | None, name: str) -> NDArray[np.floating] | None:
    """Copy a caller panel or return None. The array is not interpreted yet."""
    if values is None:
        return None
    array = np.array(values, dtype=float, copy=True)
    if array.size == 0:
        raise InvalidModelInputError(f"{name} must be non-empty when supplied")
    return array


def _slice_window(
    panel: NDArray[np.floating] | None,
    step: ForecastStep,
    name: str,
) -> NDArray[np.floating] | None:
    """Copy ``panel[start:end]`` through the origin. Target rows are excluded."""
    if panel is None:
        return None
    _require_origin_row(panel, step, name)
    window = np.array(
        panel[step.window_start_index : step.window_end_index],
        dtype=float,
        copy=True,
    )
    if window.shape[0] != step.m:
        raise LookaheadError(
            f"{name} window must have length m={step.m}; got {window.shape[0]}"
        )
    return window


def _slice_current(
    panel: NDArray[np.floating] | None,
    step: ForecastStep,
    name: str,
) -> NDArray[np.floating] | None:
    """Copy the origin-t row. The target row is not read."""
    if panel is None:
        return None
    _require_origin_row(panel, step, name)
    return np.array(panel[step.origin_index], dtype=float, copy=True)


def _require_origin_row(
    panel: NDArray[np.floating],
    step: ForecastStep,
    name: str,
) -> None:
    """Reject a panel that cannot supply origin t without reading t+1."""
    if panel.shape[0] <= step.origin_index:
        raise InvalidModelInputError(
            f"{name} must cover origin index {step.origin_index}; "
            f"got length {panel.shape[0]}"
        )


def _require_field(
    value: NDArray[np.floating] | None,
    name: str,
    payload: OriginPayload,
) -> NDArray[np.floating]:
    """Reject a missing origin field required by the dispatched model."""
    if value is None:
        raise InvalidModelInputError(
            f"{name} is required at origin {payload.origin.date()} "
            f"for this model cadence"
        )
    return value
