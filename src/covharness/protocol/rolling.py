"""Rolling estimation windows and monthly refit scheduling.

At forecast origin ``t`` the model may use observations through ``t``. The
target is the next trading day ``t+1``. The target date does not enter fitting,
transformation, feature creation, graph construction, scaling, or hyperparameter
selection. The window length ``m`` is part of the forecasting method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from covharness.protocol.constants import DEFAULT_REFIT_CADENCE, DEFAULT_ROLLING_WINDOW
from covharness.protocol.exceptions import LookaheadError


@dataclass(frozen=True)
class ForecastStep:
    """One leak-free origin/target pair on the trading-date index.

    Estimation uses the half-open trading-date interval
    ``[window_start, target)``, which is the inclusive window through
    ``origin`` of length ``m``.
    """

    origin: pd.Timestamp
    target: pd.Timestamp
    window_start: pd.Timestamp
    origin_index: int
    target_index: int
    window_start_index: int
    window_end_index: int
    refit: bool
    m: int

    def estimation_dates(self, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Return the ``m`` trading dates through the origin, excluding the target."""
        return calendar[self.window_start_index : self.window_end_index]


def origin_index_for_target(calendar: pd.DatetimeIndex, target: pd.Timestamp) -> int:
    """Return the position of the trading day immediately before ``target``."""
    target_index = _locate(calendar, target, "target")
    if target_index == 0:
        raise LookaheadError("the first calendar date has no origin")
    return int(target_index) - 1


def estimation_window(
    calendar: pd.DatetimeIndex,
    origin: pd.Timestamp,
    m: int = DEFAULT_ROLLING_WINDOW,
) -> pd.DatetimeIndex:
    """Return ``m`` dates through ``origin``, as ``calendar[i-m+1 : i+1]``."""
    step = step_at_origin(calendar, origin, m=m, refit=False)
    return step.estimation_dates(calendar)


def step_at_origin(
    calendar: pd.DatetimeIndex,
    origin: pd.Timestamp,
    m: int = DEFAULT_ROLLING_WINDOW,
    refit: bool = False,
) -> ForecastStep:
    """Build the unique origin/target/window triple for ``origin``."""
    if m < 1:
        raise ValueError("rolling window m must be at least 1")
    origin_index = _locate(calendar, origin, "origin")
    window_start_index = int(origin_index) - m + 1
    if window_start_index < 0:
        raise LookaheadError(
            f"origin {origin.date()} has fewer than m={m} estimation dates"
        )
    target_index = int(origin_index) + 1
    if target_index >= len(calendar):
        raise LookaheadError(
            f"origin {origin.date()} has no next-day target on the calendar"
        )
    window_end_index = int(origin_index) + 1
    target = calendar[target_index]
    window_start = calendar[window_start_index]
    step = ForecastStep(
        origin=pd.Timestamp(origin),
        target=pd.Timestamp(target),
        window_start=pd.Timestamp(window_start),
        origin_index=int(origin_index),
        target_index=target_index,
        window_start_index=window_start_index,
        window_end_index=window_end_index,
        refit=bool(refit),
        m=m,
    )
    assert_no_target_leakage(step, calendar)
    return step


def build_schedule(
    calendar: pd.DatetimeIndex,
    target_dates: pd.DatetimeIndex,
    m: int = DEFAULT_ROLLING_WINDOW,
    refit_cadence: int = DEFAULT_REFIT_CADENCE,
) -> tuple[ForecastStep, ...]:
    """Schedule leak-free steps for an ordered target block.

    Refits occur at the first origin and then every ``refit_cadence`` forecast
    origins along this schedule. Cadence is counted in forecast steps, each of
    which is one trading day, so the default 21 is a monthly trading-day gap.
    """
    if refit_cadence < 1:
        raise ValueError("refit_cadence must be at least 1")
    if len(target_dates) == 0:
        return ()
    steps: list[ForecastStep] = []
    for schedule_position, target in enumerate(target_dates):
        origin_index = origin_index_for_target(calendar, pd.Timestamp(target))
        origin = calendar[origin_index]
        refit = schedule_position % refit_cadence == 0
        steps.append(step_at_origin(calendar, origin, m=m, refit=refit))
    return tuple(steps)


def assert_no_target_leakage(
    step: ForecastStep, calendar: pd.DatetimeIndex
) -> None:
    """Reject a step whose estimation window meets or passes the target."""
    window = step.estimation_dates(calendar)
    if len(window) != step.m:
        raise LookaheadError(
            f"estimation window length is {len(window)}, expected m={step.m}"
        )
    if step.target_index != step.origin_index + 1:
        raise LookaheadError("the target must be the next trading day after the origin")
    if step.window_end_index != step.origin_index + 1:
        raise LookaheadError("the half-open estimation window must end at the target")
    if step.target in window:
        raise LookaheadError("the target date may not enter the estimation window")
    if (window > step.origin).any():
        raise LookaheadError("estimation dates after the origin are not permitted")
    if window[-1] != step.origin:
        raise LookaheadError("the estimation window must include the origin as its last date")
    if window[0] != step.window_start:
        raise LookaheadError("the estimation window does not start at window_start")


def _locate(calendar: pd.DatetimeIndex, stamp: pd.Timestamp, name: str) -> int:
    """Locate a timestamp on a unique calendar, or fail clearly."""
    try:
        located = calendar.get_loc(pd.Timestamp(stamp))
    except KeyError as exc:
        raise LookaheadError(f"{name} {pd.Timestamp(stamp).date()} is not on the calendar") from exc
    if not isinstance(located, (int, np.integer)):
        raise LookaheadError(f"{name} must match exactly one calendar date")
    return int(located)
