"""CONFIRM schedule and forecast dispatch.

The production panel is not rewritten. Model mathematics are unchanged.
``unlock_confirm=True`` appears only in ``confirm_schedule``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from covharness.data.binance_calendar import BINANCE_ASSETS, HALT_DATE
from covharness.protocol.binance import (
    BinanceBlock,
    BinanceSegmentedProtocol,
    build_binance_segmented_protocol,
)
from covharness.protocol.rolling import ForecastStep
from covharness.protocol.runner import (
    RollingForecastRecord,
    _advance_at_origin,
    _fit_at_origin,
    build_origin_payload,
)

CONFIRM_START = dt.date(2025, 4, 13)
CONFIRM_END = dt.date(2026, 8, 25)
FIRST_ORIGIN = dt.date(2025, 4, 12)
N_TARGETS = 500
WINDOW = 250


@dataclass(frozen=True)
class ConfirmView:
    """HISTORY_B, SCREEN, and CONFIRM rows. The halt is absent."""

    dates: tuple[dt.date, ...]
    calendar: pd.DatetimeIndex
    assets: tuple[str, ...]
    daily_returns: NDArray[np.floating]
    rcov: NDArray[np.floating]
    rq: NDArray[np.floating]


def confirm_schedule(
    protocol: BinanceSegmentedProtocol | None = None,
):
    """Return the CONFIRM schedule. This is the only unlock in the package."""
    protocol = protocol or build_binance_segmented_protocol()
    return protocol.forecast_schedule(BinanceBlock.CONFIRM, unlock_confirm=True)


def assert_confirm_schedule(schedule) -> None:
    """Check length, one-day lag, refit cadence, and window exclusion."""
    steps = tuple(schedule.steps)
    if len(steps) != N_TARGETS:
        raise ValueError(f"CONFIRM schedule has {len(steps)} steps; expected {N_TARGETS}")
    if steps[0].origin.date() != FIRST_ORIGIN or steps[0].target.date() != CONFIRM_START:
        raise ValueError("CONFIRM does not start at origin 2025-04-12 and target 2025-04-13")
    if steps[-1].target.date() != CONFIRM_END:
        raise ValueError("CONFIRM does not end on 2026-08-25")
    for index, step in enumerate(steps):
        origin = step.origin.date()
        target = step.target.date()
        if (target - origin).days != 1:
            raise ValueError(f"{target.isoformat()} is not one day after its origin")
        window = [stamp.date() for stamp in step.estimation_dates(schedule.calendar)]
        if len(window) != WINDOW:
            raise ValueError(f"{target.isoformat()} window length is {len(window)}")
        if target in window or any(day > origin or day == HALT_DATE for day in window):
            raise ValueError(f"{target.isoformat()} window contains the target, a future day, or the halt")
        if bool(step.refit) != (index % 21 == 0):
            raise ValueError(f"{target.isoformat()} refit flag is not index mod 21")


def load_confirm_view(panel_path, schedule) -> ConfirmView:
    """Slice the confirmatory calendar from the production panel."""
    blob = np.load(panel_path)
    stored = [str(value) for value in np.asarray(blob["dates"]).tolist()]
    lookup = {day: index for index, day in enumerate(stored)}
    dates = tuple(stamp.date() for stamp in schedule.calendar)
    if HALT_DATE in dates:
        raise ValueError("the halt is in the CONFIRM calendar")
    missing = [day.isoformat() for day in dates if day.isoformat() not in lookup]
    if missing:
        raise ValueError(f"production panel is missing {missing[0]}")
    rows = [lookup[day.isoformat()] for day in dates]
    return ConfirmView(
        dates=dates,
        calendar=pd.DatetimeIndex(dates),
        assets=tuple(str(asset) for asset in blob["assets"].tolist())
        if "assets" in blob.files
        else BINANCE_ASSETS,
        daily_returns=np.array(blob["daily_returns"][rows], dtype=float),
        rcov=np.array(blob["rcov_5min"][rows], dtype=float),
        rq=np.array(blob["rq_per_asset"][rows], dtype=float),
    )


def _local_step(step: ForecastStep, calendar: pd.DatetimeIndex) -> ForecastStep:
    """Reindex one schedule step onto the confirmatory view."""
    origin_index = int(calendar.get_loc(step.origin))
    target_index = int(calendar.get_loc(step.target))
    window_start_index = origin_index - int(step.m) + 1
    if window_start_index < 0:
        raise ValueError(f"{step.target.date().isoformat()} window starts before the view")
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


def run_confirm_forecasts(
    *,
    model,
    schedule,
    view: ConfirmView,
    on_origin=None,
    semivariance_positive: NDArray[np.floating] | None = None,
    semivariance_negative: NDArray[np.floating] | None = None,
    feature_panel: NDArray[np.floating] | None = None,
    proxy_panel: NDArray[np.floating] | None = None,
) -> tuple[list[RollingForecastRecord], NDArray[np.bool_], int, dict[str, object]]:
    """Fit and forecast one CONFIRM configuration. The target stays out of the window."""
    assert_confirm_schedule(schedule)
    records: list[RollingForecastRecord] = []
    repaired_flags: list[bool] = []
    actions: list[str] = []
    min_eigs: list[float] = []
    pd_flags: list[bool] = []
    fit_failures = 0
    for step in schedule.steps:
        window = [stamp.date() for stamp in step.estimation_dates(schedule.calendar)]
        if step.target.date() in window or any(day > step.origin.date() for day in window):
            raise ValueError(f"{step.target.date().isoformat()} window reads the target or a later day")
        local = _local_step(step, view.calendar)
        payload = build_origin_payload(
            step=local,
            calendar=view.calendar,
            daily_returns=view.daily_returns,
            realized_covariances=view.rcov,
            realized_quarticity=view.rq,
            semivariance_positive=semivariance_positive,
            semivariance_negative=semivariance_negative,
            feature_panel=feature_panel,
            proxy_panel=proxy_panel,
        )
        try:
            action = _fit_at_origin(model, payload) if step.refit else _advance_at_origin(model, payload)
            forecast = model.forecast()
        except Exception as exc:  # noqa: BLE001
            fit_failures += 1
            raise RuntimeError(
                f"model execution failed at origin {step.origin.date().isoformat()}: {exc}"
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
        if on_origin is not None:
            on_origin(step.origin.date(), bool(step.refit), model)
    extras = {
        "actions": tuple(actions),
        "min_eigenvalues": np.array(min_eigs, dtype=float),
        "positive_definite": np.array(pd_flags, dtype=bool),
    }
    return records, np.array(repaired_flags, dtype=bool), fit_failures, extras
