"""CONFIRM calendar checks. The unlock stays inside the orchestrator."""

from __future__ import annotations

import datetime as dt

from covharness.data.binance_calendar import HALT_DATE
from covharness.protocol.binance_confirm import (
    CONFIRM_END,
    CONFIRM_START,
    FIRST_ORIGIN,
    N_TARGETS,
    WINDOW,
    confirm_schedule,
)


def test_confirm_schedule_on_the_real_calendar() -> None:
    schedule = confirm_schedule()
    steps = tuple(schedule.steps)
    assert len(steps) == N_TARGETS
    assert steps[0].origin.date() == FIRST_ORIGIN
    assert steps[0].target.date() == CONFIRM_START
    assert steps[-1].target.date() == CONFIRM_END
    assert steps[0].refit is True
    for index, step in enumerate(steps):
        origin = step.origin.date()
        target = step.target.date()
        assert (target - origin).days == 1
        window = [stamp.date() for stamp in step.estimation_dates(schedule.calendar)]
        assert len(window) == WINDOW
        assert target not in window
        assert max(window) == origin
        assert HALT_DATE not in window
        assert bool(step.refit) is (index % 21 == 0)
    assert steps[21].refit is True
    assert steps[1].refit is False
    assert dt.date(2025, 4, 13) not in [
        stamp.date() for stamp in steps[0].estimation_dates(schedule.calendar)
    ]
