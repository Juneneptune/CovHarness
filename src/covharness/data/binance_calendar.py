"""Frozen segmented UTC calendar for the Binance open-data benchmark.

This calendar is branch-specific. It does not modify ``TemporalProtocol``.
Packed complete-day lags across 2023-03-24 are rejected. Recursions do not
cross the halt. A second 250-day HISTORY burn-in starts on 2023-03-25.
"""

from __future__ import annotations

import datetime as dt
from typing import Mapping

# Frozen first-universe order. Never infer from filesystem or dict iteration.
BINANCE_ASSETS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "LTCUSDT",
    "ADAUSDT",
)

HALT_DATE = dt.date(2023, 3, 24)
ANCHOR_DATE = dt.date(2021, 11, 8)
SAMPLE_END = dt.date(2026, 8, 25)
UNUSED_AFTER = dt.date(2026, 8, 26)

# Inclusive UTC date bounds for each named segment.
SEGMENT_BOUNDS: dict[str, tuple[dt.date, dt.date]] = {
    "HISTORY_A": (dt.date(2021, 11, 9), dt.date(2022, 7, 16)),
    "VALIDATION": (dt.date(2022, 7, 17), dt.date(2023, 3, 23)),
    "HISTORY_B": (dt.date(2023, 3, 25), dt.date(2023, 11, 29)),
    "SCREEN": (dt.date(2023, 11, 30), dt.date(2025, 4, 12)),
    "CONFIRM": (dt.date(2025, 4, 13), dt.date(2026, 8, 25)),
}

EXPECTED_SEGMENT_COUNTS: dict[str, int] = {
    "HISTORY_A": 250,
    "VALIDATION": 250,
    "HISTORY_B": 250,
    "SCREEN": 500,
    "CONFIRM": 500,
}

EXPECTED_PRODUCTION_DATES = 1750

# Five-minute interval endpoints as minute-of-day. 00:04, 00:09, ..., 23:59.
FIVE_MINUTE_END_MINUTES: tuple[int, ...] = tuple(range(4, 1440, 5))


class InvalidBinanceCalendarError(ValueError):
    """A frozen Binance calendar identity failed an exact-count check."""


def utc_dates_inclusive(start: dt.date, end: dt.date) -> tuple[dt.date, ...]:
    """Return every UTC calendar date in ``[start, end]``."""
    if end < start:
        raise InvalidBinanceCalendarError(
            f"end {end.isoformat()} precedes start {start.isoformat()}"
        )
    n = (end - start).days + 1
    return tuple(start + dt.timedelta(days=i) for i in range(n))


def verify_frozen_segment_counts() -> dict[str, int]:
    """Check advertised segment lengths against actual UTC date counts."""
    if len(FIVE_MINUTE_END_MINUTES) != 288:
        raise InvalidBinanceCalendarError(
            "five-minute endpoint count must be 288; "
            f"got {len(FIVE_MINUTE_END_MINUTES)}"
        )
    if FIVE_MINUTE_END_MINUTES[0] != 4 or FIVE_MINUTE_END_MINUTES[-1] != 1439:
        raise InvalidBinanceCalendarError(
            "five-minute endpoints must run from 00:04 through 23:59"
        )
    counts: dict[str, int] = {}
    for name, (start, end) in SEGMENT_BOUNDS.items():
        n = len(utc_dates_inclusive(start, end))
        expected = EXPECTED_SEGMENT_COUNTS[name]
        if n != expected:
            raise InvalidBinanceCalendarError(
                f"{name} has {n} UTC dates; frozen count is {expected}"
            )
        counts[name] = n
    total = sum(counts.values())
    if total != EXPECTED_PRODUCTION_DATES:
        raise InvalidBinanceCalendarError(
            f"production date total is {total}; frozen total is "
            f"{EXPECTED_PRODUCTION_DATES}"
        )
    # Halt sits strictly between VALIDATION and HISTORY_B.
    if HALT_DATE != dt.date(2023, 3, 24):
        raise InvalidBinanceCalendarError("halt date identity changed")
    if SEGMENT_BOUNDS["VALIDATION"][1] + dt.timedelta(days=1) != HALT_DATE:
        raise InvalidBinanceCalendarError("VALIDATION does not end the day before the halt")
    if HALT_DATE + dt.timedelta(days=1) != SEGMENT_BOUNDS["HISTORY_B"][0]:
        raise InvalidBinanceCalendarError("HISTORY_B does not start the day after the halt")
    if SEGMENT_BOUNDS["HISTORY_A"][1] + dt.timedelta(days=1) != SEGMENT_BOUNDS["VALIDATION"][0]:
        raise InvalidBinanceCalendarError("VALIDATION is not adjacent to HISTORY_A")
    if SEGMENT_BOUNDS["HISTORY_B"][1] + dt.timedelta(days=1) != SEGMENT_BOUNDS["SCREEN"][0]:
        raise InvalidBinanceCalendarError("SCREEN is not adjacent to HISTORY_B")
    if SEGMENT_BOUNDS["SCREEN"][1] + dt.timedelta(days=1) != SEGMENT_BOUNDS["CONFIRM"][0]:
        raise InvalidBinanceCalendarError("CONFIRM is not adjacent to SCREEN")
    leftover = utc_dates_inclusive(UNUSED_AFTER, dt.date(2026, 8, 31))
    if leftover[0] <= SAMPLE_END:
        raise InvalidBinanceCalendarError("unused tail overlaps the sample")
    return counts


def production_dates() -> tuple[dt.date, ...]:
    """Retained statistical dates in chronological order. The halt is omitted."""
    verify_frozen_segment_counts()
    dates: list[dt.date] = []
    for name in ("HISTORY_A", "VALIDATION", "HISTORY_B", "SCREEN", "CONFIRM"):
        start, end = SEGMENT_BOUNDS[name]
        dates.extend(utc_dates_inclusive(start, end))
    if HALT_DATE in dates:
        raise InvalidBinanceCalendarError("halt date leaked into production dates")
    if any(d > SAMPLE_END for d in dates):
        raise InvalidBinanceCalendarError("a production date falls after SAMPLE_END")
    if len(dates) != EXPECTED_PRODUCTION_DATES:
        raise InvalidBinanceCalendarError(
            f"assembled production length {len(dates)} != {EXPECTED_PRODUCTION_DATES}"
        )
    return tuple(dates)


def segment_label(day: dt.date) -> str:
    """Return the frozen segment name for a retained date."""
    if day == HALT_DATE:
        raise InvalidBinanceCalendarError(
            "2023-03-24 is the excluded halt, not a production segment"
        )
    for name, (start, end) in SEGMENT_BOUNDS.items():
        if start <= day <= end:
            return name
    raise InvalidBinanceCalendarError(
        f"{day.isoformat()} is outside the frozen Binance sample"
    )


def segment_dates() -> Mapping[str, tuple[dt.date, ...]]:
    """Map each segment name to its inclusive UTC dates."""
    verify_frozen_segment_counts()
    return {
        name: utc_dates_inclusive(start, end)
        for name, (start, end) in SEGMENT_BOUNDS.items()
    }


def previous_calendar_date(day: dt.date) -> dt.date:
    """Return the previous UTC calendar date, including the halt when required."""
    return day - dt.timedelta(days=1)
