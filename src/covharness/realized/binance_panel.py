"""Binance open-data five-minute RCov, RQ, and aligned daily returns.

Implements the frozen measurement in docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md.
Does not call TAQ quote or previous-tick classes. Does not fit models.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from covharness.data.binance_calendar import (
    ANCHOR_DATE,
    BINANCE_ASSETS,
    EXPECTED_PRODUCTION_DATES,
    EXPECTED_SEGMENT_COUNTS,
    FIVE_MINUTE_END_MINUTES,
    HALT_DATE,
    SAMPLE_END,
    SEGMENT_BOUNDS,
    dates_through_screen,
    interval_end_minutes,
    production_dates,
    segment_label,
    verify_frozen_segment_counts,
)
from covharness.data.binance_klines import (
    load_monthly_closes,
    monthly_zip_paths,
    verify_checksum,
)
from covharness.diagnostics.epps import matrix_eigen_diagnostics
from covharness.losses.contracts import PSD_ATOL, SYMMETRY_ATOL
from covharness.realized.rcov import realized_covariance

N_ASSETS = len(BINANCE_ASSETS)
N_INTERVALS = 288
RQ_SCALE = N_INTERVALS / 3.0
MINUTE_23_59 = 1439
SIGNATURE_INTERVALS: tuple[int, ...] = (1, 5, 15, 30)


class InvalidBinanceMeasurementError(ValueError):
    """A required Binance close or derived matrix failed the frozen contract."""


def grid_log_returns(
    endpoint_closes: NDArray[np.floating],
    prior_anchor_close: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Log returns from one anchor close and ``M`` later grid closes.

    ``endpoint_closes`` has shape ``(M, N)``. ``prior_anchor_close`` has shape
    ``(N,)``. Row 0 is ``log(endpoint_0) - log(anchor)``. Missing endpoints
    are rejected rather than filled. This is the only log-difference
    implementation used by the five-minute production path and by other
    intervals.
    """
    endpoints = np.asarray(endpoint_closes, dtype=float)
    prior = np.asarray(prior_anchor_close, dtype=float)
    if endpoints.ndim != 2 or endpoints.shape[0] < 1:
        raise InvalidBinanceMeasurementError(
            f"endpoint_closes must have shape (M, N) with M>=1; got {endpoints.shape}"
        )
    n_assets = int(endpoints.shape[1])
    if prior.shape != (n_assets,):
        raise InvalidBinanceMeasurementError(
            f"prior anchor must have shape ({n_assets},); got {prior.shape}"
        )
    # Reject missing or non-positive required closes. Do not forward-fill.
    stacked = np.vstack([prior, endpoints])
    if not np.isfinite(stacked).all():
        raise InvalidBinanceMeasurementError("required grid closes must be finite")
    if np.any(stacked <= 0.0):
        raise InvalidBinanceMeasurementError("required grid closes must be strictly positive")
    # Log-price differences on the supplied grid.
    return np.diff(np.log(stacked), axis=0)


def five_minute_log_returns(
    endpoint_closes: NDArray[np.floating],
    prior_close_2359: NDArray[np.floating],
) -> NDArray[np.floating]:
    """288 log returns from prior 23:59 and the 00:04,...,23:59 endpoint closes.

    ``endpoint_closes`` has shape ``(288, N)``. ``prior_close_2359`` has shape
    ``(N,)``. The first return is ``log(p_{t,00:04}) - log(p_{t-1,23:59})``.
    Missing endpoints are rejected rather than filled.
    """
    endpoints = np.asarray(endpoint_closes, dtype=float)
    if endpoints.ndim != 2 or endpoints.shape[0] != N_INTERVALS:
        raise InvalidBinanceMeasurementError(
            f"endpoint_closes must have shape ({N_INTERVALS}, N); got {endpoints.shape}"
        )
    # Delegate to the shared grid implementation. Do not reimplement the diff.
    return grid_log_returns(endpoints, prior_close_2359)


def realized_quarticity(five_minute_returns: NDArray[np.floating]) -> NDArray[np.floating]:
    """Per-asset RQ = (288/3) sum_j r_{i,j}^4 from the same 288 returns."""
    returns = np.asarray(five_minute_returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] != N_INTERVALS:
        raise InvalidBinanceMeasurementError(
            f"five_minute_returns must have shape ({N_INTERVALS}, N); got {returns.shape}"
        )
    if not np.isfinite(returns).all():
        raise InvalidBinanceMeasurementError("five-minute returns must be finite")
    quarticity = RQ_SCALE * np.sum(returns**4, axis=0)
    if not np.isfinite(quarticity).all() or np.any(quarticity < 0.0):
        raise InvalidBinanceMeasurementError("realized quarticity must be finite and nonnegative")
    return quarticity


def daily_log_return(
    close_2359: NDArray[np.floating],
    prior_close_2359: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Close-to-close log return between consecutive 23:59 UTC prices."""
    today = np.asarray(close_2359, dtype=float)
    prior = np.asarray(prior_close_2359, dtype=float)
    if today.shape != prior.shape or today.ndim != 1:
        raise InvalidBinanceMeasurementError(
            f"daily closes must be 1-d and aligned; got {today.shape} and {prior.shape}"
        )
    stacked = np.vstack([prior, today])
    if not np.isfinite(stacked).all() or np.any(stacked <= 0.0):
        raise InvalidBinanceMeasurementError("daily 23:59 closes must be finite and strictly positive")
    return np.log(today) - np.log(prior)


def _minute_of_day(timestamps: pd.DatetimeIndex) -> NDArray[np.int64]:
    return (
        timestamps.hour.to_numpy(dtype=np.int64) * 60
        + timestamps.minute.to_numpy(dtype=np.int64)
    )


def _load_close_cube(
    cache_root: Path,
    *,
    start: dt.date,
    end: dt.date,
) -> tuple[NDArray[np.floating], tuple[dt.date, ...], list[str]]:
    """Fill ``(N, n_days, 1440)`` 1m closes in frozen asset order."""
    n_days = (end - start).days + 1
    cube = np.full((N_ASSETS, n_days, 1440), np.nan, dtype=np.float64)
    checksums: list[str] = []
    months: list[tuple[int, int]] = []
    cursor = dt.date(start.year, start.month, 1)
    last_month = dt.date(end.year, end.month, 1)
    while cursor <= last_month:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = dt.date(cursor.year + 1, 1, 1)
        else:
            cursor = dt.date(cursor.year, cursor.month + 1, 1)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)

    # Load each official monthly ZIP in frozen asset order.
    for asset_index, symbol in enumerate(BINANCE_ASSETS):
        for year, month in months:
            zip_path, checksum_path = monthly_zip_paths(cache_root, symbol, year, month)
            digest = verify_checksum(zip_path, checksum_path)
            checksums.append(f"{zip_path.name}:{digest}")
            series = load_monthly_closes(zip_path)
            index = series.index.tz_convert("UTC")
            on_grid = (index.second == 0) & (index.microsecond == 0)
            in_span = np.asarray(
                (index >= start_ts) & (index <= end_ts) & on_grid,
                dtype=bool,
            )
            if not bool(in_span.any()):
                continue
            selected = index[in_span]
            day_index = (selected.normalize() - start_ts).days.to_numpy(dtype=np.int64)
            minutes = _minute_of_day(selected)
            values = series.to_numpy(dtype=float)[in_span]
            valid = (day_index >= 0) & (day_index < n_days) & (minutes >= 0) & (minutes < 1440)
            cube[asset_index, day_index[valid], minutes[valid]] = values[valid]

    days = tuple(start + dt.timedelta(days=i) for i in range(n_days))
    return cube, days, checksums


def _day_index(day: dt.date, start: dt.date) -> int:
    return (day - start).days


def _extract_day_endpoints(
    cube: NDArray[np.floating],
    start: dt.date,
    day: dt.date,
    end_minutes: tuple[int, ...] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return prior 23:59 closes ``(N,)`` and grid endpoint closes ``(M, N)``.

    ``end_minutes`` defaults to the frozen five-minute tuple. The anchor is
    always the prior calendar day's 23:59 close. There is no forward fill.
    """
    minutes = FIVE_MINUTE_END_MINUTES if end_minutes is None else tuple(end_minutes)
    prior = day - dt.timedelta(days=1)
    if prior < start:
        raise InvalidBinanceMeasurementError(
            f"{day.isoformat()} has no prior calendar day in the loaded cube"
        )
    prior_i = _day_index(prior, start)
    day_i = _day_index(day, start)
    prior_close = cube[:, prior_i, MINUTE_23_59].copy()
    # Take the requested minute-of-day closes on the statistical day only.
    endpoints = cube[:, day_i, list(minutes)].T.copy()
    if not np.isfinite(prior_close).all() or np.any(prior_close <= 0.0):
        raise InvalidBinanceMeasurementError(
            f"missing or invalid 23:59 anchor on {prior.isoformat()} for {day.isoformat()}"
        )
    if not np.isfinite(endpoints).all() or np.any(endpoints <= 0.0):
        label = "five-minute" if end_minutes is None else "grid"
        raise InvalidBinanceMeasurementError(
            f"missing or invalid {label} endpoint on {day.isoformat()}"
        )
    return prior_close, endpoints


def validate_rcov_slice(matrix: NDArray[np.floating], day: dt.date) -> dict[str, float | int | bool | None]:
    """Reject an invalid production RCov. No repair is applied."""
    if matrix.shape != (N_ASSETS, N_ASSETS):
        raise InvalidBinanceMeasurementError(
            f"{day.isoformat()} RCov shape {matrix.shape} != {(N_ASSETS, N_ASSETS)}"
        )
    if not np.isfinite(matrix).all():
        raise InvalidBinanceMeasurementError(f"{day.isoformat()} RCov has a nonfinite entry")
    eigvals, rank, min_eig, psd, cond, symmetry_error, _m_over_n = matrix_eigen_diagnostics(
        matrix, atol=PSD_ATOL, n_returns=N_INTERVALS
    )
    if symmetry_error > SYMMETRY_ATOL:
        raise InvalidBinanceMeasurementError(
            f"{day.isoformat()} RCov is asymmetric; max |A-A.T|={symmetry_error}"
        )
    if not psd:
        raise InvalidBinanceMeasurementError(
            f"{day.isoformat()} RCov is not PSD; min eigenvalue={min_eig}"
        )
    if np.any(np.diag(matrix) <= 0.0):
        raise InvalidBinanceMeasurementError(f"{day.isoformat()} RCov has a nonpositive diagonal")
    return {
        "min_eig": min_eig,
        "max_eig": float(eigvals.max()),
        "rank": rank,
        "condition_number": cond,
        "symmetry_error": symmetry_error,
        "psd": psd,
    }


@dataclass(frozen=True)
class BinanceProductionPanel:
    """Aligned Binance daily measurements. Raw 1m bars are not stored."""

    dates: tuple[dt.date, ...]
    assets: tuple[str, ...]
    segments: tuple[str, ...]
    daily_returns: NDArray[np.floating]
    rcov_5min: NDArray[np.floating]
    rq_per_asset: NDArray[np.floating]
    rcov_diagnostics: tuple[dict[str, float | int | bool | None], ...]
    archive_checksums: tuple[str, ...]


def build_production_panel(cache_root: Path) -> BinanceProductionPanel:
    """Construct the 1750-date production panel from official cached 1m ZIPs."""
    verify_frozen_segment_counts()
    dates = production_dates()
    cube, cube_days, checksums = _load_close_cube(
        Path(cache_root), start=ANCHOR_DATE, end=SAMPLE_END
    )
    if cube_days[0] != ANCHOR_DATE or cube_days[-1] != SAMPLE_END:
        raise InvalidBinanceMeasurementError("close cube calendar does not match the frozen span")

    t = len(dates)
    daily = np.empty((t, N_ASSETS), dtype=np.float64)
    rcov = np.empty((t, N_ASSETS, N_ASSETS), dtype=np.float64)
    rq = np.empty((t, N_ASSETS), dtype=np.float64)
    segments: list[str] = []
    diagnostics: list[dict[str, float | int | bool | None]] = []

    # Measure each retained statistical day with the previous calendar 23:59 anchor.
    for time_index, day in enumerate(dates):
        if day == HALT_DATE:
            raise InvalidBinanceMeasurementError("halt date entered production construction")
        prior_close, endpoints = _extract_day_endpoints(cube, ANCHOR_DATE, day)
        returns = five_minute_log_returns(endpoints, prior_close)
        matrix = realized_covariance(returns)
        diag = validate_rcov_slice(matrix, day)
        daily[time_index] = daily_log_return(endpoints[-1], prior_close)
        rcov[time_index] = matrix
        rq[time_index] = realized_quarticity(returns)
        segments.append(segment_label(day))
        diagnostics.append(diag)

    if t != EXPECTED_PRODUCTION_DATES:
        raise InvalidBinanceMeasurementError(f"built T={t} != {EXPECTED_PRODUCTION_DATES}")
    if HALT_DATE in dates:
        raise InvalidBinanceMeasurementError("halt date is a production date")
    return BinanceProductionPanel(
        dates=dates,
        assets=BINANCE_ASSETS,
        segments=tuple(segments),
        daily_returns=daily,
        rcov_5min=rcov,
        rq_per_asset=rq,
        rcov_diagnostics=tuple(diagnostics),
        archive_checksums=tuple(checksums),
    )


def save_production_panel(panel: BinanceProductionPanel, path: Path) -> str:
    """Write the compact NPZ artifact. Raw minute bars are omitted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    date_strings = np.array([d.isoformat() for d in panel.dates], dtype="U10")
    segment_arr = np.array(panel.segments, dtype="U16")
    asset_arr = np.array(panel.assets, dtype="U16")
    np.savez_compressed(
        path,
        dates=date_strings,
        assets=asset_arr,
        segments=segment_arr,
        daily_returns=panel.daily_returns,
        rcov_5min=panel.rcov_5min,
        rq_per_asset=panel.rq_per_asset,
        n_intervals=np.int64(N_INTERVALS),
        halt_date=np.array(HALT_DATE.isoformat()),
        measurement="binance_spot_1m_close_288_five_minute",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_panel_sidecar(
    panel: BinanceProductionPanel,
    npz_path: Path,
    npz_sha256: str,
    sidecar_path: Path,
    summary: dict[str, object],
) -> str:
    """Write production metadata JSON. Raw minute bars are omitted."""
    payload = {
        "artifact": str(npz_path),
        "artifact_sha256": npz_sha256,
        "source": "https://data.binance.vision",
        "spec": "docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md",
        "calendar": "docs/BINANCE_OPEN_DATA_PANEL.md",
        "universe": list(panel.assets),
        "n_intervals": N_INTERVALS,
        "halt_date": HALT_DATE.isoformat(),
        "measurement": "binance_spot_1m_close_288_five_minute",
        "rq_scale": RQ_SCALE,
        "annualization": False,
        "repair": False,
        "forward_fill": False,
        "temporal_protocol": "binance_segmented_hard_break",
        "generic_temporal_protocol_modified": False,
        "archive_checksums": list(panel.archive_checksums),
        "summary": summary,
    }
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(sidecar_path.read_bytes()).hexdigest()


def panel_summary(panel: BinanceProductionPanel) -> dict[str, object]:
    """Descriptive measurement diagnostics. No forecast comparison."""
    rcov = panel.rcov_5min
    rq = panel.rq_per_asset
    daily = panel.daily_returns
    diags = panel.rcov_diagnostics
    min_eigs = np.array([d["min_eig"] for d in diags], dtype=float)
    ranks = np.array([d["rank"] for d in diags], dtype=int)
    conds = np.array(
        [d["condition_number"] if d["condition_number"] is not None else np.nan for d in diags],
        dtype=float,
    )
    counts = Counter(panel.segments)
    symmetry_fail = int(
        sum(1 for d in diags if float(d["symmetry_error"]) > SYMMETRY_ATOL)
    )
    psd_fail = int(sum(1 for d in diags if not bool(d["psd"])))
    return {
        "T": int(len(panel.dates)),
        "N": int(len(panel.assets)),
        "assets": list(panel.assets),
        "date_start": panel.dates[0].isoformat(),
        "date_end": panel.dates[-1].isoformat(),
        "segment_counts": {k: int(counts[k]) for k in EXPECTED_SEGMENT_COUNTS},
        "daily_returns_shape": list(daily.shape),
        "rcov_shape": list(rcov.shape),
        "rq_shape": list(rq.shape),
        "daily_nonfinite": int(np.size(daily) - np.isfinite(daily).sum()),
        "rcov_nonfinite": int(np.size(rcov) - np.isfinite(rcov).sum()),
        "rq_nonfinite": int(np.size(rq) - np.isfinite(rq).sum()),
        "symmetry_failures": symmetry_fail,
        "psd_failures": psd_fail,
        "rcov_min_eigenvalue": float(np.min(min_eigs)),
        "rcov_max_eigenvalue": float(max(float(d["max_eig"]) for d in diags)),
        "rcov_rank_counts": {str(k): int(v) for k, v in sorted(Counter(ranks.tolist()).items())},
        "condition_number_min": float(np.nanmin(conds)),
        "condition_number_median": float(np.nanmedian(conds)),
        "condition_number_max": float(np.nanmax(conds)),
        "rq_min": float(np.min(rq)),
        "rq_median": float(np.median(rq)),
        "rq_max": float(np.max(rq)),
        "daily_return_min": float(np.min(daily)),
        "daily_return_max": float(np.max(daily)),
        "halt_in_dates": HALT_DATE in panel.dates,
        "any_date_after_sample_end": any(d > SAMPLE_END for d in panel.dates),
        "history_b_after_validation": panel.dates[counts["HISTORY_A"] + counts["VALIDATION"]]
        == dt.date(2023, 3, 25),
        "screen_after_history_b": min(
            d for d, s in zip(panel.dates, panel.segments) if s == "SCREEN"
        )
        > max(d for d, s in zip(panel.dates, panel.segments) if s == "HISTORY_B"),
        "confirm_after_screen": min(
            d for d, s in zip(panel.dates, panel.segments) if s == "CONFIRM"
        )
        > max(d for d, s in zip(panel.dates, panel.segments) if s == "SCREEN"),
        "n_archive_checksums": len(panel.archive_checksums),
    }


@dataclass(frozen=True)
class IntervalMeasurement:
    """One pre-CONFIRM sampling interval. Raw minute bars are not stored.

    ``returns`` has shape ``(T, M, N)``. ``rcov`` has shape ``(T, N, N)``.
    ``daily_returns`` has shape ``(T, N)`` and uses the UTC 23:59 close on
    every interval, so it does not depend on the grid.
    """

    dates: tuple[dt.date, ...]
    assets: tuple[str, ...]
    interval_minutes: int
    n_intervals: int
    returns: NDArray[np.floating]
    rcov: NDArray[np.floating]
    daily_returns: NDArray[np.floating]


def measure_day_on_interval(
    cube: NDArray[np.floating],
    start: dt.date,
    day: dt.date,
    interval_minutes: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """One statistical day on a uniform grid.

    Returns ``(M, N)`` log returns, ``(N, N)`` realized covariance, and the
    ``(N,)`` UTC daily log return. ``M`` is ``1440 / interval_minutes``.
    """
    minutes = interval_end_minutes(interval_minutes)
    expected = 1440 // int(interval_minutes)
    if len(minutes) != expected:
        raise InvalidBinanceMeasurementError(
            f"interval {interval_minutes} produced {len(minutes)} endpoints; expected {expected}"
        )
    prior_close, endpoints = _extract_day_endpoints(
        cube, start, day, end_minutes=minutes
    )
    returns = grid_log_returns(endpoints, prior_close)
    if returns.shape != (expected, prior_close.shape[0]):
        raise InvalidBinanceMeasurementError(
            f"{day.isoformat()} return shape {returns.shape} != {(expected, prior_close.shape[0])}"
        )
    matrix = realized_covariance(returns)
    daily = daily_log_return(endpoints[-1], prior_close)
    return returns, matrix, daily


def build_preconfirm_interval_measurements(
    cache_root: Path,
    intervals: tuple[int, ...] = SIGNATURE_INTERVALS,
) -> tuple[IntervalMeasurement, ...]:
    """RCov on HISTORY_A through SCREEN. CONFIRM months are not the end date.

    The cube ends on the last SCREEN date. The halt remains in the cube only
    as the 23:59 anchor for the next statistical day. It is not a row.
    """
    dates = dates_through_screen()
    end = dates[-1]
    if end != SEGMENT_BOUNDS["SCREEN"][1]:
        raise InvalidBinanceMeasurementError("pre-CONFIRM end is not the last SCREEN date")
    if any(day > end or day == HALT_DATE for day in dates):
        raise InvalidBinanceMeasurementError("pre-CONFIRM dates include the halt or a later day")
    cube, cube_days, _checksums = _load_close_cube(
        Path(cache_root), start=ANCHOR_DATE, end=end
    )
    if cube_days[0] != ANCHOR_DATE or cube_days[-1] != end:
        raise InvalidBinanceMeasurementError("close cube does not stop on the last SCREEN date")

    # Allocate every requested grid before the day loop so a bad interval fails first.
    grids = tuple(interval_end_minutes(interval) for interval in intervals)
    t = len(dates)
    daily = np.empty((t, N_ASSETS), dtype=np.float64)
    returns_out = [
        np.empty((t, len(minutes), N_ASSETS), dtype=np.float64) for minutes in grids
    ]
    rcov_out = [
        np.empty((t, N_ASSETS, N_ASSETS), dtype=np.float64) for _minutes in grids
    ]

    # Measure each retained day on every grid from the same minute cube.
    for time_index, day in enumerate(dates):
        if day == HALT_DATE or day > end:
            raise InvalidBinanceMeasurementError(
                f"{day.isoformat()} is outside the pre-CONFIRM measurement set"
            )
        for slot, interval in enumerate(intervals):
            returns, matrix, day_return = measure_day_on_interval(
                cube, ANCHOR_DATE, day, interval
            )
            returns_out[slot][time_index] = returns
            rcov_out[slot][time_index] = matrix
            if slot == 0:
                daily[time_index] = day_return
            elif not np.array_equal(daily[time_index], day_return):
                raise InvalidBinanceMeasurementError(
                    f"{day.isoformat()} daily return depends on the sampling interval"
                )

    if HALT_DATE in dates:
        raise InvalidBinanceMeasurementError("halt date is a measured row")
    return tuple(
        IntervalMeasurement(
            dates=dates,
            assets=BINANCE_ASSETS,
            interval_minutes=int(interval),
            n_intervals=1440 // int(interval),
            returns=returns_out[slot],
            rcov=rcov_out[slot],
            daily_returns=daily.copy(),
        )
        for slot, interval in enumerate(intervals)
    )


def production_rcov_through_screen(
    panel_path: Path,
    dates: tuple[dt.date, ...],
) -> NDArray[np.floating]:
    """Copy production RCov rows through the last requested date, then stop.

    Later rows, including CONFIRM, are not copied.
    """
    blob = np.load(panel_path, mmap_mode="r")
    stored_dates = np.asarray(blob["dates"])
    end_text = dates[-1].isoformat()
    keep: list[int] = []
    # Stop at the first date after the requested end. Do not index past it.
    for index, label in enumerate(stored_dates):
        text = str(label)
        if text > end_text:
            break
        keep.append(index)
    if len(keep) != len(dates):
        raise InvalidBinanceMeasurementError(
            f"production panel kept {len(keep)} dates; measurement has {len(dates)}"
        )
    selected = tuple(str(stored_dates[index]) for index in keep)
    expected = tuple(day.isoformat() for day in dates)
    if selected != expected:
        raise InvalidBinanceMeasurementError("production panel dates do not match the measurement")
    if any(text > end_text for text in selected):
        raise InvalidBinanceMeasurementError("a copied production date is after the requested end")
    return np.asarray(blob["rcov_5min"][np.asarray(keep, dtype=int)], dtype=float)
