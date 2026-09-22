"""Official Binance spot 1-minute kline close loader.

This path does not call quote, midpoint, NBBO, or previous-tick TAQ classes.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from covharness.data.binance_calendar import BINANCE_ASSETS


class InvalidBinanceKlineError(ValueError):
    """An official kline archive failed a documented measurement contract."""


def infer_timestamp_unit(open_times: np.ndarray) -> str:
    """Detect millisecond versus microsecond open times from the integer scale."""
    values = np.asarray(open_times, dtype=np.float64)
    if values.size == 0:
        raise InvalidBinanceKlineError("cannot infer timestamp unit from an empty file")
    median = float(np.median(values))
    if 1e11 <= median < 1e14:
        return "milliseconds"
    if 1e14 <= median < 1e17:
        return "microseconds"
    raise InvalidBinanceKlineError(f"unrecognized timestamp scale median={median}")


def monthly_zip_paths(cache_root: Path, symbol: str, year: int, month: int) -> tuple[Path, Path]:
    """Return the official monthly ZIP and CHECKSUM paths in the local cache."""
    if symbol not in BINANCE_ASSETS:
        raise InvalidBinanceKlineError(f"{symbol} is not in the frozen Binance universe")
    stem = f"{symbol}-1m-{year:04d}-{month:02d}.zip"
    directory = Path(cache_root) / symbol
    return directory / stem, directory / f"{stem}.CHECKSUM"


def verify_checksum(zip_path: Path, checksum_path: Path) -> str:
    """SHA-256 the ZIP against the official CHECKSUM file. Fail loudly."""
    if not zip_path.is_file():
        raise InvalidBinanceKlineError(f"missing monthly archive {zip_path}")
    if not checksum_path.is_file():
        raise InvalidBinanceKlineError(f"missing checksum file {checksum_path}")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
    if digest != expected:
        raise InvalidBinanceKlineError(
            f"checksum mismatch {zip_path.name}: local={digest} expected={expected}"
        )
    return digest


def load_monthly_closes(zip_path: Path) -> pd.Series:
    """Read kline closes from one official monthly ZIP. CSV is not written to disk."""
    with zipfile.ZipFile(zip_path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as handle:
            frame = pd.read_csv(
                handle,
                header=None,
                usecols=[0, 4],
                names=["open_time", "close"],
            )
    unit = infer_timestamp_unit(frame["open_time"].to_numpy())
    # Convert detected units to UTC open times. Do not assume a global scale.
    timestamps = pd.to_datetime(
        frame["open_time"],
        unit="ms" if unit == "milliseconds" else "us",
        utc=True,
    )
    closes = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=float)
    series = pd.Series(closes, index=timestamps, name="close")
    if series.index.has_duplicates:
        raise InvalidBinanceKlineError(f"duplicate open times in {zip_path.name}")
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
        if not series.index.is_monotonic_increasing:
            raise InvalidBinanceKlineError(f"non-monotone open times in {zip_path.name}")
    return series
