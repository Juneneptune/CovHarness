"""NBBO quote cleaning adapted from Barndorff-Nielsen et al. (2011), Section 5.1.

The sequence is P1, P2, Q1, Q2, Q3, Q4, then midquote construction. Q4 is a
symmetric ex-post filter on historical measurement data. It is not a
forecasting feature and intentionally uses a centered neighborhood that
includes later observations on the same stock-day.

P3 in the paper retains a single exchange. That rule is not applied here
because the input is consolidated NBBO (best bid / best ask).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


REGULAR_OPEN = dt.time(9, 30)
REGULAR_CLOSE = dt.time(16, 0)


@dataclass(frozen=True)
class QuoteCleaningDiagnostics:
    """Row counts after each cleaning stage.

    Removal counts are relative to the previous stage. ``removal_fraction`` is
    the share of input rows absent after Q4.
    """

    n_input: int
    n_after_p1: int
    n_after_p2: int
    n_after_q1: int
    n_after_q2: int
    n_after_q3: int
    n_after_q4: int

    @property
    def n_removed_p1(self) -> int:
        return self.n_input - self.n_after_p1

    @property
    def n_removed_p2(self) -> int:
        return self.n_after_p1 - self.n_after_p2

    @property
    def n_removed_q1(self) -> int:
        return self.n_after_p2 - self.n_after_q1

    @property
    def n_removed_q2(self) -> int:
        return self.n_after_q1 - self.n_after_q2

    @property
    def n_removed_q3(self) -> int:
        return self.n_after_q2 - self.n_after_q3

    @property
    def n_removed_q4(self) -> int:
        return self.n_after_q3 - self.n_after_q4

    @property
    def removal_fraction(self) -> float:
        if self.n_input == 0:
            return 0.0
        return (self.n_input - self.n_after_q4) / self.n_input


@dataclass(frozen=True)
class CleanedNbboQuotes:
    """Cleaned quotes plus stage diagnostics.

    ``quotes`` columns are ``timestamp``, ``asset``, ``date``, ``best_bid``,
    ``best_ask``, ``spread``, and ``midquote``. Rows are sorted by asset, then
    timestamp.
    """

    quotes: pd.DataFrame
    diagnostics: QuoteCleaningDiagnostics


def apply_q4_keep_mask(
    midquotes: np.ndarray,
    *,
    half_window: int = 25,
    threshold: float = 10.0,
    keep_incomplete_edges: bool = True,
) -> np.ndarray:
    """Q4 keep mask for one stock-day series of midquotes.

    For index ``i`` with a complete window, the neighborhood ``N_i`` is the
    ``half_window`` observations before ``i`` and the ``half_window`` after
    ``i``, excluding ``i`` itself (50 neighbors when ``half_window=25``).

        local_median_i = median(N_i)
        mean_abs_dev_i = mean(|N_i - local_median_i|)

    Drop ``i`` if ``|mid_i - local_median_i| > threshold * mean_abs_dev_i``.
    If ``mean_abs_dev_i == 0``, keep ``i`` only when ``mid_i`` equals the
    local median.

    Incomplete centered windows (first and last ``half_window`` rows, and any
    series shorter than ``2 * half_window + 1``) are kept when
    ``keep_incomplete_edges`` is True; otherwise they are dropped.
    """
    x = np.asarray(midquotes, dtype=float)
    n = x.size
    width = 2 * half_window + 1
    # Short series cannot form a complete centered neighborhood.
    if n == 0:
        return np.array([], dtype=bool)
    if n < width:
        return np.ones(n, dtype=bool) if keep_incomplete_edges else np.zeros(n, dtype=bool)

    # 50 neighbors per interior quote (25 before, 25 after, exclude the center).
    windows = sliding_window_view(x, width)
    neighbors = np.concatenate(
        [windows[:, :half_window], windows[:, half_window + 1 :]],
        axis=1,
    )
    center = windows[:, half_window]
    # Local median of neighbors, then mean abs deviation from that median.
    local_median = np.median(neighbors, axis=1)
    mean_abs_dev = np.mean(np.abs(neighbors - local_median[:, None]), axis=1)
    deviation = np.abs(center - local_median)
    # Flag interior outliers. Zero dispersion keeps only an identical center.
    interior_bad = np.where(
        mean_abs_dev == 0.0,
        deviation != 0.0,
        deviation > threshold * mean_abs_dev,
    )

    # Keep or drop edges by convention. Overwrite only the complete-window interior.
    keep = np.ones(n, dtype=bool) if keep_incomplete_edges else np.zeros(n, dtype=bool)
    keep[half_window : n - half_window] = ~interior_bad
    return keep


def clean_nbbo_quotes(
    data: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    asset_col: str = "asset",
    bid_col: str = "best_bid",
    ask_col: str = "best_ask",
    date_col: str | None = None,
    session_open: dt.time = REGULAR_OPEN,
    session_close: dt.time = REGULAR_CLOSE,
    q3_multiple: float = 10.0,
    q4_half_window: int = 25,
    q4_threshold: float = 10.0,
    keep_incomplete_q4_edges: bool = True,
) -> CleanedNbboQuotes:
    """Clean consolidated NBBO quotes (P1, P2, Q1–Q4), then form midquotes.

    Parameters
    ----------
    data : DataFrame
        Long-form quotes with timestamp, asset, best bid, and best ask.
    session_open, session_close : datetime.time
        Regular-session window used by P1, inclusive. Timestamps are treated
        as exchange-local clock times supplied by the caller.

    Returns
    -------
    CleanedNbboQuotes
        Filtered quotes with midquote ``(bid + ask) / 2`` and per-stage counts.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    # Require timestamp, asset, bid, and ask.
    required = (timestamp_col, asset_col, bid_col, ask_col)
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"data missing required columns: {missing}")

    # Copy quote fields, parse types, and attach a calendar date.
    frame = data.loc[:, list(required)].copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="raise")
    frame[bid_col] = pd.to_numeric(frame[bid_col], errors="raise")
    frame[ask_col] = pd.to_numeric(frame[ask_col], errors="raise")
    if date_col is not None:
        if date_col not in data.columns:
            raise ValueError(f"data missing date column {date_col}")
        frame["date"] = pd.to_datetime(data[date_col], errors="raise").dt.normalize()
    else:
        frame["date"] = frame[timestamp_col].dt.normalize()

    n_input = len(frame)

    # P1. Keep regular hours only, 09:30-16:00 inclusive.
    clock = frame[timestamp_col].dt.time
    frame = frame.loc[(clock >= session_open) & (clock <= session_close)].copy()
    n_after_p1 = len(frame)

    # P2. Keep strictly positive finite bid and ask.
    bid = frame[bid_col].to_numpy(dtype=float, na_value=np.nan)
    ask = frame[ask_col].to_numpy(dtype=float, na_value=np.nan)
    p2_ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask > 0.0)
    frame = frame.loc[p2_ok].copy()
    n_after_p2 = len(frame)

    # Empty after P1/P2. Later stages contribute zero rows.
    if frame.empty:
        quotes = _empty_quotes()
        diagnostics = QuoteCleaningDiagnostics(
            n_input=n_input,
            n_after_p1=n_after_p1,
            n_after_p2=n_after_p2,
            n_after_q1=0,
            n_after_q2=0,
            n_after_q3=0,
            n_after_q4=0,
        )
        return CleanedNbboQuotes(quotes=quotes, diagnostics=diagnostics)

    # Q1. Collapse same stock-day timestamp to median bid and median ask.
    grouped = (
        frame.groupby(["date", asset_col, timestamp_col], sort=True, as_index=False)
        .agg(best_bid=(bid_col, "median"), best_ask=(ask_col, "median"))
        .rename(columns={asset_col: "asset", timestamp_col: "timestamp"})
    )
    n_after_q1 = len(grouped)

    # Q2. Drop crossed quotes. Keep locked (zero) spreads.
    grouped["spread"] = grouped["best_ask"] - grouped["best_bid"]
    grouped = grouped.loc[grouped["spread"] >= 0.0].copy()
    n_after_q2 = len(grouped)

    # Empty after Q2. Later stages contribute zero rows.
    if grouped.empty:
        quotes = _empty_quotes()
        diagnostics = QuoteCleaningDiagnostics(
            n_input=n_input,
            n_after_p1=n_after_p1,
            n_after_p2=n_after_p2,
            n_after_q1=n_after_q1,
            n_after_q2=n_after_q2,
            n_after_q3=0,
            n_after_q4=0,
        )
        return CleanedNbboQuotes(quotes=quotes, diagnostics=diagnostics)

    # Q3. Drop spreads larger than 10 times the stock-day median.
    median_spread = grouped.groupby(["date", "asset"], sort=False)["spread"].transform(
        "median"
    )
    grouped = grouped.loc[grouped["spread"] <= q3_multiple * median_spread].copy()
    n_after_q3 = len(grouped)

    # Midquote after the spread filters.
    grouped["midquote"] = (grouped["best_bid"] + grouped["best_ask"]) / 2.0

    # Q4. Filter each stock-day with the centered 50-neighbor midquote rule.
    pieces: list[pd.DataFrame] = []
    for (_, _), stock_day in grouped.groupby(["date", "asset"], sort=False):
        stock_day = stock_day.sort_values("timestamp")
        keep = apply_q4_keep_mask(
            stock_day["midquote"].to_numpy(),
            half_window=q4_half_window,
            threshold=q4_threshold,
            keep_incomplete_edges=keep_incomplete_q4_edges,
        )
        pieces.append(stock_day.loc[keep])

    # Concatenate stock-days and sort by asset then time.
    if pieces:
        cleaned = (
            pd.concat(pieces, ignore_index=True)
            .sort_values(["asset", "timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        cleaned = _empty_quotes()
    n_after_q4 = len(cleaned)

    # Pack remaining-row counts after each stage.
    diagnostics = QuoteCleaningDiagnostics(
        n_input=n_input,
        n_after_p1=n_after_p1,
        n_after_p2=n_after_p2,
        n_after_q1=n_after_q1,
        n_after_q2=n_after_q2,
        n_after_q3=n_after_q3,
        n_after_q4=n_after_q4,
    )
    return CleanedNbboQuotes(quotes=cleaned, diagnostics=diagnostics)


def cleaned_quotes_as_ticks(cleaned: pd.DataFrame | CleanedNbboQuotes) -> pd.DataFrame:
    """Long-form ``timestamp``, ``asset``, ``price`` midquotes for previous-tick sync."""
    frame = cleaned.quotes if isinstance(cleaned, CleanedNbboQuotes) else cleaned
    # Rename midquote to price for previous_tick_sync.
    ticks = frame.loc[:, ["timestamp", "asset", "midquote"]].rename(
        columns={"midquote": "price"}
    )
    return ticks.reset_index(drop=True)


def _empty_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "asset",
            "timestamp",
            "best_bid",
            "best_ask",
            "spread",
            "midquote",
        ]
    )
