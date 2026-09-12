"""Forensic check of one-minute JPM variance on the Epps scan.

Reuses the cleaned single-exchange midquote cache and the existing previous-tick
Epps grid. Q1-Q4, the P3 adapter, and the Epps estimator are not modified.
JPM is not removed. Large returns are reported, not deleted.

The question is whether any |r|>2% one-minute JPM returns remain after the
ordinary-share identity is applied. Results are diagnostic. They do not
redefine the official five-stock universe or add a return filter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from covharness.data.exchange_quotes import clean_single_exchange_quotes
from covharness.data.returns import synchronized_log_returns
from covharness.data.synchronization import previous_tick_sync
from covharness.data.venues import (
    TAQ_ORDINARY_SHARE_SUFFIX,
    is_ordinary_taq_suffix,
    listing_venue_from_crsp_exchcd,
)
from covharness.diagnostics.epps import (
    DEFAULT_FREQUENCIES_MINUTES,
    calendar_session_grid,
    drop_incomplete_open,
    epps_across_frequencies,
    summarize_off_diagonal,
)
from experiments.taq_five_stock_epps import CACHE_TICKS, load_or_build_cleaned_stream
from experiments.taq_five_stock_pilot import SAMPLE_DATE, SYMBOLS
from experiments.taq_five_stock_single_exchange_pilot import fetch_cqm_selected_venues

# Same 23 cutoff used in the earlier NBBO displacement forensics. Diagnostic only.
DISPLACED_MID_CUTOFF = 23.0
NEIGHBOR_QUOTES = 10
RETURN_THRESHOLDS = (0.02, 0.05, 0.10)
NON_JPM_PAIRS = (
    "IBM-AAPL",
    "IBM-MSFT",
    "IBM-XOM",
    "AAPL-MSFT",
    "AAPL-XOM",
    "MSFT-XOM",
)
JPM_PAIRS = ("IBM-JPM", "AAPL-JPM", "MSFT-JPM", "JPM-XOM")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "taq_jpm_one_minute_forensics_20090213.json"


def previous_tick_quote(
    quotes: pd.DataFrame,
    grid_time: pd.Timestamp,
) -> pd.Series | None:
    """Latest cleaned quote with timestamp at or before the grid time."""
    prior = quotes.loc[quotes["timestamp"] <= grid_time]
    if prior.empty:
        return None
    return prior.iloc[-1]


def quote_neighborhood(
    quotes: pd.DataFrame,
    selected: pd.Series,
    n_neighbors: int = NEIGHBOR_QUOTES,
) -> dict[str, object]:
    """Cleaned quotes immediately before and after the selected previous-tick row."""
    ordered = quotes.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    # Locate the selected quote by timestamp, then take a symmetric message window.
    match = ordered.index[ordered["timestamp"] == selected["timestamp"]]
    if len(match) == 0:
        loc = int((ordered["timestamp"] <= selected["timestamp"]).sum() - 1)
    else:
        loc = int(match[-1])
    start = max(0, loc - n_neighbors)
    end = min(len(ordered), loc + n_neighbors + 1)
    window = ordered.iloc[start:end]
    mids = window["midquote"].to_numpy(dtype=float)
    return {
        "selected_index_in_cleaned_stream": loc,
        "n_cleaned_jpm_quotes": int(len(ordered)),
        "window_start": window["timestamp"].iloc[0].isoformat(),
        "window_end": window["timestamp"].iloc[-1].isoformat(),
        "quotes": [
            {
                "timestamp": row["timestamp"].isoformat(),
                "best_bid": float(row["best_bid"]),
                "best_ask": float(row["best_ask"]),
                "spread": float(row["spread"]),
                "midquote": float(row["midquote"]),
                "is_selected": bool(row["timestamp"] == selected["timestamp"]),
                "in_displaced_band": bool(float(row["midquote"]) < DISPLACED_MID_CUTOFF),
            }
            for _, row in window.iterrows()
        ],
        "n_window": int(len(window)),
        "n_window_in_displaced_band": int((mids < DISPLACED_MID_CUTOFF).sum()),
        "window_min_mid": float(mids.min()),
        "window_max_mid": float(mids.max()),
    }


def event_record(
    ts: pd.Timestamp,
    log_return: float,
    price_before: float,
    price_after: float,
    selected: pd.Series | None,
    neighborhood: dict[str, object] | None,
) -> dict[str, object]:
    selected_mid = None if selected is None else float(selected["midquote"])
    in_band = selected_mid is not None and selected_mid < DISPLACED_MID_CUTOFF
    return {
        "timestamp": ts.isoformat(),
        "log_return": float(log_return),
        "abs_log_return": float(abs(log_return)),
        "previous_tick_mid_before": float(price_before),
        "previous_tick_mid_after": float(price_after),
        "selected_quote_timestamp": None
        if selected is None
        else selected["timestamp"].isoformat(),
        "selected_best_bid": None if selected is None else float(selected["best_bid"]),
        "selected_best_ask": None if selected is None else float(selected["best_ask"]),
        "selected_midquote": selected_mid,
        "selected_mid_in_displaced_band": in_band,
        "neighborhood": neighborhood,
    }


def realized_variance_by_frequency(
    ticks: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """RV and largest |r| at each Epps frequency. Uses the joint complete grid."""
    out: dict[str, dict[str, object]] = {}
    for delta in DEFAULT_FREQUENCIES_MINUTES:
        grid = calendar_session_grid(SAMPLE_DATE, delta)
        prices = previous_tick_sync(ticks, grid).reindex(columns=SYMBOLS)
        complete = drop_incomplete_open(prices)
        returns = synchronized_log_returns(complete)
        values = returns.to_numpy(dtype=float)
        # Unscaled realized variance is the diagonal of R'R.
        rv = np.sum(values * values, axis=0)
        abs_r = np.abs(values)
        by_asset: dict[str, dict[str, object]] = {}
        for i, asset in enumerate(SYMBOLS):
            j = int(np.argmax(abs_r[:, i]))
            by_asset[asset] = {
                "realized_variance": float(rv[i]),
                "max_abs_return": float(abs_r[j, i]),
                "max_abs_return_timestamp": returns.index[j].isoformat(),
            }
        out[str(delta)] = {
            "interval_minutes": delta,
            "n_returns": int(len(returns)),
            "by_asset": by_asset,
        }
    return out


def pair_subset_summary(
    pairwise: dict[str, float],
    keys: tuple[str, ...],
) -> dict[str, float | int]:
    values = np.array([pairwise[key] for key in keys], dtype=float)
    summary = summarize_off_diagonal(values)
    return {
        "n_pairs": summary.n_pairs,
        "mean": summary.mean,
        "median": summary.median,
        "minimum": summary.minimum,
        "maximum": summary.maximum,
    }


def fetch_cleaned_jpm_quotes() -> tuple[pd.DataFrame, dict[str, int]]:
    """P3 then Q1-Q4 on ordinary JPM NYSE quotes. Same adapter as the Epps panel."""
    import wrds

    venues = {
        "JPM": listing_venue_from_crsp_exchcd(
            "JPM",
            1,
            taq_sym_suffix=TAQ_ORDINARY_SHARE_SUFFIX,
        )
    }
    db = wrds.Connection()
    try:
        raw = fetch_cqm_selected_venues(db, venues)
    finally:
        db.close()
    suffix = raw["sym_suffix"]
    # Count nonblank suffixes that would indicate the identity leak remaining.
    audit = {
        "n_after_identity": int(len(raw)),
        "n_ordinary_suffix": int(suffix.map(is_ordinary_taq_suffix).sum()),
        "n_nonblank_suffix": int((~suffix.map(is_ordinary_taq_suffix)).sum()),
    }
    cleaned = clean_single_exchange_quotes(raw, venues)
    return cleaned.quotes, audit


def run_forensics(
    ticks: pd.DataFrame,
    jpm_quotes: pd.DataFrame,
    suffix_audit: dict[str, int],
) -> dict[str, object]:
    # Exact one-minute series from the five-asset Epps grid, then JPM's column.
    grid = calendar_session_grid(SAMPLE_DATE, 1)
    prices = previous_tick_sync(ticks, grid).reindex(columns=SYMBOLS)
    complete = drop_incomplete_open(prices)
    returns = synchronized_log_returns(complete)
    jpm_r = returns["JPM"]
    jpm_p = complete["JPM"]

    abs_r = jpm_r.abs()
    max_ts = abs_r.idxmax()
    events = jpm_r.loc[abs_r > RETURN_THRESHOLDS[0]]
    event_records = []
    for ts, r in events.items():
        loc = complete.index.get_loc(ts)
        price_after = float(jpm_p.iloc[loc])
        price_before = float(jpm_p.iloc[loc - 1])
        selected = previous_tick_quote(jpm_quotes, ts)
        neighborhood = None if selected is None else quote_neighborhood(jpm_quotes, selected)
        event_records.append(
            event_record(ts, float(r), price_before, price_after, selected, neighborhood)
        )

    # Count remaining cleaned JPM mids in the previously observed 22.x band.
    jpm_mids = jpm_quotes["midquote"].to_numpy(dtype=float)
    n_displaced = int((jpm_mids < DISPLACED_MID_CUTOFF).sum())
    one_min_selected = [
        previous_tick_quote(jpm_quotes, ts) for ts in complete.index
    ]
    selected_mids = [
        float(row["midquote"]) for row in one_min_selected if row is not None
    ]
    n_grid_in_band = int(sum(mid < DISPLACED_MID_CUTOFF for mid in selected_mids))

    rv_table = realized_variance_by_frequency(ticks)
    curve = epps_across_frequencies(
        ticks,
        day=SAMPLE_DATE,
        assets=SYMBOLS,
        frequencies_minutes=DEFAULT_FREQUENCIES_MINUTES,
    )
    official = []
    non_jpm = []
    for item in curve.by_frequency:
        official.append(
            {
                "interval_minutes": item.interval_minutes,
                "n_returns": item.n_returns,
                "mean_all_ten_pairs": item.off_diagonal.mean,
                "median_all_ten_pairs": item.off_diagonal.median,
                "min_all_ten_pairs": item.off_diagonal.minimum,
                "max_all_ten_pairs": item.off_diagonal.maximum,
                "pairwise": item.pairwise_correlations,
            }
        )
        non_jpm.append(
            {
                "interval_minutes": item.interval_minutes,
                **pair_subset_summary(item.pairwise_correlations, NON_JPM_PAIRS),
                "jpm_pairs": pair_subset_summary(item.pairwise_correlations, JPM_PAIRS),
            }
        )

    n_gt = {
        f"n_abs_gt_{int(100 * thr)}pct": int((abs_r > thr).sum())
        for thr in RETURN_THRESHOLDS
    }
    return {
        "question": (
            "After ordinary-share TAQ identity, do any |r|>2% one-minute JPM "
            "returns remain in the common-stock stream?"
        ),
        "suffix_audit": suffix_audit,
        "sample": {
            "date": SAMPLE_DATE,
            "ticks_cache": str(CACHE_TICKS.relative_to(REPO_ROOT)),
            "ticks_cache_used": CACHE_TICKS.exists(),
            "displaced_mid_cutoff": DISPLACED_MID_CUTOFF,
            "cutoff_note": (
                "Diagnostic band only. Midquotes strictly below 23 match the earlier "
                "NBBO 22.x cluster. This cutoff is not a filter."
            ),
        },
        "one_minute_grid": {
            "n_grid_prices": int(len(prices)),
            "n_complete_prices": int(len(complete)),
            "n_returns": int(len(returns)),
            "first_complete_timestamp": complete.index[0].isoformat(),
            "jpm_max_abs_return": float(abs_r.max()),
            "jpm_max_abs_return_timestamp": max_ts.isoformat(),
            "jpm_mid_at_max_event_before": float(jpm_p.loc[:max_ts].iloc[-2]),
            "jpm_mid_at_max_event_after": float(jpm_p.loc[max_ts]),
            **n_gt,
        },
        "events_abs_gt_2pct": event_records,
        "cleaned_jpm_displaced_band": {
            "n_cleaned_quotes": int(len(jpm_quotes)),
            "n_cleaned_mids_below_23": n_displaced,
            "min_cleaned_mid": float(jpm_mids.min()),
            "max_cleaned_mid": float(jpm_mids.max()),
            "n_one_minute_previous_ticks_below_23": n_grid_in_band,
        },
        "realized_variance_by_frequency": rv_table,
        "epps_official_all_ten_pairs": official,
        "epps_diagnostic_six_pairs_excluding_jpm": non_jpm,
        "caveat": (
            "The six-pair summary is diagnostic only. The official universe remains "
            "IBM, AAPL, MSFT, JPM, XOM. Large returns are reported, not deleted."
        ),
    }


def _print_report(report: dict[str, object]) -> None:
    one = report["one_minute_grid"]
    print("Suffix audit", report["suffix_audit"])
    print("One-minute JPM series")
    print(
        f"  grid={one['n_grid_prices']} complete={one['n_complete_prices']} "
        f"returns={one['n_returns']}"
    )
    print(
        f"  max |r|={one['jpm_max_abs_return']:.6g} at {one['jpm_max_abs_return_timestamp']}"
    )
    print(
        f"  mids {one['jpm_mid_at_max_event_before']:.4f} -> "
        f"{one['jpm_mid_at_max_event_after']:.4f}"
    )
    print(
        f"  |r|>2%={one['n_abs_gt_2pct']} "
        f"|r|>5%={one['n_abs_gt_5pct']} "
        f"|r|>10%={one['n_abs_gt_10pct']}"
    )
    band = report["cleaned_jpm_displaced_band"]
    print("Cleaned JPM displaced band (mid < 23)")
    print(
        f"  cleaned quotes={band['n_cleaned_quotes']} "
        f"mids<23={band['n_cleaned_mids_below_23']} "
        f"1-min previous-ticks<23={band['n_one_minute_previous_ticks_below_23']} "
        f"mid range=[{band['min_cleaned_mid']:.4f}, {band['max_cleaned_mid']:.4f}]"
    )
    print("Events |r|>2%")
    for event in report["events_abs_gt_2pct"]:
        print(
            f"  {event['timestamp']} r={event['log_return']:.6g} "
            f"P={event['previous_tick_mid_before']:.4f}->"
            f"{event['previous_tick_mid_after']:.4f} "
            f"selected_mid={event['selected_midquote']} "
            f"in_22x={event['selected_mid_in_displaced_band']}"
        )
        neighborhood = event["neighborhood"]
        if neighborhood is not None:
            print(
                f"    window mids [{neighborhood['window_min_mid']:.4f}, "
                f"{neighborhood['window_max_mid']:.4f}] "
                f"n_in_22x={neighborhood['n_window_in_displaced_band']}"
            )
    print("Realized variance by frequency")
    for delta, row in report["realized_variance_by_frequency"].items():
        parts = [
            f"{asset} RV={stats['realized_variance']:.6g} "
            f"max|r|={stats['max_abs_return']:.4g}@{stats['max_abs_return_timestamp'][11:19]}"
            for asset, stats in row["by_asset"].items()
        ]
        print(f"  {delta} min n={row['n_returns']}  " + " | ".join(parts))
    print("Official ten-pair Epps (mean / median)")
    for row in report["epps_official_all_ten_pairs"]:
        print(
            f"  {row['interval_minutes']:>2} min  mean={row['mean_all_ten_pairs']:.4f}  "
            f"median={row['median_all_ten_pairs']:.4f}"
        )
    print("Diagnostic six pairs excluding JPM (mean / median)")
    for row in report["epps_diagnostic_six_pairs_excluding_jpm"]:
        print(
            f"  {row['interval_minutes']:>2} min  mean={row['mean']:.4f}  "
            f"median={row['median']:.4f}  "
            f"JPM-pair mean={row['jpm_pairs']['mean']:.4f}"
        )


def main() -> dict[str, object]:
    meta = load_or_build_cleaned_stream()
    ticks = meta["ticks"]
    if not CACHE_TICKS.exists():
        raise RuntimeError("expected the cleaned single-exchange tick cache used by Epps")
    jpm_quotes, suffix_audit = fetch_cleaned_jpm_quotes()
    report = run_forensics(ticks, jpm_quotes, suffix_audit)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str))
    _print_report(report)
    print("Wrote", RESULTS_PATH)
    return report


if __name__ == "__main__":
    main()
