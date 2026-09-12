"""JPM identity reconciliation. Reconstructs historical root-only extraction.

Compares the corrected ordinary-share stream with a local reconstruction of
the old root-only stream through the production previous-tick synchronizer
and unscaled RCov. Does not restore root-only identity in production code.
Does not change Q1-Q4, P3, previous-tick, or RCov.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from covharness.data.exchange_quotes import clean_single_exchange_quotes
from covharness.data.quotes import cleaned_quotes_as_ticks
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
)
from covharness.realized.daily import daily_realized_covariance
from experiments.taq_five_stock_pilot import (
    SAMPLE_DATE,
    SESSION_END_SEC,
    SESSION_START_SEC,
    WRDS_LIBRARY,
)

ASSET = "JPM"
QUOTE_TABLE = f"{WRDS_LIBRARY}.cqm_{SAMPLE_DATE.replace('-', '')}"
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "taq_jpm_identity_reconciliation_20090213.json"


def _repr_suffix(value: object) -> str:
    if is_ordinary_taq_suffix(value):
        return "<NULL>" if value is None or (not isinstance(value, str)) else "<BLANK>"
    return str(value).strip()


def fetch_jpm_nyse_root_only(db) -> pd.DataFrame:
    """Historical extract. Root plus NYSE venue, no suffix restriction."""
    sql = f"""
        SELECT date, time_m, ex, sym_root, sym_suffix, bid, ask
        FROM {QUOTE_TABLE}
        WHERE time_m >= {SESSION_START_SEC}
          AND time_m <= {SESSION_END_SEC}
          AND sym_root = '{ASSET}'
          AND ex IN ('N')
        ORDER BY time_m
    """
    raw = db.raw_sql(sql)
    # Exchange-local timestamp from calendar date plus seconds after midnight.
    raw["timestamp"] = pd.to_datetime(raw["date"]) + pd.to_timedelta(raw["time_m"], unit="s")
    raw = raw.rename(columns={"sym_root": "asset"})
    raw["bid"] = pd.to_numeric(raw["bid"], errors="raise")
    raw["ask"] = pd.to_numeric(raw["ask"], errors="raise")
    return raw.loc[:, ["timestamp", "asset", "ex", "sym_suffix", "bid", "ask"]].copy()


def q1_input_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """P2 survivors that enter Q1. Session hours are already applied remotely."""
    bid = raw["bid"].to_numpy(dtype=float, na_value=np.nan)
    ask = raw["ask"].to_numpy(dtype=float, na_value=np.nan)
    p2_ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask > 0.0)
    return raw.loc[p2_ok].copy()


def previous_tick_source_times(
    ticks: pd.DataFrame,
    grid: pd.DatetimeIndex,
) -> pd.Series:
    """Backward as-of map from grid times to the selected cleaned tick time."""
    sources = ticks.loc[:, ["timestamp"]].sort_values("timestamp")
    sources = sources.rename(columns={"timestamp": "source_timestamp"})
    grid_frame = pd.DataFrame({"grid_time": pd.DatetimeIndex(grid)})
    matched = pd.merge_asof(
        grid_frame,
        sources,
        left_on="grid_time",
        right_on="source_timestamp",
        direction="backward",
    )
    return matched.set_index("grid_time")["source_timestamp"]


def suffixes_at_timestamp(lookup: pd.DataFrame, timestamp: pd.Timestamp) -> list[str]:
    rows = lookup.loc[lookup["timestamp"] == timestamp, "sym_suffix"]
    return sorted({_repr_suffix(value) for value in rows})


def frequency_stats(
    ticks: pd.DataFrame,
    interval_minutes: int,
    *,
    lookup: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Production previous-tick plus RCov at one width. Optional suffix hits."""
    grid = calendar_session_grid(SAMPLE_DATE, interval_minutes)
    prices = previous_tick_sync(ticks, grid).reindex(columns=[ASSET])
    complete = drop_incomplete_open(prices)
    returns = synchronized_log_returns(complete)
    rcov = daily_realized_covariance(complete)
    values = returns[ASSET].to_numpy(dtype=float)
    abs_r = np.abs(values)
    j = int(np.argmax(abs_r))
    stats: dict[str, object] = {
        "interval_minutes": interval_minutes,
        "n_grid_prices": int(len(prices)),
        "n_synchronized_prices": int(len(complete)),
        "n_returns": int(len(returns)),
        "realized_variance": float(rcov[0, 0]),
        "max_abs_return": float(abs_r[j]),
        "max_abs_return_timestamp": returns.index[j].isoformat(),
        "complete_prices": complete[ASSET].to_numpy(dtype=float),
        "complete_index": [ts.isoformat() for ts in complete.index],
    }
    if lookup is None:
        return stats

    # Selected previous-tick source times on the complete synchronized grid.
    sources = previous_tick_source_times(ticks, complete.index)
    hits: list[dict[str, object]] = []
    missing_source_rows = 0
    for grid_time, source_ts in sources.items():
        if pd.isna(source_ts):
            missing_source_rows += 1
            continue
        source_ts = pd.Timestamp(source_ts)
        suffixes = suffixes_at_timestamp(lookup, source_ts)
        if not suffixes:
            missing_source_rows += 1
            continue
        if any(label not in {"<NULL>", "<BLANK>"} for label in suffixes):
            hits.append(
                {
                    "grid_time": pd.Timestamp(grid_time).isoformat(),
                    "source_timestamp": source_ts.isoformat(),
                    "suffixes": suffixes,
                    "n_q1_input_rows": int((lookup["timestamp"] == source_ts).sum()),
                }
            )
    stats["n_noncommon_suffix_hits"] = int(len(hits))
    stats["n_missing_source_suffix_lookup"] = int(missing_source_rows)
    stats["noncommon_suffix_hits"] = hits
    return stats


def five_minute_explanation(
    corrected: dict[str, object],
    root_only: dict[str, object],
) -> dict[str, object]:
    """Reconcile 5-minute RV equality from the observed suffix hits."""
    prices_equal = np.array_equal(corrected["complete_prices"], root_only["complete_prices"])
    rv_equal = bool(
        np.isclose(
            corrected["realized_variance"],
            root_only["realized_variance"],
            rtol=0.0,
            atol=0.0,
        )
        or np.isclose(
            corrected["realized_variance"],
            root_only["realized_variance"],
            rtol=1e-15,
            atol=1e-18,
        )
    )
    n_hits = int(root_only["n_noncommon_suffix_hits"])
    second_defect = False
    if n_hits == 0 and rv_equal:
        reason = (
            "The historical root-only 5-minute previous-tick grid selects zero "
            "noncommon suffixes. Every complete 5-minute grid price is sourced "
            "from an ordinary-share cleaned quote. That fully explains why the "
            "5-minute realized variance was unchanged by the identity fix."
        )
    elif n_hits > 0 and rv_equal:
        reason = (
            "The historical root-only 5-minute grid selects one or more "
            "noncommon suffixes and still reproduces the corrected 5-minute RV. "
            "See noncommon_suffix_hits for those source rows."
        )
    elif n_hits == 0 and not rv_equal:
        second_defect = True
        reason = (
            "The historical root-only 5-minute grid selects zero noncommon "
            "suffixes, but the 5-minute RV differs from the corrected stream. "
            "That is a second defect. Stop."
        )
    else:
        reason = (
            "The historical root-only 5-minute grid selects noncommon suffixes "
            "and the 5-minute RV differs from the corrected stream."
        )
    if int(root_only["n_missing_source_suffix_lookup"]) > 0:
        second_defect = True
        reason = (
            "A cleaned previous-tick timestamp could not be matched to Q1-input "
            "rows. That is a second defect. Stop."
        )
    return {
        "rv_equal": rv_equal,
        "price_series_equal": bool(prices_equal),
        "n_noncommon_suffix_hits": n_hits,
        "second_defect": second_defect,
        "explanation": reason,
    }


def _printable_stats(stats: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in stats.items()
        if key not in {"complete_prices", "complete_index"}
    }


def main() -> dict[str, object]:
    import wrds

    venues = {
        ASSET: listing_venue_from_crsp_exchcd(
            ASSET, 1, taq_sym_suffix=TAQ_ORDINARY_SHARE_SUFFIX
        )
    }
    db = wrds.Connection()
    try:
        # One NYSE JPM pull. Stream A is ordinary-only. Stream B drops suffix.
        raw_root_only = fetch_jpm_nyse_root_only(db)
    finally:
        db.close()

    ordinary_mask = raw_root_only["sym_suffix"].map(is_ordinary_taq_suffix)
    raw_corrected = raw_root_only.loc[ordinary_mask].copy()
    raw_historical = raw_root_only.drop(columns=["sym_suffix"])
    lookup = q1_input_rows(raw_root_only)

    # Production cleaner. Stream B omits sym_suffix so P3 is venue-only.
    cleaned_a = clean_single_exchange_quotes(raw_corrected, venues)
    cleaned_b = clean_single_exchange_quotes(raw_historical, venues)
    ticks_a = cleaned_quotes_as_ticks(cleaned_a.quotes)
    ticks_b = cleaned_quotes_as_ticks(cleaned_b.quotes)

    by_frequency = []
    for delta in DEFAULT_FREQUENCIES_MINUTES:
        corrected = frequency_stats(ticks_a, delta)
        root_only = frequency_stats(ticks_b, delta, lookup=lookup)
        by_frequency.append(
            {
                "interval_minutes": delta,
                "corrected": _printable_stats(corrected),
                "root_only": _printable_stats(root_only),
            }
        )

    five = next(item for item in by_frequency if item["interval_minutes"] == 5)
    five_corrected = frequency_stats(ticks_a, 5)
    five_root_only = frequency_stats(ticks_b, 5, lookup=lookup)
    five_min = five_minute_explanation(five_corrected, five_root_only)

    report = {
        "sample": {
            "date": SAMPLE_DATE,
            "asset": ASSET,
            "quote_table": QUOTE_TABLE,
            "historical_predicate": (
                "sym_root = 'JPM' AND ex IN ('N') AND session hours. "
                "No suffix restriction."
            ),
            "corrected_predicate": (
                "same venue and hours, ordinary/common suffix only "
                "(SQL NULL or blank)."
            ),
        },
        "row_counts": {
            "root_only_after_sql": int(len(raw_root_only)),
            "corrected_after_identity": int(len(raw_corrected)),
            "root_only_nonblank_suffix": int((~ordinary_mask).sum()),
            "corrected_after_q4": int(len(cleaned_a.quotes)),
            "root_only_after_q4": int(len(cleaned_b.quotes)),
        },
        "same_production_functions": {
            "previous_tick_sync": "covharness.data.synchronization.previous_tick_sync",
            "drop_incomplete_open": "covharness.diagnostics.epps.drop_incomplete_open",
            "daily_realized_covariance": "covharness.realized.daily.daily_realized_covariance",
            "cleaner": "covharness.data.exchange_quotes.clean_single_exchange_quotes",
            "second_defect": False,
            "note": (
                "Stream B is reconstructed by omitting sym_suffix so production "
                "P3 cannot apply the repaired identity filter. Production fetch "
                "is not reverted."
            ),
        },
        "by_frequency": by_frequency,
        "five_minute_reconciliation": five_min,
        "caveat": (
            "Previous-tick contamination is not the raw preferred-quote fraction. "
            "A preferred quote contaminates a grid time only while it remains "
            "the most recent cleaned quote."
        ),
    }
    if five_min["second_defect"]:
        report["same_production_functions"]["second_defect"] = True

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    _print_report(report)
    print("Wrote", RESULTS_PATH)
    return report


def _print_report(report: dict[str, object]) -> None:
    print("Row counts", report["row_counts"])
    print(
        "Minutes  n_prices  n_returns  RV_corr  RV_root  "
        "|r|_corr  |r|_root  hits_root"
    )
    for item in report["by_frequency"]:
        corr = item["corrected"]
        root = item["root_only"]
        print(
            f"  {item['interval_minutes']:>2}  "
            f"{corr['n_synchronized_prices']:>4}/{root['n_synchronized_prices']:<4}  "
            f"{corr['n_returns']:>3}/{root['n_returns']:<3}  "
            f"{corr['realized_variance']:.6g}  {root['realized_variance']:.6g}  "
            f"{corr['max_abs_return']:.6g}@{corr['max_abs_return_timestamp'][11:19]}  "
            f"{root['max_abs_return']:.6g}@{root['max_abs_return_timestamp'][11:19]}  "
            f"hits={root['n_noncommon_suffix_hits']}"
        )
    print("Root-only noncommon previous-tick hits")
    for item in report["by_frequency"]:
        hits = item["root_only"]["noncommon_suffix_hits"]
        print(
            f"  {item['interval_minutes']:>2} min  "
            f"n={item['root_only']['n_noncommon_suffix_hits']}"
        )
        for hit in hits:
            print(
                f"    grid={hit['grid_time']}  source={hit['source_timestamp']}  "
                f"suffixes={hit['suffixes']}  q1_rows={hit['n_q1_input_rows']}"
            )
    five = report["five_minute_reconciliation"]
    print("Five-minute reconciliation")
    print("  rv_equal", five["rv_equal"])
    print("  price_series_equal", five["price_series_equal"])
    print("  n_noncommon_suffix_hits", five["n_noncommon_suffix_hits"])
    print("  second_defect", five["second_defect"])
    print("  ", five["explanation"])


if __name__ == "__main__":
    main()
