"""Forensic check. Is the surviving JPM cluster an NBBO-adaptation problem?

Compares three JPM quote series on 2009-02-13 without changing ``clean_nbbo_quotes``.

A. Naive consolidated NBBO (positive prices, nonnegative spread, last duplicate).
B. Production Q1-Q4 on consolidated NBBO (no P3).
C. Paper-style P3 NYSE quotes from ``cqm_20090213``, then the same production Q1-Q4.

NYSE venue code ``N`` is taken from the Daily TAQ Client Specification exchange
field, which is the metadata for TAQ ``ex``. It is confirmed against distinct
``ex`` values present in the JPM quote table that day.

Trades are an independent diagnostic only. They do not overwrite quote values.
Raw ticks are not written to the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from covharness.data.quotes import clean_nbbo_quotes, cleaned_quotes_as_ticks
from covharness.data.returns import synchronized_log_returns
from covharness.data.synchronization import previous_tick_sync
from experiments.taq_five_stock_pilot import (
    SAMPLE_DATE,
    SESSION_END_SEC,
    SESSION_START_SEC,
    WRDS_LIBRARY,
    diagnostics_as_dict,
    drop_incomplete_open,
    five_minute_grid,
    naive_midquote_ticks,
)

# Daily TAQ Client Specification, quote Exchange field: N = New York Stock Exchange.
NYSE_EXCHANGE_CODE = "N"
NYSE_CODE_SOURCE = (
    "NYSE Daily TAQ Client Specification, Exchange field on the Daily Quotes file. "
    "N denotes New York Stock Exchange. WRDS ``cqm_*.ex`` is that same TAQ field."
)

CHECK_TIMES = [
    "09:40",
    "09:45",
    "09:50",
    "10:00",
    "10:05",
    "10:10",
    "10:40",
]
TRADE_WINDOWS = {
    "09:45": (35090, 35110),
    "10:05": (36290, 36310),
    "10:40": (38390, 38410),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "taq_jpm_p3_forensics_20090213.json"


def _stamp(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["date"]) + pd.to_timedelta(out["time_m"], unit="s")
    return out


def fetch_jpm_nbbo(db) -> pd.DataFrame:
    sql = f"""
        SELECT date, time_m, sym_root, best_bid, best_ask, best_bidex, best_askex
        FROM {WRDS_LIBRARY}.nbbom_20090213
        WHERE sym_root = 'JPM'
          AND time_m >= {SESSION_START_SEC}
          AND time_m <= {SESSION_END_SEC}
        ORDER BY time_m
    """
    raw = _stamp(db.raw_sql(sql))
    raw = raw.rename(columns={"sym_root": "asset"})
    for col in ("best_bid", "best_ask"):
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    return raw


def fetch_jpm_cqm(db) -> pd.DataFrame:
    sql = f"""
        SELECT date, time_m, ex, sym_root, bid, ask, qu_cond, qu_cancel, qu_seqnum
        FROM {WRDS_LIBRARY}.cqm_20090213
        WHERE sym_root = 'JPM'
          AND time_m >= {SESSION_START_SEC}
          AND time_m <= {SESSION_END_SEC}
        ORDER BY time_m, qu_seqnum
    """
    raw = _stamp(db.raw_sql(sql))
    raw = raw.rename(columns={"sym_root": "asset"})
    raw["bid"] = pd.to_numeric(raw["bid"], errors="raise")
    raw["ask"] = pd.to_numeric(raw["ask"], errors="raise")
    return raw


def fetch_jpm_trades(db, lo: float, hi: float) -> pd.DataFrame:
    sql = f"""
        SELECT time_m, price, size, ex, tr_scond, tr_corr
        FROM {WRDS_LIBRARY}.ctm_20090213
        WHERE sym_root = 'JPM'
          AND time_m >= {lo}
          AND time_m <= {hi}
          AND price > 0
          AND size > 0
        ORDER BY time_m
    """
    return db.raw_sql(sql)


def paper_style_nyse_quotes(cqm: pd.DataFrame) -> pd.DataFrame:
    """P3 first. Keep NYSE quotes only, then map bid/ask onto the production cleaner."""
    nyse = cqm.loc[cqm["ex"].astype(str) == NYSE_EXCHANGE_CODE].copy()
    return nyse.rename(columns={"bid": "best_bid", "ask": "best_ask"})[
        ["timestamp", "asset", "best_bid", "best_ask"]
    ]


def sync_jpm(ticks: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    synced = previous_tick_sync(ticks, grid)
    return synced["JPM"]


def grid_slice(series: pd.Series) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for hhmm in CHECK_TIMES:
        ts = pd.Timestamp(f"{SAMPLE_DATE} {hhmm}:00")
        if ts not in series.index:
            out[hhmm] = None
            continue
        value = series.loc[ts]
        out[hhmm] = None if pd.isna(value) else float(value)
    return out


def max_abs_return(prices: pd.Series) -> dict[str, object]:
    wide = prices.to_frame("JPM")
    complete = drop_incomplete_open(wide)
    returns = synchronized_log_returns(complete)["JPM"]
    abs_r = returns.abs()
    return {
        "n_returns": int(len(returns)),
        "max_abs_return": float(abs_r.max()),
        "timestamp": abs_r.idxmax().isoformat(),
        "n_abs_gt_2pct": int((abs_r > 0.02).sum()),
        "n_abs_gt_5pct": int((abs_r > 0.05).sum()),
        "n_abs_gt_10pct": int((abs_r > 0.10).sum()),
    }


def trade_summary(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {"n": 0}
    price = pd.to_numeric(trades["price"], errors="raise")
    return {
        "n": int(len(trades)),
        "min": float(price.min()),
        "median": float(price.median()),
        "max": float(price.max()),
        "mean": float(price.mean()),
    }


def venue_of_displaced_nbbo(nbbo: pd.DataFrame, cutoff: float = 23.0) -> dict[str, object]:
    """Which NBBO venues carry midquotes below 23. Diagnostic only."""
    mid = (nbbo["best_bid"] + nbbo["best_ask"]) / 2.0
    low = nbbo.loc[(nbbo["best_bid"] > 0) & (nbbo["best_ask"] > 0) & (mid < cutoff)]
    return {
        "n_mid_below_23": int(len(low)),
        "best_bidex": low["best_bidex"].astype(str).value_counts().to_dict() if len(low) else {},
        "best_askex": low["best_askex"].astype(str).value_counts().to_dict() if len(low) else {},
    }


def run_forensics(nbbo: pd.DataFrame, cqm: pd.DataFrame, trade_windows: dict[str, pd.DataFrame]) -> dict[str, object]:
    grid = five_minute_grid()

    # A. Naive consolidated NBBO.
    naive_ticks = naive_midquote_ticks(nbbo.drop(columns=["best_bidex", "best_askex"], errors="ignore"))
    prices_a = sync_jpm(naive_ticks, grid)

    # B. Production Q1-Q4 on consolidated NBBO. No P3.
    cleaned_b = clean_nbbo_quotes(nbbo)
    prices_b = sync_jpm(cleaned_quotes_as_ticks(cleaned_b), grid)

    # C. P3 NYSE, then the same production Q1-Q4.
    observed_ex = sorted(cqm["ex"].astype(str).fillna("").unique().tolist())
    n_cqm = int(len(cqm))
    nyse_quotes = paper_style_nyse_quotes(cqm)
    n_after_p3 = int(len(nyse_quotes))
    cleaned_c = clean_nbbo_quotes(nyse_quotes)
    prices_c = sync_jpm(cleaned_quotes_as_ticks(cleaned_c), grid)

    # NYSE midquotes below 23 after P3+Q1-Q4.
    nyse_low = int((cleaned_c.quotes["midquote"] < 23.0).sum())
    nbbo_low = int((cleaned_b.quotes["midquote"] < 23.0).sum())

    return {
        "sample": {
            "date": SAMPLE_DATE,
            "asset": "JPM",
            "quote_table": "taqmsamp_all.cqm_20090213",
            "nbbo_table": "taqmsamp_all.nbbom_20090213",
            "trade_table": "taqmsamp_all.ctm_20090213",
        },
        "p3": {
            "nyse_exchange_code": NYSE_EXCHANGE_CODE,
            "code_source": NYSE_CODE_SOURCE,
            "observed_ex_codes_in_jpm_cqm": observed_ex,
            "n_cqm_session": n_cqm,
            "n_after_p3_nyse": n_after_p3,
            "p3_retention_fraction": n_after_p3 / n_cqm if n_cqm else 0.0,
        },
        "cqm_quote_condition": cqm["qu_cond"].astype(str).value_counts().to_dict(),
        "cqm_quote_cancel": cqm["qu_cancel"].astype(str).value_counts().to_dict(),
        "nyse_quote_condition": (
            cqm.loc[cqm["ex"].astype(str) == NYSE_EXCHANGE_CODE, "qu_cond"]
            .astype(str)
            .value_counts()
            .to_dict()
        ),
        "nyse_quote_cancel": (
            cqm.loc[cqm["ex"].astype(str) == NYSE_EXCHANGE_CODE, "qu_cancel"]
            .astype(str)
            .value_counts()
            .to_dict()
        ),
        "cleaning_B_nbbo_q1q4": diagnostics_as_dict(cleaned_b.diagnostics),
        "cleaning_C_nyse_p3_q1q4": diagnostics_as_dict(cleaned_c.diagnostics),
        "n_cleaned_midquotes_below_23": {"B_nbbo": nbbo_low, "C_nyse": nyse_low},
        "displaced_nbbo_venues": venue_of_displaced_nbbo(nbbo),
        "five_minute_jpm": {
            "A_naive_nbbo": grid_slice(prices_a),
            "B_cleaned_nbbo": grid_slice(prices_b),
            "C_paper_nyse": grid_slice(prices_c),
        },
        "max_abs_5min_return": {
            "A_naive_nbbo": max_abs_return(prices_a),
            "B_cleaned_nbbo": max_abs_return(prices_b),
            "C_paper_nyse": max_abs_return(prices_c),
        },
        "trades_around_windows": {
            name: trade_summary(frame) for name, frame in trade_windows.items()
        },
        "cleaner_unchanged": "covharness.data.quotes.clean_nbbo_quotes",
        "caveat": (
            "P3 was applied as a pre-filter on exchange-level quotes. "
            "Q1-Q4 were the existing production functions. "
            "No new outlier threshold was introduced."
        ),
    }


def _print_report(report: dict[str, object]) -> None:
    print("NYSE code", report["p3"]["nyse_exchange_code"])
    print("Source", report["p3"]["code_source"])
    print("Observed cqm ex codes", report["p3"]["observed_ex_codes_in_jpm_cqm"])
    print("cqm rows", report["p3"]["n_cqm_session"], "after P3 NYSE", report["p3"]["n_after_p3_nyse"])
    print("B cleaning", report["cleaning_B_nbbo_q1q4"])
    print("C cleaning", report["cleaning_C_nyse_p3_q1q4"])
    print("mids < 23 after cleaning", report["n_cleaned_midquotes_below_23"])
    print("Displaced NBBO venues", report["displaced_nbbo_venues"])
    print("Five-minute JPM")
    five = report["five_minute_jpm"]
    for hhmm in CHECK_TIMES:
        print(
            f"  {hhmm}: A={five['A_naive_nbbo'][hhmm]}  "
            f"B={five['B_cleaned_nbbo'][hhmm]}  "
            f"C={five['C_paper_nyse'][hhmm]}"
        )
    print("Max |r|")
    for key, stats in report["max_abs_5min_return"].items():
        print(f"  {key}: {stats}")
    print("Trades")
    for name, stats in report["trades_around_windows"].items():
        print(f"  {name}: {stats}")


def main() -> dict[str, object]:
    import wrds

    db = wrds.Connection()
    try:
        nbbo = fetch_jpm_nbbo(db)
        cqm = fetch_jpm_cqm(db)
        trade_windows = {
            name: fetch_jpm_trades(db, lo, hi) for name, (lo, hi) in TRADE_WINDOWS.items()
        }
    finally:
        db.close()
    report = run_forensics(nbbo, cqm, trade_windows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    _print_report(report)
    print("Wrote", RESULTS_PATH)
    return report


if __name__ == "__main__":
    main()
