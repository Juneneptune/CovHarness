"""Real-data validation of the measurement chain on the WRDS TAQ millisecond sample.

Sample. Five stocks on 2009-02-13 from ``taqmsamp_all.nbbom_20090213``.
This script extracts a filtered NBBO slice, runs the production cleaner, and
compares naive versus cleaned previous-tick realized covariance. It does not
implement Q1-Q4, realized covariance, or realized kernels.

Raw licensed ticks are not written to the repository. Derived diagnostics may
be written to ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from covharness.data.quotes import (
    QuoteCleaningDiagnostics,
    clean_nbbo_quotes,
    cleaned_quotes_as_ticks,
)
from covharness.data.returns import synchronized_log_returns
from covharness.data.synchronization import previous_tick_sync
from covharness.realized.daily import daily_realized_covariance

SYMBOLS = ["IBM", "AAPL", "MSFT", "JPM", "XOM"]
SAMPLE_DATE = "2009-02-13"
WRDS_LIBRARY = "taqmsamp_all"
WRDS_TABLE = "nbbom_20090213"
SESSION_START_SEC = 9 * 3600 + 30 * 60
SESSION_END_SEC = 16 * 3600
JPM_CHECK_TIMES = [
    "09:40",
    "09:45",
    "09:50",
    "10:00",
    "10:05",
    "10:10",
]
RETURN_THRESHOLDS = (0.02, 0.05, 0.10)
PSD_ATOL = 1e-12

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "taq_five_stock_pilot_20090213.json"


def fetch_nbbo_sample(db) -> pd.DataFrame:
    """Pull the five-stock regular-session NBBO slice. Session hours are remote."""
    symbols_sql = ", ".join(f"'{s}'" for s in SYMBOLS)
    sql = f"""
        SELECT
            date,
            time_m,
            sym_root,
            best_bid,
            best_ask
        FROM {WRDS_LIBRARY}.{WRDS_TABLE}
        WHERE sym_root IN ({symbols_sql})
          AND time_m >= {SESSION_START_SEC}
          AND time_m <= {SESSION_END_SEC}
        ORDER BY sym_root, time_m
    """
    raw = db.raw_sql(sql)
    # Build exchange-local timestamps from calendar date plus seconds after midnight.
    raw["timestamp"] = pd.to_datetime(raw["date"]) + pd.to_timedelta(raw["time_m"], unit="s")
    raw = raw.rename(columns={"sym_root": "asset"})
    raw["best_bid"] = pd.to_numeric(raw["best_bid"], errors="raise")
    raw["best_ask"] = pd.to_numeric(raw["best_ask"], errors="raise")
    return raw.loc[:, ["timestamp", "asset", "best_bid", "best_ask"]]


def naive_midquote_ticks(raw: pd.DataFrame) -> pd.DataFrame:
    """Minimal diagnostic path. Positive prices and nonnegative spreads only.

    Duplicate timestamps keep the last row so previous-tick sync can run.
    That last-row collapse is not Q1 (median bid and ask).
    """
    naive = raw.copy()
    naive = naive.loc[(naive["best_bid"] > 0) & (naive["best_ask"] > 0)].copy()
    naive["spread"] = naive["best_ask"] - naive["best_bid"]
    naive = naive.loc[naive["spread"] >= 0].copy()
    naive = naive.sort_values(["asset", "timestamp"]).drop_duplicates(
        ["asset", "timestamp"], keep="last"
    )
    naive["price"] = (naive["best_bid"] + naive["best_ask"]) / 2.0
    return naive.loc[:, ["timestamp", "asset", "price"]].reset_index(drop=True)


def five_minute_grid(day: str = SAMPLE_DATE) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{day} 09:30:00")
    end = pd.Timestamp(f"{day} 16:00:00")
    return pd.date_range(start, end, freq="5min")


def synchronize_panel(ticks: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    synced = previous_tick_sync(ticks, grid)
    return synced.reindex(columns=SYMBOLS)


def drop_incomplete_open(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop grid rows with any missing price. Do not fill the 09:30 point."""
    return prices.dropna(how="any")


def diagnostics_as_dict(diag: QuoteCleaningDiagnostics) -> dict[str, float | int]:
    return {
        "n_input": diag.n_input,
        "n_after_p1": diag.n_after_p1,
        "n_after_p2": diag.n_after_p2,
        "n_after_q1": diag.n_after_q1,
        "n_after_q2": diag.n_after_q2,
        "n_after_q3": diag.n_after_q3,
        "n_after_q4": diag.n_after_q4,
        "n_removed_p1": diag.n_removed_p1,
        "n_removed_p2": diag.n_removed_p2,
        "n_removed_q1": diag.n_removed_q1,
        "n_removed_q2": diag.n_removed_q2,
        "n_removed_q3": diag.n_removed_q3,
        "n_removed_q4": diag.n_removed_q4,
        "removal_fraction": diag.removal_fraction,
    }


def per_stock_stage_counts(raw: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Call the production cleaner once per stock. Q3 and Q4 are already stock-day local."""
    counts: dict[str, dict[str, float | int]] = {}
    for symbol in SYMBOLS:
        subset = raw.loc[raw["asset"] == symbol]
        result = clean_nbbo_quotes(subset)
        counts[symbol] = diagnostics_as_dict(result.diagnostics)
    return counts


def jpm_grid_slice(prices: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for hhmm in JPM_CHECK_TIMES:
        ts = pd.Timestamp(f"{SAMPLE_DATE} {hhmm}:00")
        if ts not in prices.index or "JPM" not in prices.columns:
            out[hhmm] = None
            continue
        value = prices.loc[ts, "JPM"]
        out[hhmm] = None if pd.isna(value) else float(value)
    return out


def return_sanity(returns: pd.DataFrame) -> dict[str, object]:
    abs_r = returns.abs()
    max_abs = abs_r.max()
    max_times = abs_r.idxmax()
    counts = {
        f"n_abs_gt_{int(100 * thr)}pct": int((abs_r > thr).sum().sum())
        for thr in RETURN_THRESHOLDS
    }
    by_stock = {
        symbol: {
            "max_abs_return": float(max_abs[symbol]),
            "timestamp": max_times[symbol].isoformat(),
            **{
                f"n_abs_gt_{int(100 * thr)}pct": int((abs_r[symbol] > thr).sum())
                for thr in RETURN_THRESHOLDS
            },
        }
        for symbol in returns.columns
    }
    return {"counts_all_entries": counts, "by_stock": by_stock}


def matrix_diagnostics(sigma: np.ndarray, assets: list[str]) -> dict[str, object]:
    # Eigenvalues of the symmetric Gram matrix. No repair.
    eigvals = np.linalg.eigvalsh(sigma)
    min_eig = float(eigvals.min())
    rank = int(np.linalg.matrix_rank(sigma, tol=PSD_ATOL))
    psd = bool(min_eig >= -PSD_ATOL)
    if min_eig > PSD_ATOL:
        cond = float(eigvals.max() / eigvals.min())
    else:
        cond = float("inf")
    vol = np.sqrt(np.clip(np.diag(sigma), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = sigma / np.outer(vol, vol)
    return {
        "assets": assets,
        "rcov": sigma.tolist(),
        "realized_variance": {a: float(v) for a, v in zip(assets, np.diag(sigma))},
        "symmetry_error": float(np.max(np.abs(sigma - sigma.T))),
        "eigenvalues": eigvals.tolist(),
        "min_eigenvalue": min_eig,
        "numerical_rank": rank,
        "psd": psd,
        "condition_number": None if not np.isfinite(cond) else cond,
        "realized_correlation": corr.tolist(),
    }


def run_pilot(raw: pd.DataFrame) -> dict[str, object]:
    grid = five_minute_grid()

    # Production Q1-Q4 path, then previous-tick on cleaned midquotes.
    cleaned = clean_nbbo_quotes(raw)
    clean_ticks = cleaned_quotes_as_ticks(cleaned)
    prices_clean = synchronize_panel(clean_ticks, grid)
    missing_clean = prices_clean.isna().sum().astype(int).to_dict()
    complete_clean = drop_incomplete_open(prices_clean)
    returns_clean = synchronized_log_returns(complete_clean)
    rcov_clean = daily_realized_covariance(complete_clean)

    # Naive diagnostic path. Not the production cleaner.
    naive_ticks = naive_midquote_ticks(raw)
    prices_naive = synchronize_panel(naive_ticks, grid)
    missing_naive = prices_naive.isna().sum().astype(int).to_dict()
    complete_naive = drop_incomplete_open(prices_naive)
    rcov_naive = daily_realized_covariance(complete_naive)

    jpm = SYMBOLS.index("JPM")
    diff = rcov_naive - rcov_clean
    # JPM-only level check. A cutoff of 23 is not a filter and is not applied to other names.
    jpm_mids = cleaned.quotes.loc[cleaned.quotes["asset"] == "JPM", "midquote"]
    n_jpm_mid_below_23 = int((jpm_mids < 23.0).sum())
    report = {
        "sample": {
            "date": SAMPLE_DATE,
            "library": WRDS_LIBRARY,
            "table": WRDS_TABLE,
            "symbols": SYMBOLS,
            "session": "09:30-16:00 inclusive, seconds after midnight",
        },
        "cleaning": diagnostics_as_dict(cleaned.diagnostics),
        "cleaning_by_stock": per_stock_stage_counts(raw),
        "rows_by_stock": {
            "input": raw.groupby("asset").size().reindex(SYMBOLS).astype(int).to_dict(),
            "cleaned": cleaned.quotes.groupby("asset").size().reindex(SYMBOLS).astype(int).to_dict(),
        },
        "open_convention": (
            "09:30 remains missing if no regular-session quote is available at or "
            "before that grid time. Incomplete opening rows are dropped. They are "
            "not filled from pre-market quotes."
        ),
        "missing_prices": {"naive": missing_naive, "cleaned": missing_clean},
        "n_complete_grid_rows": {
            "naive": int(len(complete_naive)),
            "cleaned": int(len(complete_clean)),
        },
        "n_returns": {
            "naive": int(len(complete_naive) - 1),
            "cleaned": int(len(complete_clean) - 1),
        },
        "jpm_five_minute_midquotes": {
            "naive": jpm_grid_slice(prices_naive),
            "cleaned": jpm_grid_slice(prices_clean),
        },
        "n_jpm_cleaned_midquotes_below_23": n_jpm_mid_below_23,
        "return_sanity_cleaned": return_sanity(returns_clean),
        "rcov_cleaned": matrix_diagnostics(rcov_clean, SYMBOLS),
        "rcov_naive": matrix_diagnostics(rcov_naive, SYMBOLS),
        "naive_vs_cleaned": {
            "frobenius_norm_difference": float(np.linalg.norm(diff, ord="fro")),
            "jpm_realized_variance": {
                "naive": float(rcov_naive[jpm, jpm]),
                "cleaned": float(rcov_clean[jpm, jpm]),
            },
            "jpm_covariance_row": {
                "assets": SYMBOLS,
                "naive": rcov_naive[jpm, :].tolist(),
                "cleaned": rcov_clean[jpm, :].tolist(),
            },
        },
        "caveat": (
            "The cleaned matrix is a realized-covariance proxy, not ground truth "
            "and not a realized-kernel estimator."
        ),
    }
    return report


def _print_report(report: dict[str, object]) -> None:
    clean = report["cleaning"]
    print("WRDS sample", report["sample"])
    print("Cleaning diagnostics (full panel)")
    for key, value in clean.items():
        print(f"  {key}: {value}")
    print("Cleaning by stock")
    for symbol, counts in report["cleaning_by_stock"].items():
        print(
            f"  {symbol}: input={counts['n_input']} "
            f"after_Q4={counts['n_after_q4']} "
            f"removed_Q3={counts['n_removed_q3']} "
            f"removed_Q4={counts['n_removed_q4']}"
        )
    print("JPM five-minute midquotes")
    naive_jpm = report["jpm_five_minute_midquotes"]["naive"]
    clean_jpm = report["jpm_five_minute_midquotes"]["cleaned"]
    for hhmm in JPM_CHECK_TIMES:
        print(f"  {hhmm}: naive={naive_jpm[hhmm]}  cleaned={clean_jpm[hhmm]}")
    print("Missing prices by stock (naive / cleaned)")
    print(" ", report["missing_prices"])
    print("Largest absolute cleaned 5-minute returns")
    for symbol, stats in report["return_sanity_cleaned"]["by_stock"].items():
        print(
            f"  {symbol}: {stats['max_abs_return']:.6g} at {stats['timestamp']} "
            f"|r|>2%={stats['n_abs_gt_2pct']} "
            f"|r|>5%={stats['n_abs_gt_5pct']} "
            f"|r|>10%={stats['n_abs_gt_10pct']}"
        )
    print("Cleaned RCov")
    print(np.array(report["rcov_cleaned"]["rcov"]))
    print("Eigenvalues", report["rcov_cleaned"]["eigenvalues"])
    print("min eig", report["rcov_cleaned"]["min_eigenvalue"])
    print("rank", report["rcov_cleaned"]["numerical_rank"])
    print("PSD", report["rcov_cleaned"]["psd"])
    print("condition", report["rcov_cleaned"]["condition_number"])
    print("symmetry error", report["rcov_cleaned"]["symmetry_error"])
    print("Naive vs cleaned Frobenius", report["naive_vs_cleaned"]["frobenius_norm_difference"])
    print("JPM variance naive/cleaned", report["naive_vs_cleaned"]["jpm_realized_variance"])
    print("JPM row naive", report["naive_vs_cleaned"]["jpm_covariance_row"]["naive"])
    print("JPM row cleaned", report["naive_vs_cleaned"]["jpm_covariance_row"]["cleaned"])


def main() -> dict[str, object]:
    import wrds

    db = wrds.Connection()
    try:
        raw = fetch_nbbo_sample(db)
    finally:
        db.close()
    report = run_pilot(raw)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    _print_report(report)
    print(f"Wrote derived diagnostics to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    main()
