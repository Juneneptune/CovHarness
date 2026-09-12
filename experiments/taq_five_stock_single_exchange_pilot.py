"""Five-stock paper-style single-exchange measurement panel.

Looks up CRSP listing venues, pulls ``cqm_20090213`` quotes from the selected
TAQ exchange, applies P3 then the existing Q1-Q4 cleaner, and forms a
five-minute realized covariance. Consolidated NBBO remains a labeled
alternative path and is not deleted.

Raw ticks are not written to the repository.
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
    ListingVenue,
    is_ordinary_taq_suffix,
    listing_venue_from_crsp_exchcd,
    taq_suffix_matches,
    taq_suffix_sql_predicate,
)
from covharness.realized.daily import daily_realized_covariance
from experiments.taq_five_stock_pilot import (
    SAMPLE_DATE,
    SESSION_END_SEC,
    SESSION_START_SEC,
    SYMBOLS,
    WRDS_LIBRARY,
    diagnostics_as_dict,
    drop_incomplete_open,
    five_minute_grid,
    matrix_diagnostics,
    return_sanity,
    synchronize_panel,
)

RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "taq_five_stock_single_exchange_20090213.json"
)
NBBO_ALT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "taq_five_stock_pilot_20090213.json"
)


def lookup_listing_venues(db, symbols: list[str], day: str = SAMPLE_DATE) -> dict[str, ListingVenue]:
    """CRSP stocknames listing exchange on ``day``, mapped to TAQ ``ex`` codes."""
    symbols_sql = ", ".join(f"'{s}'" for s in symbols)
    sql = f"""
        SELECT ticker, permno, exchcd, namedt, nameenddt
        FROM crsp.stocknames
        WHERE ticker IN ({symbols_sql})
          AND namedt <= '{day}'
          AND (nameenddt IS NULL OR nameenddt >= '{day}')
        ORDER BY ticker, namedt
    """
    names = db.raw_sql(sql)
    if names.empty:
        raise RuntimeError(f"no CRSP stocknames rows for {symbols} on {day}")

    # One listing venue per ticker from the name-history row that covers the date.
    venues: dict[str, ListingVenue] = {}
    for ticker, group in names.groupby("ticker"):
        row = group.iloc[-1]
        # Ordinary/common share. Preferred and class suffixes are other securities.
        venues[str(ticker)] = listing_venue_from_crsp_exchcd(
            str(ticker),
            int(row["exchcd"]),
            taq_sym_suffix=TAQ_ORDINARY_SHARE_SUFFIX,
        )
    missing = [s for s in symbols if s not in venues]
    if missing:
        raise RuntimeError(f"CRSP listing venue missing for {missing}")
    return venues


def fetch_cqm_selected_venues(
    db,
    venues: dict[str, ListingVenue],
) -> pd.DataFrame:
    """Exchange-level quotes for the P3 venue and requested TAQ suffix.

    Session hours are remote. Security identity is ``sym_root`` plus
    ``sym_suffix``, not the root ticker alone.
    """
    # Restrict remotely to each stock's venue, suffix identity, and session hours.
    clauses = []
    for symbol, venue in venues.items():
        codes = ", ".join(f"'{c}'" for c in venue.taq_ex_codes)
        suffix_pred = taq_suffix_sql_predicate("sym_suffix", venue.taq_sym_suffix)
        clauses.append(
            f"(sym_root = '{symbol}' AND ex IN ({codes}) AND {suffix_pred})"
        )
    where_venues = " OR ".join(clauses)
    sql = f"""
        SELECT date, time_m, ex, sym_root, sym_suffix, bid, ask
        FROM {WRDS_LIBRARY}.cqm_{SAMPLE_DATE.replace('-', '')}
        WHERE time_m >= {SESSION_START_SEC}
          AND time_m <= {SESSION_END_SEC}
          AND ({where_venues})
        ORDER BY sym_root, time_m
    """
    raw = db.raw_sql(sql)
    # Exchange-local timestamp from calendar date plus seconds after midnight.
    raw["timestamp"] = pd.to_datetime(raw["date"]) + pd.to_timedelta(raw["time_m"], unit="s")
    raw = raw.rename(columns={"sym_root": "asset"})
    raw["bid"] = pd.to_numeric(raw["bid"], errors="raise")
    raw["ask"] = pd.to_numeric(raw["ask"], errors="raise")
    out = raw.loc[:, ["timestamp", "asset", "ex", "sym_suffix", "bid", "ask"]].copy()
    # Reject any non-matching suffix that survived the remote predicate.
    for symbol, venue in venues.items():
        subset = out.loc[out["asset"] == symbol, "sym_suffix"]
        bad = ~subset.map(lambda value: taq_suffix_matches(value, venue.taq_sym_suffix))
        if bool(bad.any()):
            raise RuntimeError(
                f"non-matching TAQ suffixes entered the extract for {symbol}"
            )
    return out


def run_single_exchange_panel(
    raw: pd.DataFrame,
    venues: dict[str, ListingVenue],
) -> dict[str, object]:
    grid = five_minute_grid()
    # P3 then existing Q1-Q4. Midquote is formed inside the cleaner.
    cleaned = clean_single_exchange_quotes(raw, venues)
    ticks = cleaned_quotes_as_ticks(cleaned.quotes)
    # Previous-tick onto the five-minute grid. Do not fill 09:30 from pre-market.
    prices = synchronize_panel(ticks, grid)
    missing = prices.isna().sum().astype(int).to_dict()
    complete = drop_incomplete_open(prices)
    returns = synchronized_log_returns(complete)
    rcov = daily_realized_covariance(complete)

    # Pack venue choices and remaining-row counts for the report.
    venue_report = {
        asset: {
            "crsp_exchcd": v.crsp_exchcd,
            "crsp_exchange_name": v.crsp_exchange_name,
            "taq_ex_codes": list(v.taq_ex_codes),
            "taq_sym_suffix": v.taq_sym_suffix,
            "source": v.source,
        }
        for asset, v in venues.items()
    }
    rows_input = raw.groupby("asset").size().reindex(SYMBOLS).fillna(0).astype(int).to_dict()
    rows_clean = (
        cleaned.quotes.groupby("asset").size().reindex(SYMBOLS).fillna(0).astype(int).to_dict()
    )
    # Audit that only ordinary-share suffixes remain in the extract.
    suffix_audit = {
        symbol: {
            "n_rows": int((raw["asset"] == symbol).sum()),
            "n_ordinary_suffix": int(
                raw.loc[raw["asset"] == symbol, "sym_suffix"].map(is_ordinary_taq_suffix).sum()
            ),
            "n_nonblank_suffix": int(
                (~raw.loc[raw["asset"] == symbol, "sym_suffix"].map(is_ordinary_taq_suffix)).sum()
            ),
        }
        for symbol in SYMBOLS
    }
    return {
        "path": "paper_style_single_exchange",
        "sample": {
            "date": SAMPLE_DATE,
            "quote_table": f"{WRDS_LIBRARY}.cqm_{SAMPLE_DATE.replace('-', '')}",
            "symbols": SYMBOLS,
        },
        "selected_venues": venue_report,
        "p3": {
            "n_before_p3": cleaned.n_before_p3,
            "n_after_p3": cleaned.n_after_p3,
            "n_removed_p3": cleaned.n_removed_p3,
        },
        "q1q4": diagnostics_as_dict(cleaned.diagnostics),
        "rows_by_stock": {"after_sql_p3": rows_input, "after_q4": rows_clean},
        "suffix_audit": suffix_audit,
        "missing_prices": missing,
        "n_complete_grid_rows": int(len(complete)),
        "n_returns": int(len(complete) - 1),
        "return_sanity": return_sanity(returns),
        "rcov": matrix_diagnostics(rcov, SYMBOLS),
        "open_convention": (
            "09:30 remains missing if no regular-session quote is available at or "
            "before that grid time. Incomplete opening rows are dropped."
        ),
        "alternative_path": (
            "consolidated NBBO via clean_nbbo_quotes remains available. "
            "See results/taq_five_stock_pilot_20090213.json."
        ),
        "caveat": (
            "The single-exchange matrix is a realized-covariance proxy, not ground "
            "truth and not a realized-kernel estimator."
        ),
    }


def _print_report(report: dict[str, object]) -> None:
    print("Selected venues")
    for asset, info in report["selected_venues"].items():
        print(
            f"  {asset}: CRSP {info['crsp_exchcd']} "
            f"{info['crsp_exchange_name']} -> TAQ {info['taq_ex_codes']} "
            f"suffix={info['taq_sym_suffix']!r}"
        )
    print("P3", report["p3"])
    print("Q1-Q4", report["q1q4"])
    print("Rows by stock", report["rows_by_stock"])
    print("Suffix audit", report["suffix_audit"])
    print("Missing prices", report["missing_prices"])
    print("Largest absolute 5-minute returns")
    for symbol, stats in report["return_sanity"]["by_stock"].items():
        print(
            f"  {symbol}: {stats['max_abs_return']:.6g} at {stats['timestamp']} "
            f"|r|>2%={stats['n_abs_gt_2pct']}"
        )
    print("RCov")
    print(np.array(report["rcov"]["rcov"]))
    print("eigenvalues", report["rcov"]["eigenvalues"])
    print("min eig", report["rcov"]["min_eigenvalue"])
    print("rank", report["rcov"]["numerical_rank"])
    print("PSD", report["rcov"]["psd"])
    print("condition", report["rcov"]["condition_number"])
    print("symmetry error", report["rcov"]["symmetry_error"])
    if "nbbo_alternative" in report:
        alt = report["nbbo_alternative"]
        print("NBBO alternative JPM variance", alt.get("jpm_realized_variance"))
        print("NBBO alternative max |r| JPM", alt.get("jpm_max_abs_return"))


def main() -> dict[str, object]:
    import wrds

    db = wrds.Connection()
    try:
        # CRSP listing venue, then exchange-level quotes from that venue only.
        venues = lookup_listing_venues(db, SYMBOLS)
        raw = fetch_cqm_selected_venues(db, venues)
    finally:
        db.close()

    report = run_single_exchange_panel(raw, venues)
    # Attach the saved NBBO alternative diagnostics. Do not overwrite that path.
    if NBBO_ALT_PATH.exists():
        nbbo = json.loads(NBBO_ALT_PATH.read_text())
        report["nbbo_alternative"] = {
            "label": "consolidated_nbbo_q1q4",
            "table": nbbo["sample"]["table"],
            "jpm_five_minute_midquotes": nbbo["jpm_five_minute_midquotes"]["cleaned"],
            "jpm_realized_variance": nbbo["naive_vs_cleaned"]["jpm_realized_variance"]["cleaned"],
            "jpm_max_abs_return": nbbo["return_sanity_cleaned"]["by_stock"]["JPM"]["max_abs_return"],
            "rcov_condition_number": nbbo["rcov_cleaned"]["condition_number"],
        }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    _print_report(report)
    print("Wrote", RESULTS_PATH)
    return report


if __name__ == "__main__":
    main()
