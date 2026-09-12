"""Epps-effect scan on the five-stock single-exchange measurement path.

Reuses the validated P3 venue lookup and ``cqm_20090213`` extract, then the
existing Q1-Q4 cleaner, once. Realized covariance at each frequency is the
production unscaled Gram matrix. Q1-Q4, the venue adapter, and the
realized-covariance formula are not modified.

Raw licensed ticks are not written to the repository. Derived diagnostics
and the Epps figure may be written to ``results/``. A local cleaned-midquote
cache under ``data/cache/`` avoids a second WRDS pull when it already exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from covharness.data.exchange_quotes import clean_single_exchange_quotes
from covharness.data.quotes import cleaned_quotes_as_ticks
from covharness.data.venues import ListingVenue, is_ordinary_taq_suffix
from covharness.diagnostics.epps import (
    DEFAULT_FREQUENCIES_MINUTES,
    epps_across_frequencies,
    epps_curve_to_dict,
    plot_epps_curve,
)
from experiments.taq_five_stock_pilot import (
    SAMPLE_DATE,
    SYMBOLS,
    diagnostics_as_dict,
)
from experiments.taq_five_stock_single_exchange_pilot import (
    fetch_cqm_selected_venues,
    lookup_listing_venues,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = REPO_ROOT / "results" / "taq_five_stock_epps_20090213.json"
RESULTS_FIGURE = REPO_ROOT / "results" / "taq_five_stock_epps_20090213.png"
CACHE_DIR = REPO_ROOT / "data" / "cache"
CACHE_TICKS = CACHE_DIR / "taq_five_stock_single_exchange_cleaned_20090213.pkl"
CACHE_META = CACHE_DIR / "taq_five_stock_single_exchange_cleaning_20090213.json"


def per_stock_stage_counts(
    raw: pd.DataFrame,
    venues: dict[str, ListingVenue],
) -> dict[str, dict[str, float | int]]:
    """Stage counts on the same P3-then-Q1-Q4 adapter, one stock at a time."""
    counts: dict[str, dict[str, float | int]] = {}
    for symbol in SYMBOLS:
        subset = raw.loc[raw["asset"] == symbol]
        cleaned = clean_single_exchange_quotes(subset, {symbol: venues[symbol]})
        counts[symbol] = {
            "n_before_p3": cleaned.n_before_p3,
            "n_after_p3": cleaned.n_after_p3,
            "n_removed_p3": cleaned.n_removed_p3,
            **diagnostics_as_dict(cleaned.diagnostics),
        }
    return counts


def _venue_report(venues: dict[str, ListingVenue]) -> dict[str, dict[str, object]]:
    return {
        asset: {
            "crsp_exchcd": venue.crsp_exchcd,
            "crsp_exchange_name": venue.crsp_exchange_name,
            "taq_ex_codes": list(venue.taq_ex_codes),
            "taq_sym_suffix": venue.taq_sym_suffix,
            "source": venue.source,
        }
        for asset, venue in venues.items()
    }


def load_or_build_cleaned_stream() -> dict[str, object]:
    """Reuse a local cleaned-midquote cache, otherwise fetch and clean once."""
    if CACHE_TICKS.exists() and CACHE_META.exists():
        ticks = pd.read_pickle(CACHE_TICKS)
        meta = json.loads(CACHE_META.read_text())
        meta["ticks"] = ticks
        meta["cache_used"] = True
        return meta

    import wrds

    db = wrds.Connection()
    try:
        # Same CRSP listing-venue map and cqm extract as the validated panel.
        venues = lookup_listing_venues(db, SYMBOLS)
        raw = fetch_cqm_selected_venues(db, venues)
    finally:
        db.close()

    # One panel clean for the Epps matrices. Per-stock counts are diagnostic.
    cleaned = clean_single_exchange_quotes(raw, venues)
    ticks = cleaned_quotes_as_ticks(cleaned.quotes)
    by_stock = per_stock_stage_counts(raw, venues)
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
    meta: dict[str, object] = {
        "selected_venues": _venue_report(venues),
        "p3": {
            "n_before_p3": cleaned.n_before_p3,
            "n_after_p3": cleaned.n_after_p3,
            "n_removed_p3": cleaned.n_removed_p3,
        },
        "q1q4": diagnostics_as_dict(cleaned.diagnostics),
        "cleaning_by_stock": by_stock,
        "rows_by_stock": {
            "after_sql_p3": {
                symbol: int(n)
                for symbol, n in raw.groupby("asset").size().reindex(SYMBOLS).fillna(0).items()
            },
            "after_q4": {
                symbol: int(n)
                for symbol, n in cleaned.quotes.groupby("asset").size().reindex(SYMBOLS).fillna(0).items()
            },
        },
        "suffix_audit": suffix_audit,
        "cache_used": False,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ticks.to_pickle(CACHE_TICKS)
    CACHE_META.write_text(json.dumps({k: v for k, v in meta.items() if k != "cache_used"}, indent=2))
    meta["ticks"] = ticks
    return meta


def build_report(meta: dict[str, object]) -> dict[str, object]:
    ticks = meta["ticks"]
    # Previous-tick RCov at each requested width, same cleaned midquotes.
    curve = epps_across_frequencies(
        ticks,
        day=SAMPLE_DATE,
        assets=SYMBOLS,
        frequencies_minutes=DEFAULT_FREQUENCIES_MINUTES,
    )
    figure_path = plot_epps_curve(
        curve,
        RESULTS_FIGURE,
        title="Five-stock realized correlations by sampling interval, 13 February 2009",
    )
    curve_payload = epps_curve_to_dict(curve)
    return {
        "path": "paper_style_single_exchange",
        "sample": {
            "date": SAMPLE_DATE,
            "quote_table": "taqmsamp_all.cqm_20090213",
            "symbols": SYMBOLS,
            "cache_used": meta["cache_used"],
        },
        "selected_venues": meta["selected_venues"],
        "p3": meta["p3"],
        "q1q4": meta["q1q4"],
        "cleaning_by_stock": meta["cleaning_by_stock"],
        "rows_by_stock": meta["rows_by_stock"],
        "suffix_audit": meta.get("suffix_audit", {}),
        "open_convention": (
            "09:30 remains missing if no regular-session quote is available at or "
            "before that grid time. Incomplete opening rows are dropped. They are "
            "not filled from pre-market quotes."
        ),
        "frequencies_minutes": curve_payload["frequencies_minutes"],
        "mean_off_diagonal_correlation": curve_payload["mean_off_diagonal_correlation"],
        "median_off_diagonal_correlation": curve_payload["median_off_diagonal_correlation"],
        "by_frequency": curve_payload["by_frequency"],
        "figure": str(figure_path.relative_to(REPO_ROOT)),
        "caveat": (
            "This is a one-day demonstration on five names. It is not an estimate "
            "of the magnitude of the Epps effect in U.S. equities. No sampling "
            "frequency is treated as the true covariance."
        ),
    }


def _print_report(report: dict[str, object]) -> None:
    print("Selected venues")
    for asset, info in report["selected_venues"].items():
        print(
            f"  {asset}: CRSP {info['crsp_exchcd']} "
            f"{info['crsp_exchange_name']} -> TAQ {info['taq_ex_codes']}"
        )
    print("Panel Q1-Q4", report["q1q4"])
    print("Suffix audit", report.get("suffix_audit", {}))
    print("Cleaning by stock")
    for symbol, counts in report["cleaning_by_stock"].items():
        print(
            f"  {symbol}: after_P3={counts['n_after_p3']} "
            f"after_Q1={counts['n_after_q1']} "
            f"after_Q2={counts['n_after_q2']} "
            f"after_Q3={counts['n_after_q3']} "
            f"after_Q4={counts['n_after_q4']} "
            f"removed_Q1={counts['n_removed_q1']} "
            f"removed_Q2={counts['n_removed_q2']} "
            f"removed_Q3={counts['n_removed_q3']} "
            f"removed_Q4={counts['n_removed_q4']}"
        )
    print("Epps curve (minutes, n_returns, mean/median/min/max off-diagonal corr)")
    for item in report["by_frequency"]:
        print(
            f"  {item['interval_minutes']:>2} min  n={item['n_returns']:<4}  "
            f"mean={item['mean_off_diagonal_correlation']:.4f}  "
            f"median={item['median_off_diagonal_correlation']:.4f}  "
            f"min={item['min_off_diagonal_correlation']:.4f}  "
            f"max={item['max_off_diagonal_correlation']:.4f}  "
            f"rank={item['numerical_rank']}  "
            f"min_eig={item['min_eigenvalue']:.3e}  "
            f"psd={item['psd']}  "
            f"cond={item['condition_number']}"
        )
    print("Pairwise correlations by frequency")
    for item in report["by_frequency"]:
        pairs = ", ".join(
            f"{key}={value:.4f}" for key, value in item["pairwise_correlations"].items()
        )
        print(f"  {item['interval_minutes']:>2} min  {pairs}")


def main() -> dict[str, object]:
    meta = load_or_build_cleaned_stream()
    report = build_report(meta)
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(report, indent=2))
    _print_report(report)
    print("Wrote", RESULTS_JSON)
    print("Wrote", RESULTS_FIGURE)
    return report


if __name__ == "__main__":
    main()
