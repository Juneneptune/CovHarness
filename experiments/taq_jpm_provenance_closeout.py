"""Block 1 provenance close-out. Identity, lineage, coverage. No measurement changes.

Does not modify Q1-Q4, P3, previous-tick synchronization, RCov, the universe, or
thresholds. Suffix mixing is treated as a hypothesis, not as a diagnosis.
Security identity is taken from TAQ/CRSP metadata, not from the observed price.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from covharness.data.synchronization import previous_tick_sync
from covharness.diagnostics.epps import (
    DEFAULT_FREQUENCIES_MINUTES,
    calendar_session_grid,
    drop_incomplete_open,
)
from experiments.taq_five_stock_epps import CACHE_TICKS, load_or_build_cleaned_stream
from experiments.taq_five_stock_pilot import (
    SAMPLE_DATE,
    SESSION_END_SEC,
    SESSION_START_SEC,
    SYMBOLS,
    WRDS_LIBRARY,
)
from experiments.taq_five_stock_single_exchange_pilot import fetch_cqm_selected_venues

# Down-leg one-minute grid times already listed in docs/PROJECT_STATE.md.
ANOMALOUS_GRID_TIMES = [
    "2009-02-13T10:04:00",
    "2009-02-13T11:01:00",
    "2009-02-13T11:03:00",
    "2009-02-13T12:59:00",
    "2009-02-13T13:02:00",
    "2009-02-13T14:17:00",
    "2009-02-13T15:17:00",
    "2009-02-13T15:24:00",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "taq_jpm_provenance_closeout_20090213.json"
QUOTE_TABLE = f"{WRDS_LIBRARY}.cqm_{SAMPLE_DATE.replace('-', '')}"

PRODUCTION_JPM_PREDICATE = (
    f"sym_root = 'JPM' AND ex IN ('N') "
    f"AND time_m >= {SESSION_START_SEC} AND time_m <= {SESSION_END_SEC}"
)
PRODUCTION_SQL = f"""
SELECT date, time_m, ex, sym_root, bid, ask
FROM {QUOTE_TABLE}
WHERE time_m >= {SESSION_START_SEC}
  AND time_m <= {SESSION_END_SEC}
  AND (sym_root = 'JPM' AND ex IN ('N'))
ORDER BY sym_root, time_m
"""


def _repr_suffix(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "<NULL>"
    text = str(value)
    if text.strip() == "":
        return "<BLANK>"
    return text


def inspect_quote_columns(db) -> list[str]:
    sample = db.raw_sql(f"SELECT * FROM {QUOTE_TABLE} LIMIT 1")
    return list(sample.columns)


def suffix_counts(db, extra_where: str) -> list[dict[str, object]]:
    sql = f"""
        SELECT
            sym_root,
            CAST(sym_suffix AS VARCHAR) AS sym_suffix,
            COUNT(*) AS n_rows
        FROM {QUOTE_TABLE}
        WHERE {extra_where}
        GROUP BY 1, 2
        ORDER BY n_rows DESC
    """
    frame = db.raw_sql(sql)
    out = []
    for _, row in frame.iterrows():
        out.append(
            {
                "sym_root": str(row["sym_root"]),
                "sym_suffix": None if pd.isna(row["sym_suffix"]) else str(row["sym_suffix"]),
                "sym_suffix_repr": _repr_suffix(row["sym_suffix"]),
                "n_rows": int(row["n_rows"]),
            }
        )
    return out


def crsp_jpm_identity(db) -> list[dict[str, object]]:
    sql = f"""
        SELECT ticker, permno, permco, comnam, shrcls, exchcd, ncusip, namedt, nameenddt
        FROM crsp.stocknames
        WHERE ticker = 'JPM'
          AND namedt <= '{SAMPLE_DATE}'
          AND (nameenddt IS NULL OR nameenddt >= '{SAMPLE_DATE}')
        ORDER BY namedt
    """
    frame = db.raw_sql(sql)
    records = []
    for _, row in frame.iterrows():
        records.append(
            {
                "ticker": str(row["ticker"]),
                "permno": int(row["permno"]),
                "permco": int(row["permco"]) if pd.notna(row["permco"]) else None,
                "comnam": str(row["comnam"]),
                "shrcls": None if pd.isna(row["shrcls"]) else str(row["shrcls"]),
                "exchcd": int(row["exchcd"]),
                "ncusip": None if pd.isna(row["ncusip"]) else str(row["ncusip"]),
                "namedt": str(row["namedt"]),
                "nameenddt": None if pd.isna(row["nameenddt"]) else str(row["nameenddt"]),
            }
        )
    return records


def try_taq_master(db) -> dict[str, object]:
    """Daily TAQ master is not assumed to exist in the 2009 sample library."""
    tried = [
        f"{WRDS_LIBRARY}.mastm_{SAMPLE_DATE.replace('-', '')}",
        f"{WRDS_LIBRARY}.master_{SAMPLE_DATE.replace('-', '')}",
        "taqmsec.mastm_20090213",
    ]
    for table in tried:
        try:
            frame = db.raw_sql(
                f"""
                SELECT *
                FROM {table}
                WHERE UPPER(CAST(symbol AS VARCHAR)) LIKE 'JPM%'
                   OR UPPER(CAST(sym_root AS VARCHAR)) = 'JPM'
                LIMIT 20
                """
            )
            return {
                "table": table,
                "available": True,
                "n_rows": int(len(frame)),
                "columns": list(frame.columns),
                "rows": json.loads(frame.to_json(orient="records", date_format="iso")),
            }
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            continue
    return {"available": False, "tables_tried": tried, "last_error": last}


def source_tick_from_production_sync(
    ticks: pd.DataFrame,
    grid_time: pd.Timestamp,
    synced_price: float,
) -> dict[str, object]:
    """Invert previous-tick ffill. The series itself comes from previous_tick_sync."""
    jpm = (
        ticks.loc[ticks["asset"] == "JPM", ["timestamp", "price"]]
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    # Last tick time at or before the grid point. This is the ffill source of previous_tick_sync.
    prior = jpm.loc[jpm["timestamp"] <= grid_time]
    if prior.empty:
        raise RuntimeError(f"no JPM tick at or before {grid_time}")
    source = prior.iloc[-1]
    match = bool(np.isclose(float(source["price"]), float(synced_price), rtol=0.0, atol=1e-12))
    return {
        "grid_time": grid_time.isoformat(),
        "production_synced_price": float(synced_price),
        "source_tick_timestamp": source["timestamp"].isoformat(),
        "source_tick_price": float(source["price"]),
        "matches_production_sync": match,
    }


def fetch_raw_source_rows(
    db,
    columns: list[str],
    source_ts: pd.Timestamp,
) -> list[dict[str, object]]:
    """Raw production-predicate rows at the selected cleaned-tick timestamp."""
    wanted = [
        "date",
        "time_m",
        "ex",
        "sym_root",
        "sym_suffix",
        "bid",
        "ask",
        "bidsiz",
        "asksiz",
        "qu_cond",
        "qu_seqnum",
        "seqnum",
        "qseq",
        "qu_cancel",
        "natbbo_ind",
        "nasdbbo_ind",
        "qu_source",
    ]
    present = [col for col in wanted if col in columns]
    # Match the exact millisecond timestamp used by the cleaned tick.
    seconds = (
        source_ts.hour * 3600
        + source_ts.minute * 60
        + source_ts.second
        + source_ts.microsecond / 1e6
    )
    sql = f"""
        SELECT {", ".join(present)}
        FROM {QUOTE_TABLE}
        WHERE {PRODUCTION_JPM_PREDICATE}
          AND ABS(time_m - {seconds}) < 1e-6
        ORDER BY time_m
    """
    frame = db.raw_sql(sql)
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    for rec in records:
        rec["sym_suffix_repr"] = _repr_suffix(rec.get("sym_suffix"))
        rec["midquote"] = (
            None
            if rec.get("bid") is None or rec.get("ask") is None
            else float(rec["bid"]) + (float(rec["ask"]) - float(rec["bid"])) / 2.0
        )
    return records


def coverage_from_production_sync(ticks: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for delta in DEFAULT_FREQUENCIES_MINUTES:
        grid = calendar_session_grid(SAMPLE_DATE, delta)
        # Production previous-tick, then the existing incomplete-open drop.
        prices = previous_tick_sync(ticks, grid).reindex(columns=SYMBOLS)
        incomplete = prices.isna().any(axis=1)
        complete = drop_incomplete_open(prices)
        first_c = complete.index[0]
        last_c = complete.index[-1]
        internal = prices.loc[first_c:last_c].index[
            prices.loc[first_c:last_c].isna().any(axis=1)
        ]
        returns_index = complete.index[1:]
        rows.append(
            {
                "interval_minutes": delta,
                "nominal_first_grid_time": grid[0].isoformat(),
                "nominal_last_grid_time": grid[-1].isoformat(),
                "n_nominal_grid_times": int(len(grid)),
                "n_incomplete_grid_times": int(incomplete.sum()),
                "incomplete_grid_times": [
                    ts.isoformat() for ts in prices.index[incomplete]
                ],
                "first_complete_price_time": first_c.isoformat(),
                "last_complete_price_time": last_c.isoformat(),
                "n_complete_prices": int(len(complete)),
                "first_return_interval_end": returns_index[0].isoformat(),
                "last_return_interval_end": returns_index[-1].isoformat(),
                "n_returns": int(len(returns_index)),
                "internal_missing_timestamps": [ts.isoformat() for ts in internal],
                "n_internal_missing": int(len(internal)),
            }
        )
    return rows


def existing_epps_matrix_diagnostics() -> list[dict[str, object]]:
    payload = json.loads(
        (REPO_ROOT / "results" / "taq_five_stock_epps_20090213.json").read_text()
    )
    out = []
    for item in payload["by_frequency"]:
        out.append(
            {
                "interval_minutes": item["interval_minutes"],
                "n_returns": item["n_returns"],
                "numerical_rank": item["numerical_rank"],
                "min_eigenvalue": item["min_eigenvalue"],
                "psd": item["psd"],
                "condition_number": item["condition_number"],
                "source": (
                    "results/taq_five_stock_epps_20090213.json, populated by "
                    "covharness.diagnostics.epps.matrix_eigen_diagnostics"
                ),
            }
        )
    return out


def main() -> dict[str, object]:
    meta = load_or_build_cleaned_stream()
    ticks = meta["ticks"]
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"])

    grid = calendar_session_grid(SAMPLE_DATE, 1)
    # Exact Epps one-minute series. Production previous_tick_sync, then dropna.
    synced = previous_tick_sync(ticks, grid).reindex(columns=SYMBOLS)
    complete = drop_incomplete_open(synced)

    import wrds

    db = wrds.Connection()
    try:
        columns = inspect_quote_columns(db)
        suffixes_production = suffix_counts(db, PRODUCTION_JPM_PREDICATE)
        suffixes_all_ex = suffix_counts(
            db,
            f"sym_root = 'JPM' AND time_m >= {SESSION_START_SEC} "
            f"AND time_m <= {SESSION_END_SEC}",
        )
        crsp = crsp_jpm_identity(db)
        master = try_taq_master(db)

        lineage = []
        for stamp in ANOMALOUS_GRID_TIMES:
            grid_time = pd.Timestamp(stamp)
            if grid_time not in complete.index:
                lineage.append({"grid_time": stamp, "error": "not on complete one-minute grid"})
                continue
            source = source_tick_from_production_sync(
                ticks, grid_time, float(complete.loc[grid_time, "JPM"])
            )
            source["raw_rows_at_source_timestamp"] = fetch_raw_source_rows(
                db, columns, pd.Timestamp(source["source_tick_timestamp"])
            )
            suffixes_at_source = sorted(
                {
                    _repr_suffix(row.get("sym_suffix"))
                    for row in source["raw_rows_at_source_timestamp"]
                }
            )
            source["distinct_suffixes_at_source_timestamp"] = suffixes_at_source
            source["all_raw_rows_blank_or_null_suffix"] = all(
                _repr_suffix(row.get("sym_suffix")) in {"<BLANK>", "<NULL>"}
                for row in source["raw_rows_at_source_timestamp"]
            )
            lineage.append(source)
    finally:
        db.close()

    report = {
        "production_extraction": {
            "function": "experiments.taq_five_stock_single_exchange_pilot.fetch_cqm_selected_venues",
            "table": QUOTE_TABLE,
            "jpm_predicate": PRODUCTION_JPM_PREDICATE,
            "sql": PRODUCTION_SQL.strip(),
            "note": (
                "sym_root is restricted to JPM and ex to NYSE N. "
                "sym_suffix is not in SELECT and is not in WHERE."
            ),
            "quote_table_columns": columns,
        },
        "suffix_counts_production_predicate": suffixes_production,
        "suffix_counts_jpm_all_exchanges_session": suffixes_all_ex,
        "crsp_stocknames_jpm_on_date": crsp,
        "taq_master": master,
        "identity_rule": (
            "Daily TAQ Client Specification Appendix B. A blank suffix is the "
            "ordinary/common share. Preferred, warrant, right, unit, and class "
            "shares carry an explicit suffix (PR, WS, RT, U, A, ...). "
            "Master-file security type A is Common Stock. Identity is not inferred "
            "from the quote price."
        ),
        "lineage": {
            "production_sync": "covharness.data.synchronization.previous_tick_sync",
            "anomalous_grid_times": ANOMALOUS_GRID_TIMES,
            "events": lineage,
        },
        "coverage": coverage_from_production_sync(ticks),
        "existing_epps_matrix_diagnostics": existing_epps_matrix_diagnostics(),
        "repository_checks": {
            "benchmark_plan_exists": (REPO_ROOT / "BENCHMARK_IMPLEMENTATION_PLAN.md").exists(),
            "old_demo_plan_exists": (REPO_ROOT / "DEMO_IMPLEMENTATION_PLAN.md").exists(),
        },
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(
        {
            "suffix_production": suffixes_production,
            "suffix_all_ex": suffixes_all_ex,
            "crsp": crsp,
            "master_available": master.get("available"),
            "columns": columns,
            "n_lineage": len(lineage),
            "lineage_match": [e.get("matches_production_sync") for e in lineage],
            "suffixes_at_events": [e.get("distinct_suffixes_at_source_timestamp") for e in lineage],
            "coverage": report["coverage"],
            "plans": report["repository_checks"],
        },
        indent=2,
        default=str,
    ))
    print("Wrote", RESULTS_PATH)
    return report


if __name__ == "__main__":
    main()
