"""Synthetic size sensitivity for the current HAC Diebold-Mariano procedure.

This script does not change the implemented DM statistic, the automatic
Newey-West 1994 lag, or the N(0,1) reference. Demonstration and empirical
forecast losses are not used.

These numbers are Monte Carlo calibration diagnostics, not empirical
benchmark results.
"""

from __future__ import annotations

import time
from pathlib import Path

from covharness.inference.size import (
    CALIBRATION_LAG_RULES,
    CALIBRATION_N_REPS,
    CALIBRATION_RHOS,
    CALIBRATION_SAMPLE_SIZES,
    CALIBRATION_SEED,
    SIZE_ALTERNATIVE_MEAN,
    SIZE_N_OBS,
    SIZE_N_REPS,
    SIZE_RHO,
    SIZE_SEED,
    plot_calibration_bandwidth,
    plot_calibration_size_vs_rho,
    plot_hac_size_power,
    simulate_dm_hac_calibration,
    simulate_hac_size_power,
    write_calibration_table,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLE_PATH = RESULTS / "dm_hac_calibration_sensitivity.csv"
FIGURE_RHO_PATH = RESULTS / "dm_hac_calibration_size_vs_rho.png"
FIGURE_BANDWIDTH_PATH = RESULTS / "dm_hac_calibration_bandwidth.png"
BASELINE_FIGURE_PATH = RESULTS / "dm_hac_size.png"

RECORDED_IID_NULL_NAIVE = 0.0520
RECORDED_IID_NULL_DM = 0.0545
RECORDED_AR_NULL_NAIVE = 0.3185
RECORDED_AR_NULL_DM = 0.1135
RECORDED_ALT_DM = 0.8965
RECORDED_ALT_A_BETTER = 0.9460


def reproduce_historical_baseline() -> None:
    """Re-run the recorded B=2000, T=250, seed=20260912 experiment."""
    result = simulate_hac_size_power()
    observed = (
        result.iid_null_naive_rejection,
        result.iid_null_dm_rejection,
        result.ar_null_naive_rejection,
        result.ar_null_dm_rejection,
        result.alternative_dm_rejection,
        result.alternative_dm_a_better_rejection,
    )
    recorded = (
        RECORDED_IID_NULL_NAIVE,
        RECORDED_IID_NULL_DM,
        RECORDED_AR_NULL_NAIVE,
        RECORDED_AR_NULL_DM,
        RECORDED_ALT_DM,
        RECORDED_ALT_A_BETTER,
    )
    if result.seed != SIZE_SEED or result.n_reps != SIZE_N_REPS:
        raise SystemExit("historical baseline configuration drifted")
    if result.n_obs != SIZE_N_OBS or result.rho != SIZE_RHO:
        raise SystemExit("historical baseline configuration drifted")
    if result.alternative_mean != SIZE_ALTERNATIVE_MEAN:
        raise SystemExit("historical baseline configuration drifted")
    if observed != recorded:
        raise SystemExit(
            "historical baseline rates no longer match the recorded experiment: "
            f"observed={observed} recorded={recorded}"
        )
    plot_hac_size_power(result, BASELINE_FIGURE_PATH)
    print("historical baseline reproduced")
    print(
        f"IID null naive={result.iid_null_naive_rejection:.4f} "
        f"HAC DM={result.iid_null_dm_rejection:.4f}"
    )
    print(
        f"AR(1) rho=0.6 naive={result.ar_null_naive_rejection:.4f} "
        f"HAC DM={result.ar_null_dm_rejection:.4f}"
    )
    print(
        f"IID mean shift {SIZE_ALTERNATIVE_MEAN} two-sided="
        f"{result.alternative_dm_rejection:.4f} "
        f"A-better={result.alternative_dm_a_better_rejection:.4f}"
    )


def run_expanded_grid() -> None:
    """Score the frozen (T, rho, lag-rule) null grid with common random numbers."""
    started = time.perf_counter()
    table = simulate_dm_hac_calibration(
        sample_sizes=CALIBRATION_SAMPLE_SIZES,
        rhos=CALIBRATION_RHOS,
        lag_rules=CALIBRATION_LAG_RULES,
        n_reps=CALIBRATION_N_REPS,
        seed=CALIBRATION_SEED,
    )
    elapsed = time.perf_counter() - started
    write_calibration_table(table, TABLE_PATH)
    plot_calibration_size_vs_rho(table, FIGURE_RHO_PATH)
    plot_calibration_bandwidth(table, FIGURE_BANDWIDTH_PATH)
    print(
        "expanded grid "
        f"B={CALIBRATION_N_REPS} seed={CALIBRATION_SEED} "
        f"rows={len(table)} wall_time_s={elapsed:.2f}"
    )
    print(f"wrote {TABLE_PATH}")
    print(f"wrote {FIGURE_RHO_PATH}")
    print(f"wrote {FIGURE_BANDWIDTH_PATH}")
    auto = table.loc[table["lag_rule"] == "L_auto", ["T", "rho", "hac_lag", "rejection_rate", "mcse", "ci95_lower", "ci95_upper", "mean_estimated_lrv_ratio"]]
    print("automatic-lag size (descriptive only)")
    print(auto.to_string(index=False))


def main() -> None:
    print("These numbers are Monte Carlo calibration diagnostics, not empirical benchmark results.")
    print("The current Bartlett / Newey-West DM default is unchanged.")
    reproduce_historical_baseline()
    run_expanded_grid()


if __name__ == "__main__":
    main()
