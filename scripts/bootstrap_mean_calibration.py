"""Synthetic size calibration of the recentered bootstrap mean companion.

Conventional Bartlett / Newey-West Diebold-Mariano is not modified here.
These numbers are Monte Carlo calibration diagnostics, not empirical
benchmark results. The companion is not adopted by running this script.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from covharness.inference.bootstrap_mean import (
    PAIRWISE_CALIBRATION_DGP_SEED,
    STAGE1_N_BOOT,
    STAGE1_OUTER,
    STAGE2_CELLS,
    STAGE2_N_BOOT,
    STAGE2_OUTER,
    pairwise_block_length,
    plot_bootstrap_mean_size_vs_rho,
    plot_nw_versus_bootstrap_mean,
    simulate_bootstrap_mean_size,
    write_bootstrap_mean_table,
)
from covharness.inference.size import CALIBRATION_RHOS, CALIBRATION_SAMPLE_SIZES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGE1_PATH = RESULTS / "bootstrap_mean_calibration_stage1.csv"
STAGE2_PATH = RESULTS / "bootstrap_mean_calibration_stage2.csv"
NW_PATH = RESULTS / "dm_hac_calibration_sensitivity.csv"
FIGURE_STAGE1 = RESULTS / "bootstrap_mean_calibration_size_vs_rho.png"
FIGURE_COMPARE = RESULTS / "bootstrap_mean_versus_nw_size.png"

STAGE1_CELLS = tuple(
    (int(n_obs), float(rho))
    for n_obs in CALIBRATION_SAMPLE_SIZES
    for rho in CALIBRATION_RHOS
)


def run_stage1() -> pd.DataFrame:
    """Score all 15 frozen (T, rho) cells with M1=200 and B1=499."""
    started = time.perf_counter()
    table = simulate_bootstrap_mean_size(
        cells=STAGE1_CELLS,
        n_outer=STAGE1_OUTER,
        n_boot=STAGE1_N_BOOT,
        stage="stage1",
        dgp_seed=PAIRWISE_CALIBRATION_DGP_SEED,
    )
    elapsed = time.perf_counter() - started
    write_bootstrap_mean_table(table, STAGE1_PATH)
    plot_bootstrap_mean_size_vs_rho(table, FIGURE_STAGE1)
    print(f"stage 1 rows={len(table)} wall_time_s={elapsed:.2f}")
    print(table.to_string(index=False))
    return table


def timed_stage2_pilot() -> float:
    """Time one pre-specified Stage-2 cell and extrapolate the full Stage-2 cost."""
    n_obs, rho = STAGE2_CELLS[1]
    n_pilot = 20
    started = time.perf_counter()
    simulate_bootstrap_mean_size(
        cells=((n_obs, rho),),
        n_outer=n_pilot,
        n_boot=STAGE2_N_BOOT,
        stage="pilot",
        dgp_seed=PAIRWISE_CALIBRATION_DGP_SEED,
    )
    elapsed = time.perf_counter() - started
    per_rep = elapsed / n_pilot
    # Extrapolate using the actual Stage-2 (T, B) mix, scaling T linearly with the index loop.
    estimate = 0.0
    for cell_t, _ in STAGE2_CELLS:
        estimate += STAGE2_OUTER * per_rep * (cell_t / n_obs)
    print(
        f"stage 2 pilot T={n_obs} rho={rho} n_pilot={n_pilot} "
        f"B={STAGE2_N_BOOT} elapsed_s={elapsed:.2f} "
        f"estimated_full_s={estimate:.1f}"
    )
    return estimate


def run_stage2() -> pd.DataFrame:
    """Score the 10 pre-specified Stage-2 cells with M2=1000 and B2=1999."""
    started = time.perf_counter()
    table = simulate_bootstrap_mean_size(
        cells=STAGE2_CELLS,
        n_outer=STAGE2_OUTER,
        n_boot=STAGE2_N_BOOT,
        stage="stage2",
        dgp_seed=PAIRWISE_CALIBRATION_DGP_SEED,
    )
    elapsed = time.perf_counter() - started
    write_bootstrap_mean_table(table, STAGE2_PATH)
    print(f"stage 2 rows={len(table)} wall_time_s={elapsed:.2f}")
    print(table.to_string(index=False))
    return table


def write_comparison_figure(stage2: pd.DataFrame) -> None:
    """Overlay stored automatic-NW size with Stage-2 bootstrap-mean size."""
    nw = pd.read_csv(NW_PATH)
    plot_nw_versus_bootstrap_mean(stage2, nw, FIGURE_COMPARE)
    merged = stage2.merge(
        nw.loc[nw["lag_rule"] == "L_auto", ["T", "rho", "rejection_rate"]].rename(
            columns={"rejection_rate": "nw_automatic_rejection_rate"}
        ),
        on=["T", "rho"],
        how="left",
    )
    print("stage 2 versus stored automatic NW (descriptive only)")
    print(
        merged[
            [
                "T",
                "rho",
                "block_length",
                "rejection_rate",
                "nw_automatic_rejection_rate",
                "mcse",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    print("These numbers are Monte Carlo calibration diagnostics, not empirical benchmark results.")
    print("The conventional Bartlett / Newey-West DM default is unchanged.")
    print("The companion is not adopted by this script.")
    print(
        "block lengths "
        + ", ".join(f"T={t}->{pairwise_block_length(t)}" for t in (250, 500, 1000))
    )
    run_stage1()
    estimate = timed_stage2_pilot()
    if estimate > 5400:
        raise SystemExit(
            f"stage 2 estimated wall time {estimate:.0f}s exceeds a 90-minute budget. "
            "M2 and B2 were not reduced. Stopped after the timed pilot."
        )
    stage2 = run_stage2()
    write_comparison_figure(stage2)


if __name__ == "__main__":
    main()
