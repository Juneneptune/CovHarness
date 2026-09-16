"""Recentered stationary-bootstrap test of a pairwise mean loss differential.

This is a candidate companion to Bartlett / Newey-West Diebold-Mariano. It is
not the existing DM statistic. It does not estimate a long-run variance and
does not use a normal reference. SPA and MCS remain the multiple-model
procedures. Their composite-null p-value conventions are not copied here.

For ``d_t = L_{A,t} - L_{B,t}`` the null is ``E[d_t] = 0``. The observed
statistic is ``sqrt(T) * mean(d)``. Stationary-bootstrap samples are drawn
from the recentered series ``d_t - mean(d)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from covharness.inference.bootstrap import (
    default_block_length,
    stationary_bootstrap_indices,
)
from covharness.inference.differentials import (
    ALTERNATIVE_A_BETTER,
    ALTERNATIVE_B_BETTER,
    ALTERNATIVE_TWO_SIDED,
    as_1d_finite,
    parse_alternative,
)
from covharness.inference.exceptions import DegenerateLossDifferentialError
from covharness.inference.size import (
    SIZE_ALPHA,
    monte_carlo_interval,
    monte_carlo_standard_error,
    stationary_ar1_paths,
)

PAIRWISE_BOOTSTRAP_SEED = 20260917
PAIRWISE_N_BOOT = 5000
PAIRWISE_METHOD = "stationary_bootstrap_mean"
PAIRWISE_CALIBRATION_DGP_SEED = 20260918
STAGE1_OUTER = 200
STAGE1_N_BOOT = 499
STAGE2_OUTER = 1000
STAGE2_N_BOOT = 1999
STAGE2_CELLS = (
    (250, 0.0),
    (250, 0.6),
    (250, 0.8),
    (250, 0.9),
    (500, 0.6),
    (500, 0.8),
    (500, 0.9),
    (1000, 0.6),
    (1000, 0.8),
    (1000, 0.9),
)
BOOTSTRAP_MEAN_TABLE_COLUMNS = (
    "stage",
    "T",
    "rho",
    "outer_replications",
    "inner_bootstraps",
    "block_length",
    "nominal_alpha",
    "valid_replications",
    "failures",
    "rejection_rate",
    "mcse",
    "ci95_lower",
    "ci95_upper",
)


@dataclass(frozen=True)
class StationaryBootstrapMeanResult:
    """Pairwise recentered stationary-bootstrap mean test. Not Diebold-Mariano."""

    statistic: float
    p_value: float
    alternative: str
    sample_size: int
    mean_differential: float
    n_boot: int
    block_length: int
    seed: int
    method: str = PAIRWISE_METHOD


def pairwise_block_length(n_observations: int) -> int:
    """Return the frozen confirmatory block length ``max(2, floor(T**(1/3)))``."""
    return default_block_length(n_observations)


def null_centered_differential(differential: ArrayLike) -> NDArray[np.floating]:
    """Return ``d_t - mean(d)``. The input is copied and not mutated."""
    series = as_1d_finite(differential, "differential")
    return series - float(series.mean())


def bootstrap_mean_statistics(
    centered: ArrayLike,
    indices: ArrayLike,
) -> NDArray[np.floating]:
    """Return ``sqrt(T) * mean(d0_star)`` for each bootstrap index row."""
    series = as_1d_finite(centered, "centered")
    index_array = np.asarray(indices)
    if index_array.ndim != 2 or index_array.shape[1] != series.shape[0]:
        raise ValueError(
            "indices must have shape (n_boot, T) matching the centered series"
        )
    n_obs = series.shape[0]
    star_means = series[index_array].mean(axis=1)
    return np.sqrt(n_obs) * star_means


def finite_b_pvalue(
    observed: float,
    stars: ArrayLike,
    alternative: str,
    n_boot: int,
) -> float:
    """Return ``(1 + count) / (B + 1)`` using weak inequalities."""
    alt = parse_alternative(alternative)
    draws = np.asarray(stars, dtype=float)
    if draws.ndim != 1 or draws.shape[0] != int(n_boot):
        raise ValueError("stars must be a 1-d array of length n_boot")
    if alt == ALTERNATIVE_TWO_SIDED:
        count = int(np.sum(np.abs(draws) >= abs(float(observed))))
    elif alt == ALTERNATIVE_A_BETTER:
        count = int(np.sum(draws <= float(observed)))
    else:
        count = int(np.sum(draws >= float(observed)))
    return float((1 + count) / (int(n_boot) + 1))


def derived_calibration_seed(*keys: int) -> int:
    """Return a deterministic 32-bit seed from an explicit integer key tuple."""
    sequence = np.random.SeedSequence(tuple(int(key) for key in keys))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def stationary_bootstrap_mean_test(
    differential: ArrayLike,
    *,
    alternative: str = ALTERNATIVE_TWO_SIDED,
    n_boot: int = PAIRWISE_N_BOOT,
    block_length: int | None = None,
    seed: int = PAIRWISE_BOOTSTRAP_SEED,
) -> StationaryBootstrapMeanResult:
    """Recentered stationary-bootstrap test of ``E[d_t] = 0``.

    The bootstrap population is ``d - mean(d)``. Indices come from the
    existing Politis-Romano engine. No long-run variance is estimated.
    """
    series = as_1d_finite(differential, "differential")
    n_obs = int(series.shape[0])
    if n_obs < 2:
        raise ValueError("the pairwise bootstrap mean test requires at least two observations")
    n_resamples = int(n_boot)
    if n_resamples < 1:
        raise ValueError("n_boot must be at least 1")
    # Constant series, including exact zeros, have no sampling variation.
    if np.all(series == series[0]):
        raise DegenerateLossDifferentialError(
            "all loss-differential observations are equal. The stationary-"
            "bootstrap mean test is undefined. No jitter was added."
        )
    alt = parse_alternative(alternative)
    mean_d = float(series.mean())
    centered = series - mean_d
    statistic = float(np.sqrt(n_obs) * mean_d)
    # Pass the resolved length explicitly so a later engine default cannot drift.
    length = (
        pairwise_block_length(n_obs) if block_length is None else int(block_length)
    )
    if length < 1:
        raise ValueError("block_length must be at least 1")
    indices = stationary_bootstrap_indices(
        n_obs,
        n_resamples,
        block_length=length,
        seed=int(seed),
    )
    stars = bootstrap_mean_statistics(centered, indices)
    p_value = finite_b_pvalue(statistic, stars, alt, n_resamples)
    return StationaryBootstrapMeanResult(
        statistic=statistic,
        p_value=p_value,
        alternative=alt,
        sample_size=n_obs,
        mean_differential=mean_d,
        n_boot=n_resamples,
        block_length=length,
        seed=int(seed),
        method=PAIRWISE_METHOD,
    )


def simulate_bootstrap_mean_size(
    *,
    cells: Sequence[tuple[int, float]],
    n_outer: int,
    n_boot: int,
    stage: str,
    dgp_seed: int = PAIRWISE_CALIBRATION_DGP_SEED,
    alpha: float = SIZE_ALPHA,
) -> pd.DataFrame:
    """Monte Carlo size of the recentered bootstrap mean test on AR(1) nulls.

    Outer DGP draws use ``stationary_ar1_paths``. Each replication receives a
    distinct bootstrap seed derived from ``dgp_seed``, ``T``, ``rho``, and the
    replication index. The production API seed ``20260917`` is not reused.
    """
    rows: list[dict[str, float | int | str]] = []
    for n_obs, rho in cells:
        # One DGP stream per cell, separate from the inner bootstrap seeds.
        path_seed = derived_calibration_seed(int(dgp_seed), 1, int(n_obs), int(round(rho * 1000)))
        paths = stationary_ar1_paths(
            n_reps=int(n_outer),
            n_obs=int(n_obs),
            rho=float(rho),
            seed=path_seed,
        )
        length = pairwise_block_length(int(n_obs))
        n_valid = 0
        n_fail = 0
        n_reject = 0
        for index, row in enumerate(paths):
            boot_seed = derived_calibration_seed(
                int(dgp_seed), 2, int(n_obs), int(round(rho * 1000)), int(index)
            )
            try:
                result = stationary_bootstrap_mean_test(
                    row,
                    alternative=ALTERNATIVE_TWO_SIDED,
                    n_boot=int(n_boot),
                    block_length=length,
                    seed=boot_seed,
                )
            except DegenerateLossDifferentialError:
                n_fail += 1
                continue
            n_valid += 1
            if result.p_value < float(alpha):
                n_reject += 1
        rate = float(n_reject / n_valid) if n_valid > 0 else float("nan")
        mcse = monte_carlo_standard_error(rate, n_valid if n_valid > 0 else 1)
        if n_valid > 0:
            lower, upper = monte_carlo_interval(rate, n_valid)
        else:
            lower, upper = float("nan"), float("nan")
            mcse = float("nan")
        rows.append(
            {
                "stage": str(stage),
                "T": int(n_obs),
                "rho": float(rho),
                "outer_replications": int(n_outer),
                "inner_bootstraps": int(n_boot),
                "block_length": length,
                "nominal_alpha": float(alpha),
                "valid_replications": n_valid,
                "failures": n_fail,
                "rejection_rate": rate,
                "mcse": mcse,
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
        )
    return pd.DataFrame(rows, columns=list(BOOTSTRAP_MEAN_TABLE_COLUMNS))


def write_bootstrap_mean_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Write a calibration table as CSV without an index column."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


def plot_bootstrap_mean_size_vs_rho(table: pd.DataFrame, path: str | Path) -> Path:
    """Rejection rate versus rho for the recentered bootstrap mean test."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("table has no rows to plot")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for n_obs in sorted(table["T"].unique()):
        subset = table.loc[table["T"] == n_obs].sort_values("rho")
        ax.plot(
            subset["rho"].to_numpy(),
            subset["rejection_rate"].to_numpy(),
            marker="o",
            label=f"T={int(n_obs)}",
        )
    ax.axhline(SIZE_ALPHA, color="0.2", linestyle="--", linewidth=1.0, label="nominal 0.05")
    ax.set_xlabel(r"AR(1) persistence $\rho$")
    ax.set_ylabel("two-sided rejection frequency")
    ax.set_title("Recentered stationary-bootstrap mean test")
    ax.set_ylim(0.0, max(0.20, float(table["rejection_rate"].max()) + 0.05))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_nw_versus_bootstrap_mean(
    bootstrap_table: pd.DataFrame,
    nw_table: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Descriptive overlay of automatic NW size and bootstrap-mean size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    auto = nw_table.loc[nw_table["lag_rule"] == "L_auto"].copy()
    if bootstrap_table.empty or auto.empty:
        raise ValueError("both bootstrap and automatic-NW tables are required")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_sizes = tuple(int(value) for value in sorted(bootstrap_table["T"].unique()))
    fig, axes = plt.subplots(
        1, len(sample_sizes), figsize=(4.0 * len(sample_sizes), 4.2), sharey=True
    )
    if len(sample_sizes) == 1:
        axes = [axes]
    for ax, n_obs in zip(axes, sample_sizes, strict=True):
        boot = bootstrap_table.loc[bootstrap_table["T"] == n_obs].sort_values("rho")
        rhos = set(float(value) for value in boot["rho"])
        nw = auto.loc[(auto["T"] == n_obs) & (auto["rho"].isin(rhos))].sort_values("rho")
        ax.plot(nw["rho"].to_numpy(), nw["rejection_rate"].to_numpy(), marker="o", label="automatic NW")
        ax.plot(
            boot["rho"].to_numpy(),
            boot["rejection_rate"].to_numpy(),
            marker="s",
            label="bootstrap mean",
        )
        ax.axhline(SIZE_ALPHA, color="0.2", linestyle="--", linewidth=1.0)
        ax.set_title(f"T={n_obs}")
        ax.set_xlabel(r"AR(1) persistence $\rho$")
    axes[0].set_ylabel("two-sided rejection frequency")
    axes[-1].legend(frameon=False)
    fig.suptitle("Automatic Newey-West DM and recentered bootstrap mean")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
