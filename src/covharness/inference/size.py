"""Fixed-seed size and power demonstration for HAC DM versus a naive IID t-test.

Simulation is required to validate the estimator against known truth. It does
not substitute for the project's empirical finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from covharness.inference.dm import diebold_mariano, naive_iid_t_statistic
from covharness.inference.exceptions import DegenerateLossDifferentialError

SIZE_SEED = 20260912
SIZE_N_REPS = 2000
SIZE_N_OBS = 250
SIZE_RHO = 0.6
SIZE_ALPHA = 0.05
SIZE_ALTERNATIVE_MEAN = -0.20


@dataclass(frozen=True)
class HACSizePowerResult:
    """Rejection frequencies under known-null and known-alternative DGPs."""

    n_reps: int
    n_obs: int
    seed: int
    alpha: float
    rho: float
    alternative_mean: float
    iid_null_naive_rejection: float
    iid_null_dm_rejection: float
    ar_null_naive_rejection: float
    ar_null_dm_rejection: float
    alternative_dm_rejection: float
    alternative_dm_a_better_rejection: float


def simulate_hac_size_power(
    *,
    n_reps: int = SIZE_N_REPS,
    n_obs: int = SIZE_N_OBS,
    seed: int = SIZE_SEED,
    rho: float = SIZE_RHO,
    alpha: float = SIZE_ALPHA,
    alternative_mean: float = SIZE_ALTERNATIVE_MEAN,
) -> HACSizePowerResult:
    """Monte Carlo size under IID and AR(1) nulls, and power under a mean shift."""
    rng = np.random.default_rng(seed)
    iid = rng.normal(loc=0.0, scale=1.0, size=(n_reps, n_obs))
    ar = _stationary_ar1(rng, n_reps=n_reps, n_obs=n_obs, rho=rho)
    alternative = alternative_mean + rng.normal(loc=0.0, scale=1.0, size=(n_reps, n_obs))
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    one_sided_critical = float(norm.ppf(alpha))
    return HACSizePowerResult(
        n_reps=n_reps,
        n_obs=n_obs,
        seed=seed,
        alpha=alpha,
        rho=rho,
        alternative_mean=alternative_mean,
        iid_null_naive_rejection=_naive_two_sided_rate(iid, critical),
        iid_null_dm_rejection=_dm_two_sided_rate(iid, critical),
        ar_null_naive_rejection=_naive_two_sided_rate(ar, critical),
        ar_null_dm_rejection=_dm_two_sided_rate(ar, critical),
        alternative_dm_rejection=_dm_two_sided_rate(alternative, critical),
        alternative_dm_a_better_rejection=_dm_a_better_rate(
            alternative, one_sided_critical
        ),
    )


def plot_hac_size_power(result: HACSizePowerResult, path: str | Path) -> Path:
    """Bar chart of naive versus HAC rejection rates. Reusable figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "IID null\nnaive t",
        "IID null\nHAC DM",
        "AR(1) null\nnaive t",
        "AR(1) null\nHAC DM",
        "mean shift\nHAC DM",
    ]
    rates = [
        result.iid_null_naive_rejection,
        result.iid_null_dm_rejection,
        result.ar_null_naive_rejection,
        result.ar_null_dm_rejection,
        result.alternative_dm_rejection,
    ]
    colors = ["0.65", "0.25", "0.65", "0.25", "0.40"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, rates, color=colors, width=0.7)
    ax.axhline(result.alpha, color="0.2", linestyle="--", linewidth=1.0, label="nominal 5%")
    ax.set_xticks(x, labels)
    ax.set_ylabel("rejection frequency")
    ax.set_ylim(0.0, max(0.45, max(rates) + 0.05))
    ax.set_title("Naive t-test versus HAC Diebold-Mariano")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _stationary_ar1(
    rng: np.random.Generator, *, n_reps: int, n_obs: int, rho: float
) -> np.ndarray:
    """Mean-zero Gaussian AR(1) with unit innovation variance."""
    innovations = rng.normal(loc=0.0, scale=1.0, size=(n_reps, n_obs))
    series = np.empty((n_reps, n_obs), dtype=float)
    series[:, 0] = innovations[:, 0] / np.sqrt(1.0 - rho * rho)
    for t in range(1, n_obs):
        series[:, t] = rho * series[:, t - 1] + innovations[:, t]
    return series


def _naive_two_sided_rate(paths: np.ndarray, critical: float) -> float:
    stats = np.empty(paths.shape[0], dtype=float)
    for i, row in enumerate(paths):
        stats[i] = naive_iid_t_statistic(row)
    return float(np.mean(np.abs(stats) > critical))


def _dm_two_sided_rate(paths: np.ndarray, critical: float) -> float:
    rejected = 0
    n_ok = 0
    for row in paths:
        try:
            result = diebold_mariano(row, alternative="two_sided")
        except DegenerateLossDifferentialError:
            continue
        n_ok += 1
        if abs(result.statistic) > critical:
            rejected += 1
    return float(rejected / n_ok)


def _dm_a_better_rate(paths: np.ndarray, critical: float) -> float:
    rejected = 0
    n_ok = 0
    for row in paths:
        try:
            result = diebold_mariano(row, alternative="a_better")
        except DegenerateLossDifferentialError:
            continue
        n_ok += 1
        if result.statistic < critical:
            rejected += 1
    return float(rejected / n_ok)
