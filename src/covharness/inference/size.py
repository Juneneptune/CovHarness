"""Fixed-seed size and power demonstration for HAC DM versus a naive IID t-test.

Simulation is required to validate the estimator against known truth. It does
not substitute for the project's empirical finding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from covharness.inference.dm import diebold_mariano, naive_iid_t_statistic
from covharness.inference.exceptions import DegenerateLossDifferentialError
from covharness.inference.hac import newey_west_1994_lags

SIZE_SEED = 20260912
SIZE_N_REPS = 2000
SIZE_N_OBS = 250
SIZE_RHO = 0.6
SIZE_ALPHA = 0.05
SIZE_ALTERNATIVE_MEAN = -0.20

CALIBRATION_SEED = 20260916
CALIBRATION_N_REPS = 5000
CALIBRATION_SAMPLE_SIZES = (250, 500, 1000)
CALIBRATION_RHOS = (0.0, 0.3, 0.6, 0.8, 0.9)
LAG_RULE_L0 = "L0"
LAG_RULE_AUTO = "L_auto"
LAG_RULE_TWO_AUTO = "2L_auto"
LAG_RULE_FOUR_AUTO = "4L_auto"
CALIBRATION_LAG_RULES = (
    LAG_RULE_L0,
    LAG_RULE_AUTO,
    LAG_RULE_TWO_AUTO,
    LAG_RULE_FOUR_AUTO,
)
MC_Z_95 = 1.96
CALIBRATION_TABLE_COLUMNS = (
    "T",
    "rho",
    "lag_rule",
    "hac_lag",
    "replications",
    "valid_replications",
    "failures",
    "rejection_rate",
    "mcse",
    "ci95_lower",
    "ci95_upper",
    "true_lrv",
    "mean_estimated_lrv_ratio",
    "median_estimated_lrv_ratio",
)


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
    persistence = float(rho)
    if abs(persistence) >= 1.0:
        raise ValueError(f"AR(1) persistence must satisfy |rho| < 1; got {persistence}")
    if n_reps < 1 or n_obs < 2:
        raise ValueError("AR(1) paths require n_reps >= 1 and n_obs >= 2")
    # Exact stationary start. d_0 ~ N(0, 1/(1-rho^2)). No burn-in.
    innovations = rng.normal(loc=0.0, scale=1.0, size=(n_reps, n_obs))
    series = np.empty((n_reps, n_obs), dtype=float)
    series[:, 0] = innovations[:, 0] / np.sqrt(1.0 - persistence * persistence)
    for t in range(1, n_obs):
        series[:, t] = persistence * series[:, t - 1] + innovations[:, t]
    return series


def stationary_ar1_paths(
    *,
    n_reps: int,
    n_obs: int,
    rho: float,
    seed: int,
) -> np.ndarray:
    """Draw mean-zero Gaussian AR(1) paths from an explicit seed.

    Innovations are N(0, 1). The null is E[d_t]=0. This generator does not
    consume forecast records or empirical losses.
    """
    rng = np.random.default_rng(int(seed))
    return _stationary_ar1(rng, n_reps=int(n_reps), n_obs=int(n_obs), rho=float(rho))


def ar1_true_long_run_variance(rho: float) -> float:
    """Return the AR(1) long-run variance ``1 / (1 - rho)^2``.

    Innovation variance is 1. This quantity is simulation truth. It is not
    used inside the implemented Diebold-Mariano statistic.
    """
    persistence = float(rho)
    if abs(persistence) >= 1.0:
        raise ValueError(f"AR(1) persistence must satisfy |rho| < 1; got {persistence}")
    return 1.0 / (1.0 - persistence) ** 2


def calibration_hac_lag(n_observations: int, lag_rule: str) -> int:
    """Return the frozen calibration lag, capped at ``T-1`` when required."""
    n_obs = int(n_observations)
    rule = str(lag_rule)
    if n_obs < 2:
        raise ValueError("HAC lags require at least two observations")
    if rule == LAG_RULE_L0:
        return 0
    automatic = newey_west_1994_lags(n_obs)
    if rule == LAG_RULE_AUTO:
        return automatic
    if rule == LAG_RULE_TWO_AUTO:
        return min(2 * automatic, n_obs - 1)
    if rule == LAG_RULE_FOUR_AUTO:
        return min(4 * automatic, n_obs - 1)
    raise ValueError(
        f"lag_rule must be one of {CALIBRATION_LAG_RULES}; got {lag_rule!r}"
    )


def dm_maxlags_argument(n_observations: int, lag_rule: str) -> int | None:
    """Return the ``diebold_mariano`` maxlags argument for a calibration rule.

    ``L_auto`` passes ``None`` so the current Newey-West 1994 default is used.
    Other rules pass an explicit lag into the existing HAC implementation.
    """
    if str(lag_rule) == LAG_RULE_AUTO:
        calibration_hac_lag(n_observations, lag_rule)
        return None
    return calibration_hac_lag(n_observations, lag_rule)


def monte_carlo_standard_error(rate: float, n_replications: int) -> float:
    """Return ``sqrt(p (1-p) / B)`` for a Bernoulli Monte Carlo rate."""
    if n_replications < 1:
        raise ValueError("n_replications must be >= 1")
    p_hat = float(rate)
    if not np.isfinite(p_hat):
        return float("nan")
    return float(np.sqrt(p_hat * (1.0 - p_hat) / int(n_replications)))


def monte_carlo_interval(
    rate: float,
    n_replications: int,
    *,
    z_value: float = MC_Z_95,
) -> tuple[float, float]:
    """Return ``p_hat +/- z * MCSE``. Bounds are not clipped to ``[0, 1]``."""
    p_hat = float(rate)
    half = float(z_value) * monte_carlo_standard_error(p_hat, n_replications)
    return p_hat - half, p_hat + half


def score_dm_size_on_paths(
    paths: np.ndarray,
    *,
    maxlags: int | None,
    alpha: float = SIZE_ALPHA,
) -> tuple[int, int, int, np.ndarray]:
    """Score paths with the existing DM test. Returns valid, fail, reject, LRVs.

    Rejection uses the two-sided N(0,1) critical value at ``alpha``. Degenerate
    HAC/DM outcomes are failures, not silent skips of the requested B.
    """
    if paths.ndim != 2:
        raise ValueError("paths must have shape (n_reps, T)")
    critical = float(norm.ppf(1.0 - float(alpha) / 2.0))
    n_valid = 0
    n_fail = 0
    n_reject = 0
    long_runs: list[float] = []
    for row in paths:
        try:
            result = diebold_mariano(row, alternative="two_sided", maxlags=maxlags)
        except DegenerateLossDifferentialError:
            n_fail += 1
            continue
        n_valid += 1
        long_runs.append(float(result.hac_long_run_variance))
        if abs(result.statistic) > critical:
            n_reject += 1
    return n_valid, n_fail, n_reject, np.asarray(long_runs, dtype=float)


def simulate_dm_hac_calibration(
    *,
    sample_sizes: Sequence[int] = CALIBRATION_SAMPLE_SIZES,
    rhos: Sequence[float] = CALIBRATION_RHOS,
    lag_rules: Sequence[str] = CALIBRATION_LAG_RULES,
    n_reps: int = CALIBRATION_N_REPS,
    seed: int = CALIBRATION_SEED,
    alpha: float = SIZE_ALPHA,
) -> pd.DataFrame:
    """Monte Carlo size of the current HAC DM procedure on synthetic AR(1) nulls.

    Paths are generated once per ``(T, rho)`` cell and reused across lag rules.
    The implemented DM statistic is unchanged. The theoretical long-run
    variance is a diagnostic only.
    """
    n_replications = int(n_reps)
    if n_replications < 1:
        raise ValueError("n_reps must be >= 1")
    sizes = tuple(int(value) for value in sample_sizes)
    persistences = tuple(float(value) for value in rhos)
    rules = tuple(str(value) for value in lag_rules)
    if len(sizes) < 1 or len(persistences) < 1 or len(rules) < 1:
        raise ValueError("sample_sizes, rhos, and lag_rules must be non-empty")
    unknown = [rule for rule in rules if rule not in CALIBRATION_LAG_RULES]
    if unknown:
        raise ValueError(
            f"lag_rule must be one of {CALIBRATION_LAG_RULES}; got {unknown[0]!r}"
        )
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float | int | str]] = []
    for n_obs in sizes:
        for rho in persistences:
            # One set of null paths is scored under every requested lag rule.
            paths = _stationary_ar1(
                rng, n_reps=n_replications, n_obs=n_obs, rho=rho
            )
            true_lrv = ar1_true_long_run_variance(rho)
            for lag_rule in rules:
                maxlags = dm_maxlags_argument(n_obs, lag_rule)
                hac_lag = calibration_hac_lag(n_obs, lag_rule)
                n_valid, n_fail, n_reject, long_runs = score_dm_size_on_paths(
                    paths, maxlags=maxlags, alpha=alpha
                )
                rate = float(n_reject / n_valid) if n_valid > 0 else float("nan")
                mcse = monte_carlo_standard_error(rate, n_replications)
                lower, upper = monte_carlo_interval(rate, n_replications)
                if n_valid > 0:
                    ratios = long_runs / true_lrv
                    mean_ratio = float(ratios.mean())
                    median_ratio = float(np.median(ratios))
                else:
                    mean_ratio = float("nan")
                    median_ratio = float("nan")
                rows.append(
                    {
                        "T": n_obs,
                        "rho": rho,
                        "lag_rule": lag_rule,
                        "hac_lag": hac_lag,
                        "replications": n_replications,
                        "valid_replications": n_valid,
                        "failures": n_fail,
                        "rejection_rate": rate,
                        "mcse": mcse,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "true_lrv": true_lrv,
                        "mean_estimated_lrv_ratio": mean_ratio,
                        "median_estimated_lrv_ratio": median_ratio,
                    }
                )
    table = pd.DataFrame(rows, columns=list(CALIBRATION_TABLE_COLUMNS))
    return table


def write_calibration_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Write the calibration table as CSV without an index column."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


def plot_calibration_size_vs_rho(table: pd.DataFrame, path: str | Path) -> Path:
    """Rejection rate versus rho under the current automatic lag."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    auto = table.loc[table["lag_rule"] == LAG_RULE_AUTO].copy()
    if auto.empty:
        raise ValueError("table has no L_auto rows to plot")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for n_obs in sorted(auto["T"].unique()):
        subset = auto.loc[auto["T"] == n_obs].sort_values("rho")
        ax.plot(
            subset["rho"].to_numpy(),
            subset["rejection_rate"].to_numpy(),
            marker="o",
            label=f"T={int(n_obs)}",
        )
    ax.axhline(SIZE_ALPHA, color="0.2", linestyle="--", linewidth=1.0, label="nominal 0.05")
    ax.set_xlabel(r"AR(1) persistence $\rho$")
    ax.set_ylabel("two-sided rejection frequency")
    ax.set_title("Current automatic Newey-West 1994 lag")
    ax.set_ylim(0.0, max(0.20, float(auto["rejection_rate"].max()) + 0.05))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_calibration_bandwidth(table: pd.DataFrame, path: str | Path) -> Path:
    """Bandwidth sensitivity at high persistence. One panel per sample size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    persistences = (0.6, 0.8, 0.9)
    selected = table.loc[table["rho"].isin(persistences)].copy()
    if selected.empty:
        raise ValueError("table has no high-persistence rows to plot")
    sample_sizes = tuple(int(value) for value in sorted(selected["T"].unique()))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(sample_sizes), figsize=(4.0 * len(sample_sizes), 4.2), sharey=True
    )
    if len(sample_sizes) == 1:
        axes = [axes]
    x = np.arange(len(CALIBRATION_LAG_RULES), dtype=float)
    for ax, n_obs in zip(axes, sample_sizes, strict=True):
        panel = selected.loc[selected["T"] == n_obs]
        for rho in persistences:
            rates = []
            for rule in CALIBRATION_LAG_RULES:
                match = panel.loc[
                    (panel["rho"] == rho) & (panel["lag_rule"] == rule),
                    "rejection_rate",
                ]
                rates.append(float(match.iloc[0]) if len(match) else np.nan)
            ax.plot(x, rates, marker="o", label=rf"$\rho$={rho}")
        ax.axhline(SIZE_ALPHA, color="0.2", linestyle="--", linewidth=1.0)
        ax.set_xticks(x, list(CALIBRATION_LAG_RULES))
        ax.set_title(f"T={n_obs}")
        ax.set_xlabel("HAC lag rule")
    axes[0].set_ylabel("two-sided rejection frequency")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle("HAC lag sensitivity on persistent AR(1) nulls")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


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
