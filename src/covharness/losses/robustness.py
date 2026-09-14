"""Proxy-robust ranking demonstration with a known latent covariance.

The construction is a scalar exponential multiplier ``u ~ Exp(1)`` applied to
a fixed SPD ``Sigma``, so ``S = u Sigma`` and ``E[S] = Sigma``. ``H_A`` is
the true matrix. ``H_B = log(2) Sigma`` matches the median of ``u`` and is
inferior under squared Frobenius, reduced QLIKE, and full Stein, but can win
under ordinary unsquared Frobenius because that criterion tracks the median
rather than the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covharness.losses.frobenius import squared_frobenius_loss, unsquared_frobenius_loss
from covharness.losses.qlike import full_stein_loss, reduced_qlike_loss

# Reproducible Monte Carlo. The draw count is large enough that ranking of
# mean losses is stable across reruns of the fixed seed.
ROBUSTNESS_SEED = 20260212
ROBUSTNESS_N_DRAWS = 25_000

TRUE_SIGMA = np.array(
    [
        [1.0, 0.3],
        [0.3, 1.0],
    ],
    dtype=float,
)
# Median of Exp(1) is log(2). This forecast matches the proxy median, not E[S].
MEDIAN_SCALE = float(np.log(2.0))


@dataclass(frozen=True)
class AnalyticExpectedLosses:
    """Closed-form expected losses for the Exp(1) scale-proxy family."""

    sigma_frobenius_squared: float
    squared_frobenius_true: float
    squared_frobenius_median: float
    unsquared_frobenius_true: float
    unsquared_frobenius_median: float
    reduced_qlike_true: float
    reduced_qlike_median: float


@dataclass(frozen=True)
class RobustnessMonteCarloResult:
    """Fixed-seed expected-loss ranking and single-draw H_B win rates."""

    n_draws: int
    seed: int
    mean_squared_frobenius_true: float
    mean_squared_frobenius_median: float
    mean_reduced_qlike_true: float
    mean_reduced_qlike_median: float
    mean_full_stein_true: float
    mean_full_stein_median: float
    mean_unsquared_frobenius_true: float
    mean_unsquared_frobenius_median: float
    single_draw_hb_win_rate_squared_frobenius: float
    single_draw_hb_win_rate_reduced_qlike: float
    single_draw_hb_win_rate_full_stein: float
    single_draw_hb_win_rate_unsquared_frobenius: float


def true_forecast(sigma: NDArray[np.floating] = TRUE_SIGMA) -> NDArray[np.floating]:
    """``H_A = Sigma``. Closest forecast of the latent covariance."""
    return np.array(sigma, dtype=float, copy=True)


def median_matched_forecast(
    sigma: NDArray[np.floating] = TRUE_SIGMA,
) -> NDArray[np.floating]:
    """``H_B = log(2) Sigma``. Inferior mean forecast, matched to median(u)."""
    return MEDIAN_SCALE * np.array(sigma, dtype=float, copy=True)


def analytic_expected_losses(
    sigma: NDArray[np.floating] = TRUE_SIGMA,
) -> AnalyticExpectedLosses:
    """Expected losses for ``S = u Sigma`` with ``u ~ Exp(1)``.

    ``E[(u-c)^2] = 1 + (1-c)^2``. ``E[|u-c|] = 2 e^{-c} + c - 1`` for ``c > 0``.
    Reduced QLIKE has ``E[L_Q] = N log c + logdet(Sigma) + N / c``.
    """
    sigma = np.array(sigma, dtype=float, copy=True)
    n_assets = sigma.shape[0]
    # Squared Frobenius scale ||Sigma||_F^2.
    sigma_f2 = float(np.sum(sigma * sigma))
    sigma_f = float(np.sqrt(sigma_f2))
    logdet_sigma = float(np.linalg.slogdet(sigma).logabsdet)

    def expected_squared(scale: float) -> float:
        return (1.0 + (1.0 - scale) ** 2) * sigma_f2

    def expected_unsquared(scale: float) -> float:
        return (2.0 * np.exp(-scale) + scale - 1.0) * sigma_f

    def expected_qlike(scale: float) -> float:
        return n_assets * np.log(scale) + logdet_sigma + n_assets / scale

    return AnalyticExpectedLosses(
        sigma_frobenius_squared=sigma_f2,
        squared_frobenius_true=expected_squared(1.0),
        squared_frobenius_median=expected_squared(MEDIAN_SCALE),
        unsquared_frobenius_true=expected_unsquared(1.0),
        unsquared_frobenius_median=expected_unsquared(MEDIAN_SCALE),
        reduced_qlike_true=expected_qlike(1.0),
        reduced_qlike_median=expected_qlike(MEDIAN_SCALE),
    )


def sample_scaled_proxy(
    rng: np.random.Generator,
    sigma: NDArray[np.floating] = TRUE_SIGMA,
    n_draws: int = 1,
) -> NDArray[np.floating]:
    """Draw ``S = u Sigma`` with ``u ~ Exp(1)``, so ``E[S] = Sigma``."""
    scales = rng.exponential(scale=1.0, size=n_draws)
    # Broadcast the scalar multiplier onto the known SPD matrix.
    return scales.reshape(n_draws, 1, 1) * sigma


def monte_carlo_proxy_ranking(
    *,
    n_draws: int = ROBUSTNESS_N_DRAWS,
    seed: int = ROBUSTNESS_SEED,
    sigma: NDArray[np.floating] = TRUE_SIGMA,
) -> RobustnessMonteCarloResult:
    """Fixed-seed Monte Carlo of expected ranking and single-draw H_B win rates.

    A single-draw H_B win is a noisy proxy realization on which the inferior
    forecast receives the smaller loss. Proxy robustness concerns expected
    loss rankings, not pointwise rankings on every draw.
    Robust losses preserve expected ranking. Unsquared Frobenius need not.
    For this scale family the per-draw ranking of squared and unsquared
    Frobenius coincides, while their expected rankings differ.
    """
    rng = np.random.default_rng(seed)
    proxies = sample_scaled_proxy(rng, sigma=sigma, n_draws=n_draws)
    h_true = true_forecast(sigma)
    h_median = median_matched_forecast(sigma)

    sq_true = np.empty(n_draws)
    sq_med = np.empty(n_draws)
    q_true = np.empty(n_draws)
    q_med = np.empty(n_draws)
    stein_true = np.empty(n_draws)
    stein_med = np.empty(n_draws)
    uns_true = np.empty(n_draws)
    uns_med = np.empty(n_draws)
    for i, proxy in enumerate(proxies):
        sq_true[i] = squared_frobenius_loss(proxy, h_true)
        sq_med[i] = squared_frobenius_loss(proxy, h_median)
        q_true[i] = reduced_qlike_loss(proxy, h_true)
        q_med[i] = reduced_qlike_loss(proxy, h_median)
        stein_true[i] = full_stein_loss(proxy, h_true)
        stein_med[i] = full_stein_loss(proxy, h_median)
        uns_true[i] = unsquared_frobenius_loss(proxy, h_true)
        uns_med[i] = unsquared_frobenius_loss(proxy, h_median)

    def single_draw_hb_win_rate(
        loss_true: NDArray[np.floating], loss_med: NDArray[np.floating]
    ) -> float:
        return float(np.mean(loss_true > loss_med))

    return RobustnessMonteCarloResult(
        n_draws=n_draws,
        seed=seed,
        mean_squared_frobenius_true=float(sq_true.mean()),
        mean_squared_frobenius_median=float(sq_med.mean()),
        mean_reduced_qlike_true=float(q_true.mean()),
        mean_reduced_qlike_median=float(q_med.mean()),
        mean_full_stein_true=float(stein_true.mean()),
        mean_full_stein_median=float(stein_med.mean()),
        mean_unsquared_frobenius_true=float(uns_true.mean()),
        mean_unsquared_frobenius_median=float(uns_med.mean()),
        single_draw_hb_win_rate_squared_frobenius=single_draw_hb_win_rate(sq_true, sq_med),
        single_draw_hb_win_rate_reduced_qlike=single_draw_hb_win_rate(q_true, q_med),
        single_draw_hb_win_rate_full_stein=single_draw_hb_win_rate(stein_true, stein_med),
        single_draw_hb_win_rate_unsquared_frobenius=single_draw_hb_win_rate(uns_true, uns_med),
    )


def plot_proxy_robustness(
    result: RobustnessMonteCarloResult,
    path: str | Path,
) -> Path:
    """Bar comparison of expected losses. Reusable README/notebook figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "squared\nFrobenius",
        "reduced\nQLIKE",
        "full\nStein",
        "unsquared\nFrobenius",
    ]
    true_means = [
        result.mean_squared_frobenius_true,
        result.mean_reduced_qlike_true,
        result.mean_full_stein_true,
        result.mean_unsquared_frobenius_true,
    ]
    median_means = [
        result.mean_squared_frobenius_median,
        result.mean_reduced_qlike_median,
        result.mean_full_stein_median,
        result.mean_unsquared_frobenius_median,
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    ax.bar(x - width / 2.0, true_means, width, label=r"$H_A=\Sigma$", color="0.25")
    ax.bar(
        x + width / 2.0,
        median_means,
        width,
        label=r"$H_B=\log(2)\Sigma$",
        color="0.65",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Monte Carlo mean loss")
    ax.set_title("Proxy-robust ranking versus unsquared Frobenius")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
