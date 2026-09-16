"""Synthetic aligned panels for leak-free rolling evaluation demos.

The generator is an integration fixture. It is not a research-grade market
simulator and does not represent actual markets. Evaluation targets are
noisy realized-covariance proxies, not the latent generating covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from covharness.features.origin_state import realized_quarticity_aggregate
from covharness.losses.contracts import InvalidCovarianceMatrixError, require_symmetric
from covharness.realized.rcov import realized_covariance
from covharness.simulation.intraday import simulate_synchronized_gaussian_returns

DEFAULT_BENCHMARK_SEED = 20260916
DEFAULT_N_INTERVALS = 12
MAX_PROXY_DRAWS = 32


@dataclass(frozen=True)
class SyntheticBenchmarkPanel:
    """Calendar-aligned synthetic returns, realized covariance, and quarticity.

    ``realized_covariances[t]`` is the noisy proxy ``S_t`` used as the
    evaluation target. It is not the latent generating covariance.
    """

    calendar: pd.DatetimeIndex
    daily_returns: NDArray[np.floating]
    realized_covariances: NDArray[np.floating]
    realized_quarticity: NDArray[np.floating]
    seed: int
    n_intervals: int


def synthetic_benchmark_panel(
    *,
    n_times: int,
    n_assets: int,
    seed: int = DEFAULT_BENCHMARK_SEED,
    n_intervals: int = DEFAULT_N_INTERVALS,
    start: str = "1990-01-01",
) -> SyntheticBenchmarkPanel:
    """Draw a deterministic synthetic evaluation panel.

    Latent daily covariance uses smooth time-varying volatilities and a
    time-varying equicorrelation. Interval returns are Gaussian given that
    covariance. Realized covariance is the unscaled Gram matrix of those
    intervals. Per-asset quarticity uses ``RQ_i = (M/3) sum_l r_{i,l}^4``.
    """
    if n_times < 2:
        raise ValueError(f"n_times must be >= 2; got {n_times}")
    if n_assets < 2:
        raise ValueError(f"n_assets must be >= 2; got {n_assets}")
    if n_intervals < 2:
        raise ValueError(f"n_intervals must be >= 2; got {n_intervals}")
    rng = np.random.default_rng(int(seed))
    calendar = pd.bdate_range(start, periods=int(n_times))
    returns = np.empty((n_times, n_assets), dtype=float)
    rcov = np.empty((n_times, n_assets, n_assets), dtype=float)
    rq = np.empty((n_times, n_assets), dtype=float)
    for time in range(n_times):
        # Time-varying latent covariance. This matrix is not the evaluation target.
        latent = _latent_covariance(time, n_times, n_assets)
        intervals, proxy = _draw_pd_proxy(latent, n_intervals, rng)
        returns[time] = intervals.sum(axis=0)
        rcov[time] = proxy
        rq[time] = realized_quarticity_aggregate(intervals).asset_quarticity
    return SyntheticBenchmarkPanel(
        calendar=calendar,
        daily_returns=np.array(returns, dtype=float, copy=True),
        realized_covariances=np.array(rcov, dtype=float, copy=True),
        realized_quarticity=np.array(rq, dtype=float, copy=True),
        seed=int(seed),
        n_intervals=int(n_intervals),
    )


def _latent_covariance(time: int, n_times: int, n_assets: int) -> NDArray[np.floating]:
    """Return a strictly PD equicorrelation covariance with moving scale."""
    phase = 2.0 * np.pi * time / max(n_times, 1)
    # Asset-specific volatilities. They are not constant across dates.
    scales = 0.012 * (1.4 + 0.45 * np.sin(phase + 0.7 * np.arange(n_assets)))
    rho_max = 0.85 / max(n_assets - 1, 1)
    rho = 0.18 + 0.12 * np.sin(1.3 * phase)
    rho = float(np.clip(rho, -rho_max + 1e-3, min(0.85, 1.0 - 1e-3)))
    correlation = np.full((n_assets, n_assets), rho, dtype=float)
    np.fill_diagonal(correlation, 1.0)
    covariance = correlation * np.outer(scales, scales)
    covariance = 0.5 * (covariance + covariance.T)
    require_symmetric(covariance, "synthetic latent covariance")
    np.linalg.cholesky(covariance)
    return covariance


def _draw_pd_proxy(
    latent: NDArray[np.floating],
    n_intervals: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Draw interval returns whose Gram matrix is strictly PD.

    A non-PD draw is rejected and resampled. The latent matrix is not used
    as the stored proxy.
    """
    last_error: Exception | None = None
    for _attempt in range(MAX_PROXY_DRAWS):
        intervals = simulate_synchronized_gaussian_returns(
            latent, n_intervals=n_intervals, rng=rng, n_days=1
        )[0]
        proxy = realized_covariance(intervals)
        if _strict_pd(proxy):
            return intervals, proxy
        last_error = InvalidCovarianceMatrixError(
            "synthetic realized covariance was not strictly PD"
        )
    raise InvalidCovarianceMatrixError(
        "synthetic realized covariance remained non-PD after "
        f"{MAX_PROXY_DRAWS} draws"
    ) from last_error


def _strict_pd(matrix: NDArray[np.floating]) -> bool:
    """True iff the matrix is finite, square, and Cholesky succeeds."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.isfinite(matrix).all():
        return False
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True
