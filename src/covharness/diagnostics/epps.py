"""Epps-effect diagnostics for calendar-time realized covariance.

The diagnostic recomputes unscaled realized covariance and its implied
correlation matrix on previous-tick grids of several widths. It does not
change the realized-covariance formula, quote cleaning, or the forecast
horizon. No frequency is treated as the true covariance.

Finer grids yield more return observations. They do not automatically yield
a better covariance proxy. Nonsynchronous updates can place the same
economic shock in different clock-time intervals and attenuate measured
contemporaneous dependence (Epps, 1979; Barndorff-Nielsen, Hansen, Lunde,
and Shephard, 2011). Coarser sampling can allow that dependence to recover.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from covharness.data.synchronization import previous_tick_sync
from covharness.realized.daily import daily_realized_covariance

DEFAULT_FREQUENCIES_MINUTES: tuple[int, ...] = (1, 2, 5, 10, 15, 30)
REGULAR_OPEN = dt.time(9, 30)
REGULAR_CLOSE = dt.time(16, 0)
PSD_ATOL = 1e-12


@dataclass(frozen=True)
class OffDiagonalSummary:
    """Summaries of unique pairwise correlations. The diagonal is excluded."""

    mean: float
    median: float
    minimum: float
    maximum: float
    n_pairs: int
    n_finite: int


@dataclass(frozen=True)
class FrequencyEppsResult:
    """Realized covariance and correlation at one calendar-time frequency.

    ``rcov`` is the existing unscaled Gram matrix. ``correlation`` is the
    implied correlation matrix. ``m_over_n`` is the number of synchronized
    returns divided by the number of assets. Pair keys use
    ``asset_i-asset_j`` with ``i < j`` in the supplied asset order.
    """

    interval_minutes: int
    n_grid_prices: int
    n_complete_prices: int
    n_returns: int
    m_over_n: float
    assets: tuple[str, ...]
    rcov: NDArray[np.floating]
    correlation: NDArray[np.floating]
    pairwise_correlations: dict[str, float]
    off_diagonal: OffDiagonalSummary
    eigenvalues: NDArray[np.floating]
    numerical_rank: int
    min_eigenvalue: float
    psd: bool
    condition_number: float | None
    symmetry_error: float


@dataclass(frozen=True)
class EppsCurve:
    """Frequency scan. Results keep the caller-supplied frequency order."""

    assets: tuple[str, ...]
    by_frequency: tuple[FrequencyEppsResult, ...]

    @property
    def interval_minutes(self) -> tuple[int, ...]:
        return tuple(item.interval_minutes for item in self.by_frequency)

    @property
    def mean_off_diagonal(self) -> tuple[float, ...]:
        return tuple(item.off_diagonal.mean for item in self.by_frequency)

    @property
    def median_off_diagonal(self) -> tuple[float, ...]:
        return tuple(item.off_diagonal.median for item in self.by_frequency)


def covariance_to_correlation(rcov: ArrayLike) -> NDArray[np.floating]:
    """Implied correlation from a covariance matrix. No eigenvalue repair.

    Entry ``(i, j)`` is ``Sigma_ij / sqrt(Sigma_ii Sigma_jj)``. A non-positive
    variance leaves the corresponding row and column undefined (NaN). The
    input is not symmetrized.
    """
    sigma = np.asarray(rcov, dtype=float)
    # Require a finite square covariance matrix.
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"rcov must be square 2-d; got shape {sigma.shape}")
    if sigma.size == 0:
        raise ValueError("rcov must be non-empty")
    if not np.isfinite(sigma).all():
        raise ValueError("rcov must be finite (NaN and inf are rejected)")

    # Scale by standard deviations. Non-positive variance becomes NaN.
    std = np.sqrt(np.diag(sigma).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = sigma / np.outer(std, std)
    return corr


def off_diagonal_correlations(
    correlation: ArrayLike,
    assets: list[str] | tuple[str, ...],
) -> tuple[NDArray[np.floating], tuple[str, ...]]:
    """Unique pairwise correlations. Diagonal entries are not included.

    Pairs are the upper triangle ``i < j`` in the supplied asset order.
    """
    corr = np.asarray(correlation, dtype=float)
    names = tuple(str(asset) for asset in assets)
    # Require a square correlation matrix aligned with the asset list.
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"correlation must be square 2-d; got shape {corr.shape}")
    if corr.shape[0] != len(names):
        raise ValueError("correlation dimension must match the number of assets")

    # Upper-triangle pairs only (i < j). The diagonal is excluded.
    rows, cols = np.triu_indices(corr.shape[0], k=1)
    values = corr[rows, cols]
    keys = tuple(f"{names[i]}-{names[j]}" for i, j in zip(rows, cols))
    return values, keys


def summarize_off_diagonal(values: ArrayLike) -> OffDiagonalSummary:
    """Mean, median, min, and max of pairwise correlations, skipping NaN."""
    entries = np.asarray(values, dtype=float).reshape(-1)
    n_pairs = int(entries.size)
    finite = entries[np.isfinite(entries)]
    n_finite = int(finite.size)
    if n_finite == 0:
        nan = float("nan")
        return OffDiagonalSummary(
            mean=nan,
            median=nan,
            minimum=nan,
            maximum=nan,
            n_pairs=n_pairs,
            n_finite=0,
        )
    return OffDiagonalSummary(
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        n_pairs=n_pairs,
        n_finite=n_finite,
    )


def matrix_eigen_diagnostics(
    rcov: ArrayLike,
    *,
    atol: float = PSD_ATOL,
    n_returns: int | None = None,
) -> tuple[NDArray[np.floating], int, float, bool, float | None, float, float | None]:
    """Eigenvalues, rank, PSD flag, condition number, and optional ``M / N``.

    The condition number is reported only when the smallest eigenvalue is
    strictly larger than ``atol``. Otherwise it is ``None``. ``m_over_n`` is
    ``n_returns / N`` when ``n_returns`` is supplied, else ``None``.
    """
    sigma = np.asarray(rcov, dtype=float)
    # Symmetric-part eigenvalues of the Gram matrix. The matrix is not repaired.
    eigvals = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
    min_eig = float(eigvals.min()) if eigvals.size else float("nan")
    rank = int(np.linalg.matrix_rank(sigma, tol=atol))
    psd = bool(np.isfinite(min_eig) and min_eig >= -atol)
    if np.isfinite(min_eig) and min_eig > atol:
        condition_number = float(eigvals.max() / eigvals.min())
    else:
        condition_number = None
    symmetry_error = float(np.max(np.abs(sigma - sigma.T))) if sigma.size else 0.0
    n_assets = int(sigma.shape[0]) if sigma.ndim == 2 else 0
    if n_returns is None:
        m_over_n = None
    elif n_assets == 0:
        m_over_n = float("nan")
    else:
        m_over_n = float(n_returns) / float(n_assets)
    return eigvals, rank, min_eig, psd, condition_number, symmetry_error, m_over_n


def calendar_session_grid(
    day: str,
    interval_minutes: int,
    *,
    session_open: dt.time = REGULAR_OPEN,
    session_close: dt.time = REGULAR_CLOSE,
) -> pd.DatetimeIndex:
    """Inclusive regular-session calendar grid at a fixed minute width."""
    if int(interval_minutes) != interval_minutes or interval_minutes < 1:
        raise ValueError("interval_minutes must be a positive integer")
    # Inclusive 09:30-16:00 grid at the requested calendar width.
    start = pd.Timestamp.combine(pd.Timestamp(day).date(), session_open)
    end = pd.Timestamp.combine(pd.Timestamp(day).date(), session_close)
    grid = pd.date_range(start, end, freq=f"{int(interval_minutes)}min")
    if grid.empty:
        raise ValueError("sampling grid is empty")
    return grid.rename("timestamp")


def drop_incomplete_open(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop grid rows with any missing price. Do not fill the 09:30 point."""
    return prices.dropna(how="any")


def epps_at_frequency(
    ticks: pd.DataFrame,
    interval_minutes: int,
    *,
    day: str,
    assets: list[str] | tuple[str, ...],
    session_open: dt.time = REGULAR_OPEN,
    session_close: dt.time = REGULAR_CLOSE,
) -> FrequencyEppsResult:
    """Previous-tick RCov and implied correlation at one sampling frequency.

    Uses ``previous_tick_sync`` and ``daily_realized_covariance``. Incomplete
    opening rows are dropped. They are not filled from pre-market quotes.
    """
    names = tuple(str(asset) for asset in assets)
    if not names:
        raise ValueError("assets must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("assets must be unique")

    # Previous-tick onto the calendar grid, then drop incomplete opening rows.
    grid = calendar_session_grid(
        day,
        interval_minutes,
        session_open=session_open,
        session_close=session_close,
    )
    prices = previous_tick_sync(ticks, grid).reindex(columns=list(names))
    complete = drop_incomplete_open(prices)
    n_grid = int(len(prices))
    n_complete = int(len(complete))
    if n_complete < 2:
        raise ValueError(
            "need at least two complete synchronized prices after dropping "
            "incomplete opening rows"
        )

    # Unscaled daily RCov from the existing production function.
    rcov = daily_realized_covariance(complete)
    corr = covariance_to_correlation(rcov)
    pair_values, pair_keys = off_diagonal_correlations(corr, names)
    pairwise = {
        key: float(value) if np.isfinite(value) else float("nan")
        for key, value in zip(pair_keys, pair_values)
    }
    eigvals, rank, min_eig, psd, cond, symmetry_error, m_over_n = matrix_eigen_diagnostics(
        rcov, n_returns=n_complete - 1
    )
    return FrequencyEppsResult(
        interval_minutes=int(interval_minutes),
        n_grid_prices=n_grid,
        n_complete_prices=n_complete,
        n_returns=n_complete - 1,
        m_over_n=float(m_over_n) if m_over_n is not None else float("nan"),
        assets=names,
        rcov=np.asarray(rcov, dtype=float),
        correlation=np.asarray(corr, dtype=float),
        pairwise_correlations=pairwise,
        off_diagonal=summarize_off_diagonal(pair_values),
        eigenvalues=np.asarray(eigvals, dtype=float),
        numerical_rank=rank,
        min_eigenvalue=min_eig,
        psd=psd,
        condition_number=cond,
        symmetry_error=symmetry_error,
    )


def epps_across_frequencies(
    ticks: pd.DataFrame,
    *,
    day: str,
    assets: list[str] | tuple[str, ...],
    frequencies_minutes: tuple[int, ...] | list[int] = DEFAULT_FREQUENCIES_MINUTES,
    session_open: dt.time = REGULAR_OPEN,
    session_close: dt.time = REGULAR_CLOSE,
) -> EppsCurve:
    """Scan sampling frequencies. Caller order is preserved. No monotonicity is imposed."""
    freqs = tuple(int(delta) for delta in frequencies_minutes)
    if not freqs:
        raise ValueError("frequencies_minutes must be non-empty")
    # One previous-tick RCov per requested calendar width, same cleaned ticks.
    results = tuple(
        epps_at_frequency(
            ticks,
            delta,
            day=day,
            assets=assets,
            session_open=session_open,
            session_close=session_close,
        )
        for delta in freqs
    )
    return EppsCurve(assets=tuple(str(asset) for asset in assets), by_frequency=results)


def frequency_result_to_dict(result: FrequencyEppsResult) -> dict[str, object]:
    """JSON-friendly record for one sampling frequency."""
    return {
        "interval_minutes": result.interval_minutes,
        "n_grid_prices": result.n_grid_prices,
        "n_complete_prices": result.n_complete_prices,
        "n_returns": result.n_returns,
        "m_over_n": result.m_over_n,
        "assets": list(result.assets),
        "rcov": result.rcov.tolist(),
        "correlation": result.correlation.tolist(),
        "pairwise_correlations": result.pairwise_correlations,
        "mean_off_diagonal_correlation": result.off_diagonal.mean,
        "median_off_diagonal_correlation": result.off_diagonal.median,
        "min_off_diagonal_correlation": result.off_diagonal.minimum,
        "max_off_diagonal_correlation": result.off_diagonal.maximum,
        "n_off_diagonal_pairs": result.off_diagonal.n_pairs,
        "n_finite_off_diagonal_pairs": result.off_diagonal.n_finite,
        "eigenvalues": result.eigenvalues.tolist(),
        "numerical_rank": result.numerical_rank,
        "min_eigenvalue": result.min_eigenvalue,
        "psd": result.psd,
        "condition_number": result.condition_number,
        "symmetry_error": result.symmetry_error,
    }


def epps_curve_to_dict(curve: EppsCurve) -> dict[str, object]:
    """JSON-friendly frequency scan, including the average off-diagonal series."""
    return {
        "assets": list(curve.assets),
        "frequencies_minutes": list(curve.interval_minutes),
        "mean_off_diagonal_correlation": list(curve.mean_off_diagonal),
        "median_off_diagonal_correlation": list(curve.median_off_diagonal),
        "by_frequency": [frequency_result_to_dict(item) for item in curve.by_frequency],
    }


def plot_epps_curve(
    curve: EppsCurve,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Sampling interval versus off-diagonal realized correlation.

    The average is drawn as the main series. Individual pairs are retained so
    the average does not hide heterogeneous behavior. The figure is not forced
    to be monotonic.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(curve.interval_minutes, dtype=float)
    # Pairwise series first, then the average, so the mean remains readable.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    pair_keys = list(curve.by_frequency[0].pairwise_correlations.keys())
    for key in pair_keys:
        y_pair = [item.pairwise_correlations[key] for item in curve.by_frequency]
        ax.plot(x, y_pair, color="0.75", linewidth=0.9, alpha=0.9)
    ax.plot(
        x,
        list(curve.mean_off_diagonal),
        color="0.10",
        linewidth=2.2,
        marker="o",
        label="average off-diagonal",
    )
    ax.set_xlabel("sampling interval (minutes)")
    ax.set_ylabel("realized correlation")
    ax.set_xticks(list(curve.interval_minutes))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
