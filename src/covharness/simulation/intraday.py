"""Synthetic synchronized Gaussian intraday returns."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def simulate_synchronized_gaussian_returns(
    sigma_daily: ArrayLike,
    n_intervals: int,
    rng: np.random.Generator,
    n_days: int = 1,
) -> NDArray[np.floating]:
    """Draw synchronized Gaussian intraday returns with known daily covariance.

    Each interval vector is

        r_j ~ N(0, Sigma_daily / M)

    independently across intervals ``j`` and days, so

        E[sum_j r_j r_j'] = Sigma_daily.

    Parameters
    ----------
    sigma_daily : array-like, shape (N, N)
        Target daily covariance.
    n_intervals : int
        ``M``, the number of synchronized intervals per day.
    rng : numpy.random.Generator
        Explicit RNG. Do not use the global NumPy seed.
    n_days : int
        Independent days to draw.

    Returns
    -------
    returns : ndarray, shape (n_days, M, N)
        Rows of each day's slice are synchronized return vectors.
    """
    sigma = np.asarray(sigma_daily, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"sigma_daily must be square 2-d; got shape {sigma.shape}")
    if not np.isfinite(sigma).all():
        raise ValueError("sigma_daily must be finite")
    if n_intervals < 1:
        raise ValueError(f"n_intervals must be >= 1; got {n_intervals}")
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1; got {n_days}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    n_assets = sigma.shape[0]
    interval_cov = sigma / n_intervals
    try:
        chol = np.linalg.cholesky(interval_cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "sigma_daily / n_intervals must be positive definite for Cholesky"
        ) from exc

    # z ~ N(0, I); (z @ L') has covariance L L' = interval_cov.
    noise = rng.standard_normal((n_days, n_intervals, n_assets))
    return noise @ chol.T
