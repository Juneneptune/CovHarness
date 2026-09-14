"""Joint Politis-Romano stationary bootstrap for multi-model loss matrices.

Time indices are drawn once and applied to every model column. Columns are
never resampled independently. The default expected block length is

    ell = max(2, floor(T^{1/3})),

so the restart probability is q = 1/ell. This rule is a function of T only.
It does not inspect loss autocorrelations or model rankings.

Hansen (2005) requires q_T -> 0 and T q_T^2 -> infinity. The cube-root
choice satisfies both. A sqrt(T) block length would set q = T^{-1/2} and
violate the second condition.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

BOOTSTRAP_METHOD = "stationary"
BOOTSTRAP_SEED = 20260913
DEFAULT_N_BOOT = 5000


def default_block_length(n_observations: int) -> int:
    """Return max(2, floor(T^{1/3}))."""
    if n_observations < 2:
        raise ValueError("the stationary bootstrap requires at least two observations")
    # Cube root via cbrt so perfect cubes such as 1000 are not floored by roundoff.
    return max(2, int(np.floor(np.cbrt(n_observations))))


def restart_probability(n_observations: int, block_length: int | None = None) -> float:
    """Return q = 1 / ell for the frozen block-length rule or an override."""
    length = default_block_length(n_observations) if block_length is None else int(block_length)
    if length < 1:
        raise ValueError("block_length must be at least 1")
    return 1.0 / float(length)


def as_finite_loss_matrix(losses: ArrayLike, name: str = "losses") -> NDArray[np.floating]:
    """Copy ``losses`` to a finite (T, M) array. The input is not mutated."""
    array = np.array(losses, dtype=float, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-d array of shape (T, M); got {array.shape}")
    n_obs, n_models = array.shape
    if n_obs < 2:
        raise ValueError(f"{name} requires at least two evaluation dates")
    if n_models < 1:
        raise ValueError(f"{name} requires at least one model column")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite (NaN and inf are rejected)")
    return array


def stationary_bootstrap_indices(
    n_observations: int,
    n_boot: int = DEFAULT_N_BOOT,
    *,
    block_length: int | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> NDArray[np.int64]:
    """Return joint time-index draws of shape ``(n_boot, T)``.

    Indices are 0-based and wrap circularly, so a continuation from ``T-1``
    is ``0``. A restart draws a new uniform origin in ``{0, ..., T-1}``.
    """
    if n_observations < 2:
        raise ValueError("the stationary bootstrap requires at least two observations")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    length = default_block_length(n_observations) if block_length is None else int(block_length)
    if length < 1:
        raise ValueError("block_length must be at least 1")
    restart_q = 1.0 / float(length)

    # Draw the first index and then apply geometric restarts with wraparound.
    rng = np.random.default_rng(seed)
    indices = np.empty((n_boot, n_observations), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n_observations, size=n_boot)
    for time in range(1, n_observations):
        restart = rng.random(n_boot) < restart_q
        restarted = rng.integers(0, n_observations, size=n_boot)
        continued = (indices[:, time - 1] + 1) % n_observations
        indices[:, time] = np.where(restart, restarted, continued)
    return indices
