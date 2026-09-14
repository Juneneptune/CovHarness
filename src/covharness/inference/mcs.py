"""Hansen-Lunde-Nason (2011) Model Confidence Set.

The input is a loss matrix ``L[t, m]`` with lower loss better. MCS is not
pooled across losses or covariance proxies. There is no SPA-style benchmark.

Pairwise differentials on a current set M are

    d[i, j, t] = L[t, i] - L[t, j],

so a positive value means i is worse than j. The vs-rest differential is

    dbar_i_dot = mean_loss_i - average mean loss on the current set.

Two coherent procedures are implemented.

    Range.  T_R = max |t_ij|,  e_R = argmax_i sup_j t_ij
    Max.    T_max = max t_i_dot,  e_max = argmax_i t_i_dot

The primary SCREEN procedure is the range pair. The max pair is a companion.
Studentization uses standard errors, not raw variances.

Bootstrap indices are drawn once and reused at every elimination step.
EPA p-values use ``mean(T* >= T)``. Model MCS p-values are the running
maximum along the elimination path. The last surviving model has p-value 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.inference.bootstrap import (
    BOOTSTRAP_METHOD,
    BOOTSTRAP_SEED,
    DEFAULT_N_BOOT,
    as_finite_loss_matrix,
    default_block_length,
    restart_probability,
    stationary_bootstrap_indices,
)
from covharness.inference.exceptions import DegenerateMCSDifferentialError

PROCEDURE_RANGE = "range"
PROCEDURE_MAX = "max"
VALID_PROCEDURES = (PROCEDURE_RANGE, PROCEDURE_MAX)
MCS_ALPHA = 0.10


@dataclass(frozen=True)
class MCSResult:
    """Hansen-Lunde-Nason MCS for one coherent (statistic, elimination) pair."""

    n_observations: int
    n_models: int
    procedure: str
    alpha: float
    mean_losses: NDArray[np.floating]
    mcs_pvalues: NDArray[np.floating]
    in_mcs: NDArray[np.bool_]
    elimination_order: NDArray[np.int64]
    epa_pvalues_along_path: NDArray[np.floating]
    statistics_along_path: NDArray[np.floating]
    block_length: int
    restart_probability: float
    n_boot: int
    seed: int
    bootstrap_method: str


def model_confidence_set(
    losses: ArrayLike,
    *,
    procedure: str = PROCEDURE_RANGE,
    alpha: float = MCS_ALPHA,
    block_length: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> MCSResult:
    """Return MCS p-values and membership for one coherent HLN procedure."""
    loss_matrix = as_finite_loss_matrix(losses, "losses")
    n_obs, n_models = loss_matrix.shape
    key = str(procedure).strip().lower()
    if key not in VALID_PROCEDURES:
        raise ValueError(f"procedure must be one of {VALID_PROCEDURES}; got {procedure!r}")
    if n_models < 1:
        raise ValueError("MCS requires at least one model")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    # Reject nonzero constant pairs once, before any studentized ratio.
    _reject_nonzero_constant_pairs(loss_matrix)

    length = default_block_length(n_obs) if block_length is None else int(block_length)
    restart_q = restart_probability(n_obs, length)
    mean_losses = loss_matrix.mean(axis=0)
    if n_models == 1:
        return MCSResult(
            n_observations=n_obs,
            n_models=1,
            procedure=key,
            alpha=float(alpha),
            mean_losses=np.asarray(mean_losses, dtype=float),
            mcs_pvalues=np.array([1.0], dtype=float),
            in_mcs=np.array([True], dtype=bool),
            elimination_order=np.array([0], dtype=np.int64),
            epa_pvalues_along_path=np.array([], dtype=float),
            statistics_along_path=np.array([], dtype=float),
            block_length=length,
            restart_probability=restart_q,
            n_boot=int(n_boot),
            seed=int(seed),
            bootstrap_method=BOOTSTRAP_METHOD,
        )

    # Draw the index matrix once. Every elimination step reuses these rows.
    indices = stationary_bootstrap_indices(
        n_obs, n_boot, block_length=length, seed=seed
    )
    boot_means = loss_matrix[indices].mean(axis=1)
    zeta = boot_means - mean_losses

    remaining = list(range(n_models))
    elimination_order: list[int] = []
    epa_pvalues: list[float] = []
    statistics: list[float] = []
    while len(remaining) > 1:
        statistic, victim, p_value = _epa_step(
            remaining, mean_losses, zeta, key
        )
        statistics.append(statistic)
        epa_pvalues.append(p_value)
        remaining.remove(victim)
        elimination_order.append(victim)
    elimination_order.append(remaining[0])

    # Running-max MCS p-values along the nested elimination path.
    mcs_pvalues = np.empty(n_models, dtype=float)
    running = 0.0
    for step, model in enumerate(elimination_order[:-1]):
        running = max(running, epa_pvalues[step])
        mcs_pvalues[model] = running
    mcs_pvalues[elimination_order[-1]] = 1.0
    in_mcs = mcs_pvalues >= float(alpha)
    return MCSResult(
        n_observations=n_obs,
        n_models=n_models,
        procedure=key,
        alpha=float(alpha),
        mean_losses=np.asarray(mean_losses, dtype=float),
        mcs_pvalues=mcs_pvalues,
        in_mcs=in_mcs,
        elimination_order=np.asarray(elimination_order, dtype=np.int64),
        epa_pvalues_along_path=np.asarray(epa_pvalues, dtype=float),
        statistics_along_path=np.asarray(statistics, dtype=float),
        block_length=length,
        restart_probability=restart_q,
        n_boot=int(n_boot),
        seed=int(seed),
        bootstrap_method=BOOTSTRAP_METHOD,
    )


def _reject_nonzero_constant_pairs(losses: NDArray[np.floating]) -> None:
    """Raise if any pair has a constant nonzero loss differential."""
    n_models = losses.shape[1]
    for i in range(n_models):
        for j in range(i + 1, n_models):
            diff = losses[:, i] - losses[:, j]
            if np.all(diff == diff[0]) and diff[0] != 0.0:
                raise DegenerateMCSDifferentialError(
                    f"loss columns {i} and {j} differ by a nonzero constant. "
                    "The MCS t-ratio is undefined. No jitter was added."
                )


def _epa_step(
    remaining: list[int],
    mean_losses: NDArray[np.floating],
    zeta: NDArray[np.floating],
    procedure: str,
) -> tuple[float, int, float]:
    """Return statistic, eliminated original index, and EPA p-value on this set."""
    idx = np.asarray(remaining, dtype=int)
    if procedure == PROCEDURE_RANGE:
        observed, t_star, scores = _range_components(idx, mean_losses, zeta)
    else:
        observed, t_star, scores = _max_components(idx, mean_losses, zeta)
    p_value = float(np.mean(t_star >= observed))
    # Tie-break by original column index. np.argmax keeps the first maximum.
    local = int(np.argmax(scores))
    victim = int(idx[local])
    return float(observed), victim, p_value


def _range_components(
    idx: NDArray[np.int64],
    mean_losses: NDArray[np.floating],
    zeta: NDArray[np.floating],
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Range statistic, bootstrap replicates, and e_R scores on the current set."""
    means = mean_losses[idx]
    dbar = means[:, None] - means[None, :]
    zeta_s = zeta[:, idx]
    # Bootstrap variance of each pairwise mean differential.
    pair_dev = zeta_s[:, :, None] - zeta_s[:, None, :]
    var_hat = np.mean(pair_dev * pair_dev, axis=0)
    t_ij = _studentize_pairwise(dbar, var_hat)
    t_star_ij = _studentize_pairwise_star(pair_dev, var_hat)
    observed = float(np.max(np.abs(t_ij)))
    t_star = np.max(np.abs(t_star_ij), axis=(1, 2))
    scores = np.max(t_ij, axis=1)
    return observed, t_star, scores


def _max_components(
    idx: NDArray[np.int64],
    mean_losses: NDArray[np.floating],
    zeta: NDArray[np.floating],
) -> tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Max vs-rest statistic, bootstrap replicates, and e_max scores."""
    means = mean_losses[idx]
    dbar = means - means.mean()
    zeta_s = zeta[:, idx]
    zeta_dot = zeta_s.mean(axis=1, keepdims=True)
    rest_dev = zeta_s - zeta_dot
    var_hat = np.mean(rest_dev * rest_dev, axis=0)
    t_i = _studentize_vector(dbar, var_hat)
    t_star_i = _studentize_star_vector(rest_dev, var_hat)
    observed = float(np.max(t_i))
    t_star = np.max(t_star_i, axis=1)
    return observed, t_star, t_i


def _studentize_pairwise(
    dbar: NDArray[np.floating],
    var_hat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return t_ij, using 0 for exact ties with zero bootstrap variance."""
    t_ij = np.zeros_like(dbar, dtype=float)
    positive = var_hat > 0.0
    t_ij[positive] = dbar[positive] / np.sqrt(var_hat[positive])
    return t_ij


def _studentize_pairwise_star(
    pair_dev: NDArray[np.floating],
    var_hat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Studentize bootstrap pairwise deviations with the observed SE."""
    se = np.sqrt(var_hat)
    t_star = np.zeros_like(pair_dev, dtype=float)
    positive = var_hat > 0.0
    t_star[:, positive] = pair_dev[:, positive] / se[positive]
    return t_star


def _studentize_vector(
    dbar: NDArray[np.floating],
    var_hat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return t_i_dot, using 0 when bootstrap variance is zero."""
    t_i = np.zeros_like(dbar, dtype=float)
    positive = var_hat > 0.0
    t_i[positive] = dbar[positive] / np.sqrt(var_hat[positive])
    return t_i


def _studentize_star_vector(
    rest_dev: NDArray[np.floating],
    var_hat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Studentize bootstrap vs-rest deviations with the observed SE."""
    se = np.sqrt(var_hat)
    t_star = np.zeros_like(rest_dev, dtype=float)
    positive = var_hat > 0.0
    t_star[:, positive] = rest_dev[:, positive] / se[positive]
    return t_star
