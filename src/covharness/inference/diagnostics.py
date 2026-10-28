"""Diagnostics for a pairwise loss-differential series.

These quantities describe serial dependence in ``d_t``. They are not
additional hypothesis tests.

    C_t = sum_{s<=t} d_s

    rho_j = gamma_j / gamma_0     (sample ACF, 1/T autocovariances)

    kappa = HAC_long_run_variance / sample_variance

    T_eff = T / kappa

``kappa`` and ``T_eff`` are left undefined when the sample variance is zero.
``T_eff`` is not clipped into ``[1, T]``. Under negative serial correlation
``kappa`` can be below one and ``T_eff`` can exceed ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from covharness.inference.differentials import as_1d_finite
from covharness.inference.exceptions import DegenerateLossDifferentialError
from covharness.inference.hac import (
    bartlett_weights,
    hac_long_run_variance,
    resolve_hac_lag,
    sample_autocovariances,
)


@dataclass(frozen=True)
class LossDifferentialDiagnostic:
    """Reusable pairwise diagnostics. Not a hypothesis test."""

    n_observations: int
    mean_differential: float
    cumulative: NDArray[np.floating]
    acf: NDArray[np.floating]
    acf_lags: NDArray[np.int_]
    sample_variance: float
    hac_long_run_variance: float | None
    kappa: float | None
    t_eff: float | None
    hac_maxlags: int
    notes: str


def loss_differential_diagnostics(
    differential: ArrayLike,
    *,
    acf_lags: int = 20,
    maxlags: int | None = None,
) -> LossDifferentialDiagnostic:
    """Return cumulative sums, ACF, kappa, and T_eff for ``d_t``."""
    series = as_1d_finite(differential, "differential")
    n_obs = int(series.shape[0])
    if acf_lags < 1:
        raise ValueError("acf_lags must be at least 1")
    n_acf = min(int(acf_lags), n_obs - 1)
    if n_acf < 1:
        raise ValueError("ACF requires at least two observations")

    cumulative = np.cumsum(series)
    centered = series - series.mean()
    gammas = sample_autocovariances(centered, n_acf)
    sample_variance = float(gammas[0])
    lags = np.arange(1, n_acf + 1, dtype=int)
    if sample_variance == 0.0:
        acf = np.full(n_acf, np.nan)
        acf_note = "sample variance is zero, so ACF is undefined"
    else:
        acf = gammas[1:] / sample_variance
        acf_note = ""

    # Record the HAC lag that will be attempted, including when HAC later fails.
    attempted_lag, _ = resolve_hac_lag(n_obs, maxlags)
    try:
        hac = hac_long_run_variance(series, maxlags=maxlags)
        omega = hac.long_run_variance
        hac_lag = hac.maxlags
        kappa = omega / sample_variance if sample_variance > 0.0 else None
        t_eff = (n_obs / kappa) if (kappa is not None and kappa > 0.0) else None
        notes = acf_note
        if kappa is not None and kappa < 1.0:
            extra = (
                "kappa < 1 under negative serial correlation, so T_eff exceeds T. "
                "The unclipped mathematical value is reported."
            )
            notes = f"{notes}; {extra}".strip("; ")
    except DegenerateLossDifferentialError as exc:
        omega = None
        hac_lag = attempted_lag
        kappa = None
        t_eff = None
        notes = str(exc) if not acf_note else f"{acf_note}; {exc}"

    return LossDifferentialDiagnostic(
        n_observations=n_obs,
        mean_differential=float(series.mean()),
        cumulative=cumulative,
        acf=np.asarray(acf, dtype=float),
        acf_lags=lags,
        sample_variance=sample_variance,
        hac_long_run_variance=omega,
        kappa=None if kappa is None else float(kappa),
        t_eff=None if t_eff is None else float(t_eff),
        hac_maxlags=hac_lag,
        notes=notes,
    )


def sample_acf(differential: ArrayLike, n_lags: int) -> NDArray[np.floating]:
    """Return sample ACF at lags ``1, ..., n_lags`` using the ``1/T`` divisor."""
    series = as_1d_finite(differential, "differential")
    n_obs = int(series.shape[0])
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if n_lags >= n_obs:
        raise ValueError("n_lags must be smaller than the sample length")
    centered = series - series.mean()
    gammas = sample_autocovariances(centered, n_lags)
    if gammas[0] == 0.0:
        raise DegenerateLossDifferentialError(
            "sample variance is zero, so ACF is undefined. No jitter was added."
        )
    return np.asarray(gammas[1:] / gammas[0], dtype=float)


def ar1_effective_sample_size(n_observations: int, rho: float) -> float | None:
    """Return ``T(1-rho)/(1+rho)``, or None when the AR(1) formula is undefined."""
    if n_observations < 1:
        raise ValueError("n_observations must be at least 1")
    if not np.isfinite(rho):
        return None
    if abs(float(rho)) >= 1.0:
        return None
    denom = 1.0 + float(rho)
    if denom == 0.0:
        return None
    return float(n_observations) * (1.0 - float(rho)) / denom


def refit_phase_index(
    n_observations: int,
    *,
    cadence: int = 21,
    first_is_refit: bool = True,
) -> NDArray[np.int_]:
    """Return days since the last refit, with ``k=0`` on each refit origin."""
    if n_observations < 1:
        raise ValueError("n_observations must be at least 1")
    if cadence < 1:
        raise ValueError("cadence must be at least 1")
    if not first_is_refit:
        raise ValueError("A1 refit-phase labels require the first origin to be a refit")
    return np.arange(n_observations, dtype=int) % int(cadence)


def refit_phase_from_flags(refit: ArrayLike) -> NDArray[np.int_]:
    """Map a boolean refit flag series onto days since the last refit."""
    flags = np.asarray(refit, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("refit flags must be a one-dimensional array")
    if flags.size < 1:
        raise ValueError("refit flags must contain at least one origin")
    if not bool(flags[0]):
        raise ValueError("the first origin must be a refit")
    phase = np.empty(flags.size, dtype=int)
    since = 0
    # Reset the day-since-refit counter on each refit origin.
    for time, is_refit in enumerate(flags):
        if is_refit:
            since = 0
        phase[time] = since
        since += 1
    return phase


@dataclass(frozen=True)
class RefitCycleMeanResult:
    """Mean loss differential by days since the last parameter refit."""

    occupied_phases: NDArray[np.int_]
    counts: NDArray[np.int_]
    means: NDArray[np.floating]
    n_observations: int
    cadence: int
    n_complete_cycles: int
    final_cycle_length: int


def refit_cycle_means(
    differential: ArrayLike,
    phase: ArrayLike,
    *,
    cadence: int = 21,
) -> RefitCycleMeanResult:
    """Return occupancy and mean ``d_t`` for each occupied refit phase ``k``."""
    series = as_1d_finite(differential, "differential")
    labels = np.asarray(phase, dtype=int)
    if labels.ndim != 1 or labels.shape[0] != series.shape[0]:
        raise ValueError("phase labels must match the differential length")
    if np.any(labels < 0):
        raise ValueError("phase labels cannot be negative")
    occupied = np.unique(labels)
    counts = np.array([(labels == k).sum() for k in occupied], dtype=int)
    means = np.array([series[labels == k].mean() for k in occupied], dtype=float)
    n_obs = int(series.shape[0])
    n_complete = int(n_obs // cadence)
    remainder = int(n_obs % cadence)
    return RefitCycleMeanResult(
        occupied_phases=occupied,
        counts=counts,
        means=means,
        n_observations=n_obs,
        cadence=int(cadence),
        n_complete_cycles=n_complete,
        final_cycle_length=remainder,
    )


@dataclass(frozen=True)
class EqualPhaseMeanWaldResult:
    """Descriptive Wald test that mean ``d_t`` is equal across refit phases."""

    statistic: float | None
    df: int | None
    p_value: float | None
    hac_maxlags: int
    n_phases: int
    coefficients: NDArray[np.floating]
    occupied_phases: NDArray[np.int_]
    notes: str


def _bartlett_score_long_run_covariance(
    scores: NDArray[np.floating],
    *,
    maxlags: int,
) -> NDArray[np.floating]:
    """Bartlett HAC of a daily score matrix using the ``1/T`` divisor."""
    n_obs, n_coef = scores.shape
    if n_obs < 2:
        raise ValueError("HAC requires at least two daily scores")
    centered = scores - scores.mean(axis=0, keepdims=True)
    omega = (centered.T @ centered) / n_obs
    weights = bartlett_weights(maxlags)
    for lag_index in range(1, maxlags + 1):
        gamma = (centered[lag_index:].T @ centered[:-lag_index]) / n_obs
        omega = omega + weights[lag_index - 1] * (gamma + gamma.T)
    if not np.isfinite(omega).all():
        raise DegenerateLossDifferentialError(
            "HAC long-run covariance of dummy scores is not finite. "
            "No repair was applied."
        )
    return omega


def equal_refit_phase_means_wald(
    differential: ArrayLike,
    phase: ArrayLike,
    *,
    maxlags: int = 29,
) -> EqualPhaseMeanWaldResult:
    """Test equal phase means by dummy OLS with Bartlett HAC at ``maxlags``.

    The design is a full set of occupied ``k`` dummies and no intercept. The
    null is that all dummy coefficients are equal. The test is descriptive.
    """
    series = as_1d_finite(differential, "differential")
    labels = np.asarray(phase, dtype=int)
    if labels.shape != series.shape:
        raise ValueError("phase labels must match the differential length")
    occupied = np.unique(labels)
    n_obs = int(series.shape[0])
    n_phases = int(occupied.size)
    # Freeze the Wald HAC lag at the caller value, including the A1 lag of 29.
    lag, _ = resolve_hac_lag(n_obs, maxlags)
    if n_phases < 2:
        return EqualPhaseMeanWaldResult(
            statistic=None,
            df=None,
            p_value=None,
            hac_maxlags=lag,
            n_phases=n_phases,
            coefficients=np.array([float(series.mean())], dtype=float),
            occupied_phases=occupied,
            notes="fewer than two occupied phases, so equal-mean Wald is undefined",
        )

    # Build the no-intercept dummy design and OLS group means.
    design = np.column_stack([(labels == k).astype(float) for k in occupied])
    gram = design.T @ design
    coefficients = np.linalg.solve(gram, design.T @ series)
    residuals = series - design @ coefficients
    scores = design * residuals[:, np.newaxis]
    omega = _bartlett_score_long_run_covariance(scores, maxlags=lag)
    covariance = np.linalg.solve(gram, np.linalg.solve(gram, n_obs * omega).T).T
    df = n_phases - 1
    # Contrast every later phase mean against the first occupied phase.
    restriction = np.column_stack([np.ones(df), -np.eye(df)])
    gap = restriction @ coefficients
    middle = restriction @ covariance @ restriction.T
    rank = int(np.linalg.matrix_rank(middle))
    if rank < df or not np.isfinite(middle).all():
        return EqualPhaseMeanWaldResult(
            statistic=None,
            df=df,
            p_value=None,
            hac_maxlags=lag,
            n_phases=n_phases,
            coefficients=np.asarray(coefficients, dtype=float),
            occupied_phases=occupied,
            notes="HAC sandwich of dummy scores is singular. No repair was applied.",
        )
    statistic = float(gap @ np.linalg.solve(middle, gap))
    if not np.isfinite(statistic):
        return EqualPhaseMeanWaldResult(
            statistic=None,
            df=df,
            p_value=None,
            hac_maxlags=lag,
            n_phases=n_phases,
            coefficients=np.asarray(coefficients, dtype=float),
            occupied_phases=occupied,
            notes="Wald statistic is not finite. No repair was applied.",
        )
    return EqualPhaseMeanWaldResult(
        statistic=statistic,
        df=df,
        p_value=float(chi2.sf(statistic, df=df)),
        hac_maxlags=lag,
        n_phases=n_phases,
        coefficients=np.asarray(coefficients, dtype=float),
        occupied_phases=occupied,
        notes="",
    )
