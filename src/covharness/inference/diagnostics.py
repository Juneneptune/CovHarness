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

from covharness.inference.differentials import as_1d_finite
from covharness.inference.exceptions import DegenerateLossDifferentialError
from covharness.inference.hac import (
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
