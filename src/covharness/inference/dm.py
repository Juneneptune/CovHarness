"""Diebold-Mariano test of equal average predictive accuracy.

Forecast losses are the primitive inputs. The test does not differentiate
through forecasting-model parameters.

For ``d_t = L_{A,t} - L_{B,t}``,

    DM = mean(d) / HAC_SE(mean(d)).

The null is ``H0: E[d_t] = 0``. Alternatives are

    two_sided  H1: E[d_t] != 0
    a_better   H1: E[d_t] < 0   (A has lower expected loss)
    b_better   H1: E[d_t] > 0   (B has lower expected loss)

p-values use the asymptotic N(0,1) reference of Diebold and Mariano (1995).
The Harvey-Leybourne-Newbold small-sample correction is not applied. The
repository design has not committed to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from covharness.inference.differentials import (
    ALTERNATIVE_A_BETTER,
    ALTERNATIVE_B_BETTER,
    ALTERNATIVE_TWO_SIDED,
    as_1d_finite,
    loss_differential,
    parse_alternative,
)
from covharness.inference.hac import hac_long_run_variance


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Pairwise DM test with Bartlett HAC standard errors."""

    n_observations: int
    mean_differential: float
    hac_long_run_variance: float
    standard_error: float
    statistic: float
    p_value: float
    alternative: str
    hac_kernel: str
    hac_maxlags: int
    hac_lag_rule: str
    hln_small_sample_correction: bool


def diebold_mariano(
    differential: ArrayLike,
    *,
    alternative: str = ALTERNATIVE_TWO_SIDED,
    maxlags: int | None = None,
) -> DieboldMarianoResult:
    """DM test on an already-computed loss-differential series ``d_t``."""
    series = as_1d_finite(differential, "differential")
    alt = parse_alternative(alternative)
    hac = hac_long_run_variance(series, maxlags=maxlags)
    statistic = hac.mean / hac.standard_error_of_mean
    p_value = _normal_p_value(statistic, alt)
    return DieboldMarianoResult(
        n_observations=hac.n_observations,
        mean_differential=hac.mean,
        hac_long_run_variance=hac.long_run_variance,
        standard_error=hac.standard_error_of_mean,
        statistic=float(statistic),
        p_value=float(p_value),
        alternative=alt,
        hac_kernel=hac.kernel,
        hac_maxlags=hac.maxlags,
        hac_lag_rule=hac.lag_rule,
        hln_small_sample_correction=False,
    )


def diebold_mariano_from_losses(
    loss_a: ArrayLike,
    loss_b: ArrayLike,
    *,
    alternative: str = ALTERNATIVE_TWO_SIDED,
    maxlags: int | None = None,
) -> DieboldMarianoResult:
    """DM test from paired loss series, using ``d_t = L_A,t - L_B,t``."""
    return diebold_mariano(
        loss_differential(loss_a, loss_b),
        alternative=alternative,
        maxlags=maxlags,
    )


def naive_iid_t_statistic(differential: ArrayLike) -> float:
    """IID t-statistic using the sample standard deviation.

    This is the labeled contrast that ignores serial correlation. It is not
    the production DM test. ``ddof=1`` sample variance is used.
    """
    series = as_1d_finite(differential, "differential")
    n_obs = series.shape[0]
    if n_obs < 2:
        raise ValueError("the IID t-statistic requires at least two observations")
    mean = float(series.mean())
    sample_std = float(series.std(ddof=1))
    if sample_std == 0.0:
        raise ValueError("IID t-statistic is undefined for a constant series")
    return mean / (sample_std / np.sqrt(n_obs))


def _normal_p_value(statistic: float, alternative: str) -> float:
    """Asymptotic N(0,1) p-value for the documented alternative."""
    if alternative == ALTERNATIVE_TWO_SIDED:
        return float(2.0 * norm.sf(abs(statistic)))
    if alternative == ALTERNATIVE_A_BETTER:
        return float(norm.cdf(statistic))
    if alternative == ALTERNATIVE_B_BETTER:
        return float(norm.sf(statistic))
    raise ValueError(f"unknown alternative {alternative!r}")
