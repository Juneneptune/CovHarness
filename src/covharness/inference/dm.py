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
from scipy.stats import norm, t as student_t

from covharness.inference.differentials import (
    ALTERNATIVE_A_BETTER,
    ALTERNATIVE_B_BETTER,
    ALTERNATIVE_TWO_SIDED,
    as_1d_finite,
    loss_differential,
    parse_alternative,
)
from covharness.inference.hac import (
    andrews_1991_bartlett_lag,
    bartlett_fixed_b_critical,
    bartlett_fixed_b_pvalue,
    ewc_long_run_variance,
    hac_long_run_variance,
    llsw_sqrt_lag,
    newey_west_1994_lags,
)


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
    reference: str = "normal"
    critical_value: float | None = None


def diebold_mariano(
    differential: ArrayLike,
    *,
    alternative: str = ALTERNATIVE_TWO_SIDED,
    maxlags: int | None = None,
    lag_rule: str | None = None,
    reference: str = "normal",
) -> DieboldMarianoResult:
    """DM test on an already-computed loss-differential series ``d_t``.

    ``lag_rule=None`` and ``reference="normal"`` preserve the historical
    Newey-West automatic lag and N(0,1) p-value. Named rules are opt-in.
    """
    series = as_1d_finite(differential, "differential")
    alt = parse_alternative(alternative)
    if reference == "ewc":
        return _ewc_diebold_mariano(series, alt)
    lag, lag_name = _resolve_named_lag(series, maxlags=maxlags, lag_rule=lag_rule)
    hac = hac_long_run_variance(series, maxlags=lag, lag_rule_name=lag_name)
    statistic = hac.mean / hac.standard_error_of_mean
    if reference == "normal":
        p_value = _normal_p_value(statistic, alt)
        critical = float(norm.ppf(0.975)) if alt == ALTERNATIVE_TWO_SIDED else float(norm.ppf(0.95))
    elif reference == "fixed_b":
        bandwidth = (hac.maxlags + 1) / hac.n_observations
        critical = bartlett_fixed_b_critical(bandwidth, 0.975 if alt == ALTERNATIVE_TWO_SIDED else 0.95)
        p_value = bartlett_fixed_b_pvalue(statistic, bandwidth)
        if alt != ALTERNATIVE_TWO_SIDED:
            p_value = p_value / 2.0
    else:
        raise ValueError(f"unknown DM reference {reference}")
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
        reference=reference,
        critical_value=float(critical),
    )


def _resolve_named_lag(
    series: np.ndarray,
    *,
    maxlags: int | None,
    lag_rule: str | None,
) -> tuple[int | None, str | None]:
    """Map a named lag rule onto an explicit lag. ``None`` keeps the default."""
    if lag_rule is None:
        return maxlags, None
    if lag_rule == "newey_west_1994":
        return newey_west_1994_lags(series.shape[0]), lag_rule
    if lag_rule == "four_newey_west_1994":
        return 4 * newey_west_1994_lags(series.shape[0]), lag_rule
    if lag_rule == "llsw_sqrt":
        return llsw_sqrt_lag(series.shape[0]), lag_rule
    if lag_rule == "andrews_1991":
        return andrews_1991_bartlett_lag(series), lag_rule
    raise ValueError(f"unknown DM lag rule {lag_rule}")


def _ewc_diebold_mariano(series: np.ndarray, alternative: str) -> DieboldMarianoResult:
    """EWC mean test with Student-t critical values. Not the default DM test."""
    omega, terms = ewc_long_run_variance(series)
    mean = float(series.mean())
    standard_error = float(np.sqrt(omega / series.shape[0]))
    statistic = mean / standard_error
    if alternative == ALTERNATIVE_TWO_SIDED:
        p_value = float(2.0 * student_t.sf(abs(statistic), terms))
        critical = float(student_t.ppf(0.975, terms))
    elif alternative == ALTERNATIVE_A_BETTER:
        p_value = float(student_t.cdf(statistic, terms))
        critical = float(student_t.ppf(0.05, terms))
    else:
        p_value = float(student_t.sf(statistic, terms))
        critical = float(student_t.ppf(0.95, terms))
    return DieboldMarianoResult(
        n_observations=int(series.shape[0]),
        mean_differential=mean,
        hac_long_run_variance=omega,
        standard_error=standard_error,
        statistic=float(statistic),
        p_value=p_value,
        alternative=alternative,
        hac_kernel="ewc",
        hac_maxlags=terms,
        hac_lag_rule="ewc_nu",
        hln_small_sample_correction=False,
        reference="student_t",
        critical_value=critical,
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
