"""Origin-day measurement and stress instruments from synchronized returns.

These series are constructed from the forecast-origin trading day only.
Target-day returns never enter. The constructions do not annualize, winsorize,
clip, or repair the input panel.

Aggregate realized quarticity is the cross-sectional mean of the per-asset
estimators

    RQ_i = (M / 3) * sum_j r[j, i]^4.

The jump state is the one-sided 1 percent Barndorff-Nielsen and Shephard
(2006) adjusted-ratio test applied to the equal-weight intraday market
return. It detects a jump in that common-market portfolio. It is not an
any-constituent-jumped indicator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from covharness.features.exceptions import InvalidOriginStateError

QUARTICITY_MIN_INTERVALS = 2
JUMP_MIN_INTERVALS = 4
JUMP_ALPHA = 0.01
JUMP_MU1 = float(np.sqrt(2.0 / np.pi))
JUMP_VARTHETA = float(np.pi**2 / 4.0 + np.pi - 5.0)
JUMP_CRITICAL_VALUE = float(norm.ppf(JUMP_ALPHA))
MEASUREMENT_STATE_SPECIFICATION = "origin_log_aggregate_rq_and_bns_market_jump"


@dataclass(frozen=True)
class QuarticityResult:
    """Cross-sectional mean of per-asset realized quarticity for one origin day."""

    n_intervals: int
    n_assets: int
    asset_quarticity: NDArray[np.floating]
    aggregate: float


@dataclass(frozen=True)
class BNSJumpResult:
    """BNS adjusted-ratio jump test on the equal-weight intraday market return.

    ``jump_indicator`` is 1 when the standardized statistic falls below the
    frozen one-sided normal critical value at alpha 0.01. Large negative
    values indicate jumps. The threshold is not estimated from the sample.
    """

    n_intervals: int
    realized_variance: float
    bipower_raw: float
    continuous_variance_estimate: float
    quadpower: float
    adjusted_ratio_statistic: float
    standardized_statistic: float
    alpha: float
    critical_value: float
    jump_indicator: int


def as_intraday_returns(
    returns: ArrayLike,
    *,
    min_intervals: int,
    name: str = "intraday_returns",
) -> NDArray[np.floating]:
    """Copy a finite ``(M, N)`` synchronized return matrix. Inputs are not mutated."""
    array = np.array(returns, dtype=float, copy=True)
    if array.ndim != 2:
        raise InvalidOriginStateError(
            f"{name} must have shape (M, N); got {array.shape}"
        )
    n_intervals, n_assets = array.shape
    if n_intervals < min_intervals:
        raise InvalidOriginStateError(
            f"{name} requires at least {min_intervals} intraday intervals; "
            f"got M={n_intervals}"
        )
    if n_assets < 1:
        raise InvalidOriginStateError(f"{name} requires at least one asset")
    if not np.isfinite(array).all():
        raise InvalidOriginStateError(
            f"{name} must be finite (NaN and inf are rejected)"
        )
    return array


def as_intraday_return_panel(
    returns: ArrayLike,
    *,
    min_intervals: int,
    name: str = "origin_intraday_returns",
) -> NDArray[np.floating]:
    """Copy a finite ``(T, M, N)`` origin-day return panel. Inputs are not mutated."""
    array = np.array(returns, dtype=float, copy=True)
    if array.ndim != 3:
        raise InvalidOriginStateError(
            f"{name} must have shape (T, M, N); got {array.shape}"
        )
    n_obs, n_intervals, n_assets = array.shape
    if n_obs < 1:
        raise InvalidOriginStateError(f"{name} must be non-empty")
    if n_intervals < min_intervals:
        raise InvalidOriginStateError(
            f"{name} requires at least {min_intervals} intraday intervals; "
            f"got M={n_intervals}"
        )
    if n_assets < 1:
        raise InvalidOriginStateError(f"{name} requires at least one asset")
    if not np.isfinite(array).all():
        raise InvalidOriginStateError(
            f"{name} must be finite (NaN and inf are rejected)"
        )
    return array


def realized_quarticity_aggregate(returns: ArrayLike) -> QuarticityResult:
    """Return per-asset and mean realized quarticity from one origin day.

    ``RQ_i = (M / 3) * sum_j r[j, i]^4`` and ``RQ_agg = mean_i RQ_i``.
    The aggregate must be strictly positive before a later log transform.
    """
    panel = as_intraday_returns(
        returns, min_intervals=QUARTICITY_MIN_INTERVALS, name="intraday_returns"
    )
    n_intervals, n_assets = panel.shape
    # Scale fourth powers by M/3, then average.
    asset_quarticity = (n_intervals / 3.0) * np.sum(panel**4, axis=0)
    aggregate = float(np.mean(asset_quarticity))
    if not np.isfinite(aggregate):
        raise InvalidOriginStateError(
            "aggregate realized quarticity is not finite. No repair was applied."
        )
    if aggregate <= 0.0:
        raise InvalidOriginStateError(
            "aggregate realized quarticity must be strictly positive before "
            "the log transform. It is not clipped."
        )
    return QuarticityResult(
        n_intervals=n_intervals,
        n_assets=n_assets,
        asset_quarticity=asset_quarticity,
        aggregate=aggregate,
    )


def bns_market_jump(returns: ArrayLike) -> BNSJumpResult:
    """One-sided BNS adjusted-ratio jump test on the equal-weight market return.

    The test uses origin-day synchronized returns only. It is a common-market
    jump state, not a collection of 30 asset-level tests.
    """
    panel = as_intraday_returns(
        returns, min_intervals=JUMP_MIN_INTERVALS, name="intraday_returns"
    )
    n_intervals = int(panel.shape[0])
    # Equal-weight market return.
    market = panel.mean(axis=1)
    delta = 1.0 / n_intervals
    realized_variance = float(np.sum(market**2))
    if realized_variance <= 0.0:
        raise InvalidOriginStateError(
            "BNS market realized variance must be strictly positive. "
            "No epsilon floor was applied."
        )
    # Adjacent absolute products.
    bipower_raw = float(np.sum(np.abs(market[1:]) * np.abs(market[:-1])))
    if bipower_raw <= 0.0:
        raise InvalidOriginStateError(
            "BNS market bipower variation must be strictly positive. "
            "No epsilon floor was applied."
        )
    # Four-lag product, scaled by 1/delta.
    quad_terms = (
        np.abs(market[3:])
        * np.abs(market[2:-1])
        * np.abs(market[1:-2])
        * np.abs(market[:-3])
    )
    quadpower = float((1.0 / delta) * np.sum(quad_terms))
    if (not np.isfinite(quadpower)) or quadpower < 0.0:
        raise InvalidOriginStateError(
            "BNS market quadpower must be finite and nonnegative. "
            "No replacement was applied."
        )
    mu1_inv_sq = 1.0 / (JUMP_MU1**2)
    continuous_variance = float(mu1_inv_sq * bipower_raw)
    # BNS adjustment max(1, QP / BV^2).
    relative_quad = quadpower / (bipower_raw**2)
    adjustment = float(np.sqrt(max(1.0, relative_quad)))
    ratio_term = continuous_variance / realized_variance - 1.0
    adjusted = float((delta ** (-0.5)) / adjustment * ratio_term)
    standardized = float(adjusted / np.sqrt(JUMP_VARTHETA))
    jump_indicator = int(standardized < JUMP_CRITICAL_VALUE)
    return BNSJumpResult(
        n_intervals=n_intervals,
        realized_variance=realized_variance,
        bipower_raw=bipower_raw,
        continuous_variance_estimate=continuous_variance,
        quadpower=quadpower,
        adjusted_ratio_statistic=adjusted,
        standardized_statistic=standardized,
        alpha=JUMP_ALPHA,
        critical_value=JUMP_CRITICAL_VALUE,
        jump_indicator=jump_indicator,
    )


def measurement_stress_instruments(origin_intraday_returns: ArrayLike) -> NDArray[np.floating]:
    """Return origin-day measurement/stress instruments of shape ``(T, 3)``.

    Column 0 is 1. Column 1 is ``log(RQ_agg,t)``. Column 2 is the BNS
    equal-weight market jump indicator. The series are not standardized.
    Target-day returns are not used.
    """
    panel = as_intraday_return_panel(
        origin_intraday_returns,
        min_intervals=JUMP_MIN_INTERVALS,
        name="origin_intraday_returns",
    )
    n_obs = int(panel.shape[0])
    instruments = np.empty((n_obs, 3), dtype=float)
    instruments[:, 0] = 1.0
    for time in range(n_obs):
        quarticity = realized_quarticity_aggregate(panel[time])
        jump = bns_market_jump(panel[time])
        instruments[time, 1] = float(np.log(quarticity.aggregate))
        instruments[time, 2] = float(jump.jump_indicator)
    return instruments
