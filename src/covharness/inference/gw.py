"""One-step Giacomini-White tests of conditional predictive ability.

For methods A and B the project differential is

    d_t = L_{A,t} - L_{B,t}.

Negative d_t means A has lower loss on date t. Instruments h_t are known at
the forecast origin. The one-step moment and statistic are

    Z_t = h_t * d_t,
    Zbar = mean_t Z_t,
    Omega_hat = (1/T) sum_t Z_t Z_t',
    GW = T * Zbar' Omega_hat^{-1} Zbar ~ chi^2_q.

Z_t is not demeaned. The default one-step covariance is the outer product,
not Bartlett / Newey-West HAC. A nonzero constant differential remains valid
when the instruments have full column rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from covharness.features.origin_state import measurement_stress_instruments
from covharness.inference.differentials import as_1d_finite, loss_differential
from covharness.inference.exceptions import (
    DegenerateGWCovarianceError,
    RankDeficientGWInstrumentsError,
)
from covharness.losses.contracts import (
    InvalidCovarianceMatrixError,
    as_square_finite,
    require_symmetric,
)
from covharness.losses.localization import implied_correlation

GW_ALPHA = 0.05
GW_FAMILY_SIZE = 2
GW_FAMILY_ALPHA = 0.05
GW_BONFERRONI_CUTOFF = 0.025
GW_ESTIMATOR = "outer_product"
GW_HORIZON = 1
MARKET_STATE_SPECIFICATION = "origin_log_mean_variance_and_mean_correlation"


@dataclass(frozen=True)
class GWResult:
    """One-step Giacomini-White test for one instrument specification."""

    n_observations: int
    n_instruments: int
    mean_differential: float
    moment_means: NDArray[np.floating]
    omega: NDArray[np.floating]
    statistic: float
    df: int
    p_value: float
    alpha: float
    estimator: str
    horizon: int


@dataclass(frozen=True)
class BonferroniFamilyResult:
    """Bonferroni adjustment for a pre-specified family of exactly two tests."""

    p_value_a: float
    p_value_b: float
    p_value_a_bonferroni: float
    p_value_b_bonferroni: float
    family_size: int
    family_alpha: float
    per_test_cutoff: float
    label_a: str
    label_b: str


@dataclass(frozen=True)
class GWFamilyResult:
    """Two separate GW specifications with Bonferroni metadata."""

    market: GWResult
    measurement: GWResult
    p_value_market: float
    p_value_measurement: float
    p_value_market_bonferroni: float
    p_value_measurement_bonferroni: float
    family_size: int
    family_alpha: float
    per_test_cutoff: float


def as_finite_instrument_matrix(
    instruments: ArrayLike, name: str = "instruments"
) -> NDArray[np.floating]:
    """Copy ``instruments`` to a finite ``(T, q)`` matrix."""
    array = np.array(instruments, dtype=float, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-d array of shape (T, q); got {array.shape}")
    n_obs, n_instruments = array.shape
    if n_obs < 2:
        raise ValueError(f"{name} requires at least two evaluation dates")
    if n_instruments < 1:
        raise ValueError(f"{name} requires at least one instrument column")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite (NaN and inf are rejected)")
    return array


def _as_origin_covariance_panel(
    origin_covariances: ArrayLike, name: str
) -> NDArray[np.floating]:
    """Copy a finite ``(T, N, N)`` panel of origin-day covariance matrices."""
    array = np.array(origin_covariances, dtype=float, copy=True)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(
            f"{name} must have shape (T, N, N); got {array.shape}"
        )
    if array.shape[0] < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite (NaN and inf are rejected)")
    return array


def market_state_instruments(origin_covariances: ArrayLike) -> NDArray[np.floating]:
    """Return origin-day market-state instruments of shape ``(T, 3)``.

    Column 0 is 1. Column 1 is ``log(mean_i S_ii,t)``. Column 2 is the mean
    unique off-diagonal implied correlation of origin-day ``S_t``. The series
    are not standardized. Target-day matrices are not used.
    """
    panel = _as_origin_covariance_panel(origin_covariances, "origin_covariances")
    n_obs, n_assets, _ = panel.shape
    if n_assets < 2:
        raise ValueError("market-state instruments require at least two assets")

    instruments = np.empty((n_obs, 3), dtype=float)
    instruments[:, 0] = 1.0
    for time in range(n_obs):
        matrix = as_square_finite(panel[time], "origin S_t")
        require_symmetric(matrix, "origin S_t")
        # Average realized variance is the mean origin-day diagonal.
        mean_variance = float(np.mean(np.diag(matrix)))
        if mean_variance <= 0.0:
            raise InvalidCovarianceMatrixError(
                "origin-day average realized variance must be strictly positive "
                "before the log transform. It is not clipped."
            )
        correlation = implied_correlation(matrix, "origin S_t")
        # Unique pairs i < j, excluding the diagonal.
        off_diag = correlation[np.triu_indices(n_assets, k=1)]
        instruments[time, 1] = float(np.log(mean_variance))
        instruments[time, 2] = float(np.mean(off_diag))
    return instruments


def market_state_augmenting_instruments(
    origin_covariances: ArrayLike,
) -> NDArray[np.floating]:
    """Return the two non-constant market-state columns, without the intercept."""
    return market_state_instruments(origin_covariances)[:, 1:]


def bonferroni_pair(
    p_value_a: float,
    p_value_b: float,
    *,
    family_alpha: float = GW_FAMILY_ALPHA,
    label_a: str = "a",
    label_b: str = "b",
) -> BonferroniFamilyResult:
    """Return Bonferroni metadata for a family of exactly two tests."""
    if family_alpha <= 0.0 or family_alpha >= 1.0:
        raise ValueError("family_alpha must lie in (0, 1)")
    p_a = float(p_value_a)
    p_b = float(p_value_b)
    if not np.isfinite(p_a) or not np.isfinite(p_b):
        raise ValueError("p-values must be finite")
    if p_a < 0.0 or p_a > 1.0 or p_b < 0.0 or p_b > 1.0:
        raise ValueError("p-values must lie in [0, 1]")
    return BonferroniFamilyResult(
        p_value_a=p_a,
        p_value_b=p_b,
        p_value_a_bonferroni=float(min(1.0, 2.0 * p_a)),
        p_value_b_bonferroni=float(min(1.0, 2.0 * p_b)),
        family_size=GW_FAMILY_SIZE,
        family_alpha=float(family_alpha),
        per_test_cutoff=float(family_alpha / 2.0),
        label_a=label_a,
        label_b=label_b,
    )


def _require_full_column_rank(instruments: NDArray[np.floating]) -> None:
    """Reject duplicate or collinear instrument directions before forming Z."""
    n_instruments = instruments.shape[1]
    rank = int(np.linalg.matrix_rank(instruments))
    if rank < n_instruments:
        raise RankDeficientGWInstrumentsError(
            "GW instruments do not have full column rank. Duplicate or "
            "collinear directions are rejected. No column was dropped."
        )


def giacomini_white(
    loss_a: ArrayLike,
    loss_b: ArrayLike,
    instruments: ArrayLike,
    *,
    alpha: float = GW_ALPHA,
) -> GWResult:
    """One-step GW test from paired losses and origin-measurable instruments."""
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    differential = loss_differential(loss_a, loss_b)
    instrument_matrix = as_finite_instrument_matrix(instruments)
    if instrument_matrix.shape[0] != differential.shape[0]:
        raise ValueError(
            "instruments and losses must have the same number of dates; "
            f"got {instrument_matrix.shape[0]} and {differential.shape[0]}"
        )
    # Rank-check h before multiplying by d.
    _require_full_column_rank(instrument_matrix)

    # Z_t = h_t * d_t. Do not demean.
    moments = instrument_matrix * differential[:, np.newaxis]
    n_obs, n_instruments = moments.shape
    moment_means = moments.mean(axis=0)
    omega = moments.T @ moments / n_obs
    if not np.isfinite(omega).all():
        raise DegenerateGWCovarianceError(
            "GW outer-product covariance is not finite. No repair was applied."
        )
    omega_rank = int(np.linalg.matrix_rank(omega))
    if omega_rank < n_instruments:
        raise DegenerateGWCovarianceError(
            "GW outer-product covariance is singular. All-zero moments and "
            "other rank-deficient Z'Z/T cases are rejected. No ridge or "
            "pseudo-inverse was applied."
        )
    # Solve Omega^{-1} Zbar.
    solved = np.linalg.solve(omega, moment_means)
    statistic = float(n_obs * moment_means @ solved)
    p_value = float(chi2.sf(statistic, df=n_instruments))
    return GWResult(
        n_observations=n_obs,
        n_instruments=n_instruments,
        mean_differential=float(differential.mean()),
        moment_means=moment_means,
        omega=omega,
        statistic=statistic,
        df=n_instruments,
        p_value=p_value,
        alpha=float(alpha),
        estimator=GW_ESTIMATOR,
        horizon=GW_HORIZON,
    )


def giacomini_white_two_specifications(
    loss_a: ArrayLike,
    loss_b: ArrayLike,
    instruments_market: ArrayLike,
    instruments_measurement: ArrayLike,
    *,
    alpha: float = GW_ALPHA,
    family_alpha: float = GW_FAMILY_ALPHA,
) -> GWFamilyResult:
    """Run two GW specifications separately and attach Bonferroni metadata.

    The six instrument columns are never stacked into one omnibus test.
    The market-state matrix is built by ``market_state_instruments``. The
    measurement/stress matrix is built by ``measurement_stress_instruments``
    from origin-day synchronized returns.
    """
    loss_a_copy = as_1d_finite(loss_a, "loss_a")
    loss_b_copy = as_1d_finite(loss_b, "loss_b")
    market = giacomini_white(
        loss_a_copy, loss_b_copy, instruments_market, alpha=alpha
    )
    measurement = giacomini_white(
        loss_a_copy, loss_b_copy, instruments_measurement, alpha=alpha
    )
    family = bonferroni_pair(
        market.p_value,
        measurement.p_value,
        family_alpha=family_alpha,
        label_a="market",
        label_b="measurement",
    )
    return GWFamilyResult(
        market=market,
        measurement=measurement,
        p_value_market=family.p_value_a,
        p_value_measurement=family.p_value_b,
        p_value_market_bonferroni=family.p_value_a_bonferroni,
        p_value_measurement_bonferroni=family.p_value_b_bonferroni,
        family_size=family.family_size,
        family_alpha=family.family_alpha,
        per_test_cutoff=family.per_test_cutoff,
    )
