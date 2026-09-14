"""Clark-West nested comparison for scalar squared error.

The classical Clark and West (2007) adjustment is defined for nested models
compared under squared forecast error. Let model R be the restricted /
parsimonious model and model U the unrestricted model that nests R. With
scalar outcome ``y_t`` and forecasts ``f_{R,t}``, ``f_{U,t}``,

    d_t^{CW} = (y_t - f_{R,t})^2 - (y_t - f_{U,t})^2 + (f_{R,t} - f_{U,t})^2.

The usual Diebold-Mariano / HAC test is then applied to ``d_t^{CW}``. Under
the project's sign convention this differential is ``L_R - L_U`` plus the
Clark-West adjustment, so ``E[d^{CW}] > 0`` is the alternative that the
unrestricted model is more accurate.

This procedure is not applied automatically to models with more parameters.
The caller must pass ``nested=True``. It is not defined for reduced QLIKE,
squared Frobenius, or other matrix losses. A later nested covariance
comparison may need a loss-specific justified procedure rather than this
scalar formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.inference.differentials import (
    ALTERNATIVE_B_BETTER,
    as_1d_finite,
)
from covharness.inference.dm import DieboldMarianoResult, diebold_mariano
from covharness.inference.exceptions import ClarkWestScopeError


@dataclass(frozen=True)
class ClarkWestResult:
    """Clark-West test on an explicitly declared nested scalar-SE pair."""

    n_observations: int
    mean_adjusted_differential: float
    statistic: float
    p_value: float
    alternative: str
    nested_declared: bool
    loss: str
    dm: DieboldMarianoResult


def clark_west_squared_error(
    actual: ArrayLike,
    forecast_restricted: ArrayLike,
    forecast_unrestricted: ArrayLike,
    *,
    nested: bool,
    alternative: str = ALTERNATIVE_B_BETTER,
    maxlags: int | None = None,
) -> ClarkWestResult:
    """Clark-West test for nested scalar squared-error forecasts.

    ``nested`` has no default. It must be passed as ``True``. If the
    comparison is not nested, use :func:`diebold_mariano` instead.
    The default alternative is ``b_better``, i.e. the unrestricted model
    has lower expected adjusted squared error.
    """
    if nested is not True:
        raise ClarkWestScopeError(
            "Clark-West requires an explicit nested=True declaration. "
            "It is not applied merely because one model has more parameters. "
            "Use diebold_mariano for non-nested pairs."
        )
    y = _require_scalar_series(actual, "actual")
    f_r = _require_scalar_series(forecast_restricted, "forecast_restricted")
    f_u = _require_scalar_series(forecast_unrestricted, "forecast_unrestricted")
    if not (y.shape == f_r.shape == f_u.shape):
        raise ValueError("actual and both forecasts must have the same length")

    # Classical CW adjusted differential for squared error.
    error_r = y - f_r
    error_u = y - f_u
    adjusted = error_r**2 - error_u**2 + (f_r - f_u) ** 2
    dm = diebold_mariano(adjusted, alternative=alternative, maxlags=maxlags)
    return ClarkWestResult(
        n_observations=dm.n_observations,
        mean_adjusted_differential=dm.mean_differential,
        statistic=dm.statistic,
        p_value=dm.p_value,
        alternative=dm.alternative,
        nested_declared=True,
        loss="squared_error",
        dm=dm,
    )


def _require_scalar_series(values: ArrayLike, name: str) -> NDArray[np.floating]:
    """Reject matrix or other non-scalar forecast objects."""
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ClarkWestScopeError(
            f"{name} must be a scalar 1-d series. Clark-West in this "
            "harness is defined only for squared forecast error, not for "
            f"QLIKE or matrix losses. Received shape {array.shape}."
        )
    return as_1d_finite(array, name)
