"""Squared Frobenius loss and the labeled unsquared (non-robust) counterpart."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from covharness.losses.contracts import (
    as_square_finite,
    require_matching_square,
    require_symmetric,
)


def squared_frobenius_loss(proxy: ArrayLike, forecast: ArrayLike) -> float:
    """Squared Frobenius distance ``||S - H||_F^2``.

    ``L_F(S, H) = trace((S-H).T @ (S-H)) = sum_{ij} (S_ij - H_ij)^2``.

    This is the matrix analogue of MSE and is the adopted Frobenius ranking
    loss. PSD or PD is not required to compute it. There is no scaling or
    annualization. Inputs are not mutated.
    """
    # Copy and check square, finite, matching shape, and documented symmetry.
    proxy_s = as_square_finite(proxy, "proxy S")
    forecast_h = as_square_finite(forecast, "forecast H")
    require_matching_square(proxy_s, forecast_h)
    require_symmetric(proxy_s, "proxy S")
    require_symmetric(forecast_h, "forecast H")

    # Elementwise squared error on the full matrix, including both triangles.
    residual = proxy_s - forecast_h
    return float(np.sum(residual * residual))


def unsquared_frobenius_loss(proxy: ArrayLike, forecast: ArrayLike) -> float:
    """Ordinary Frobenius norm ``||S - H||_F``. Not a ranking loss.

    This is the square root of ``squared_frobenius_loss``. The square root is
    a nonlinear transform of the proxy residual. It is retained only as a
    labeled non-robust contrast and is not used to rank models.
    """
    return float(np.sqrt(squared_frobenius_loss(proxy, forecast)))
