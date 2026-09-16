"""Random-walk realized-covariance baseline.

At origin ``t``, with realized covariance ``S_t`` of shape ``(N, N)``,

    H_{t+1|t} = S_t.

The forecast is an independent copy of the last matrix in the supplied
origin window. Earlier matrices do not enter. There is no jitter,
shrinkage, clipping, diagonal loading, eigenvalue flooring, or
symmetrization. A singular PSD origin matrix remains singular.
"""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray

from covharness.models.base import (
    CovarianceForecast,
    ModelIdentity,
    RealizedCovarianceModel,
    as_realized_covariance_history,
    pack_forecast,
    require_fitted_state,
)

MODEL_NAME = "random_walk_rcov"


class RandomWalkRealizedCovariance(RealizedCovarianceModel):
    """One-day-ahead random walk on realized covariance. ``H_{t+1|t}=S_t``."""

    def __init__(self) -> None:
        self._origin_matrix: NDArray | None = None

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(name=MODEL_NAME, configuration={})

    def fit(self, realized_covariances: ArrayLike) -> RandomWalkRealizedCovariance:
        """Store a copy of the last origin-window matrix. Inputs are not mutated."""
        history = as_realized_covariance_history(realized_covariances)
        # Keep the last matrix.
        self._origin_matrix = history[-1].copy()
        return self

    def forecast(self) -> CovarianceForecast:
        """Return ``H_{t+1|t}=S_t`` as an independent copy."""
        origin = require_fitted_state(self._origin_matrix, MODEL_NAME)
        return pack_forecast(origin, self.identity)
