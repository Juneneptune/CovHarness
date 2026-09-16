"""EWMA realized-covariance baseline.

For an origin-window sequence ``S_0, ..., S_{T-1}`` of shape ``(T, N, N)``
and decay ``lambda`` in ``(0, 1)``,

    H_0 = S_0
    H_j = lambda H_{j-1} + (1 - lambda) S_j,    j = 1, ..., T-1.

The one-day-ahead forecast is ``H_{T|T-1} = H_{T-1}``. This is EWMA on
realized covariance matrices, not on daily-return outer products.

``lambda`` is an explicit configuration. The conventional RiskMetrics
reference 0.94 may be passed by a caller. It is not a tuned project
choice. The 20-point VALIDATION grid is not frozen here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from covharness.models.base import (
    CovarianceForecast,
    ModelIdentity,
    RealizedCovarianceModel,
    as_realized_covariance_history,
    as_realized_covariance_matrix,
    pack_forecast,
    require_fitted_state,
)
from covharness.models.capabilities import (
    FitInput,
    ModelCapabilities,
    RollingCadence,
    UpdateObservable,
)
from covharness.models.exceptions import InvalidModelConfigurationError

MODEL_NAME = "ewma_rcov"


class EWMARealizedCovariance(RealizedCovarianceModel):
    """One-day-ahead EWMA of a realized-covariance window."""

    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.RECURSIVE_STATE,
        fit_input=FitInput.REALIZED_COVARIANCE,
        update_observable=UpdateObservable.REALIZED_COVARIANCE,
    )

    def __init__(self, *, decay: float) -> None:
        self._decay = _require_decay(decay)
        self._state: NDArray | None = None

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            name=MODEL_NAME,
            configuration={"decay": float(self._decay)},
        )

    def fit(self, realized_covariances: ArrayLike) -> EWMARealizedCovariance:
        """Run the EWMA recursion through the origin. Inputs are not mutated."""
        history = as_realized_covariance_history(realized_covariances)
        decay = self._decay
        one_minus = 1.0 - decay
        # Start from the first matrix.
        state = history[0].copy()
        # Update with later matrices.
        for time_index in range(1, history.shape[0]):
            state = decay * state + one_minus * history[time_index]
        self._state = state
        return self

    def update(self, new_realized_covariance: ArrayLike) -> EWMARealizedCovariance:
        """Advance the frozen-lambda recursion by one realized covariance."""
        state = require_fitted_state(self._state, MODEL_NAME)
        observed = as_realized_covariance_matrix(
            new_realized_covariance, n_assets=state.shape[0]
        )
        decay = self._decay
        # Recurse one day. Lambda is never changed here.
        self._state = decay * state + (1.0 - decay) * observed
        return self

    def forecast(self) -> CovarianceForecast:
        """Return the terminal EWMA state as ``H_{t+1|t}``."""
        state = require_fitted_state(self._state, MODEL_NAME)
        return pack_forecast(state, self.identity)


def _require_decay(decay: float) -> float:
    """Reject a decay outside the open interval (0, 1)."""
    value = float(decay)
    if not np.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise InvalidModelConfigurationError(
            f"EWMA decay must satisfy 0 < lambda < 1; got {decay!r}"
        )
    return value
