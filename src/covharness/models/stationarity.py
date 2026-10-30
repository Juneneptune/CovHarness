"""Shared stationarity map for likelihood covariance models.

Carry-forward is the primary failure policy. The interior map is a labeled
robustness variant. Forecast matrices are never repaired here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

STATIONARITY_EPSILON = 1e-6
NEAR_BOUNDARY_SUM = 0.999
POLICY_REJECT = "reject"
POLICY_CARRY_FORWARD = "carry_forward"
POLICY_INTERIOR_MAP = "interior_map"
POLICIES = (POLICY_REJECT, POLICY_CARRY_FORWARD, POLICY_INTERIOR_MAP)


@dataclass(frozen=True)
class PersistenceMap:
    """Slack and two nonnegative weights with ``a + b <= 1 - epsilon``."""

    slack: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return float(self.alpha + self.beta)


def softmax_persistence(logits: NDArray[np.floating]) -> PersistenceMap:
    """Map three unconstrained logits to a stationarity triple.

    The weights are a softmax, then scaled by ``1 - epsilon``. The
    resulting ``alpha + beta`` is strictly below 1.
    """
    raw = np.asarray(logits, dtype=float)
    if raw.shape != (3,) or not np.isfinite(raw).all():
        raise ValueError("stationarity logits must be three finite numbers")
    # Shift before the exponential so a large logit does not overflow.
    shifted = raw - np.max(raw)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    scale = 1.0 - STATIONARITY_EPSILON
    mapped = scale * weights
    return PersistenceMap(
        slack=float(mapped[0]),
        alpha=float(mapped[1]),
        beta=float(mapped[2]),
    )


def logits_from_persistence(alpha: float, beta: float) -> NDArray[np.floating]:
    """Invert the scaled softmax at an interior ``(alpha, beta)`` pair."""
    if alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0 - STATIONARITY_EPSILON:
        raise ValueError("persistence pair is outside the interior map")
    slack = 1.0 - STATIONARITY_EPSILON - alpha - beta
    if slack <= 0.0:
        raise ValueError("persistence pair leaves no positive slack")
    scale = 1.0 - STATIONARITY_EPSILON
    return np.log(np.array([slack, alpha, beta], dtype=float) / scale)


def is_near_boundary(alpha: float, beta: float) -> bool:
    """True when persistence exceeds the recorded near-boundary thresholds."""
    total = float(alpha + beta)
    cap = 1.0 - STATIONARITY_EPSILON
    return bool(total > cap or total > NEAR_BOUNDARY_SUM)


def is_strictly_stationary(alpha: float, beta: float) -> bool:
    """True for the project rule ``alpha >= 0``, ``beta >= 0``, ``alpha + beta < 1``."""
    return bool(
        np.isfinite(alpha)
        and np.isfinite(beta)
        and alpha >= 0.0
        and beta >= 0.0
        and alpha + beta < 1.0
    )
