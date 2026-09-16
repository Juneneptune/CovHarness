"""Auditable rolling-cadence capabilities for covariance models.

Parameter refit, daily observable-state update, and forecast formation are
distinct. The runner dispatches from these typed capabilities. It does not
branch on class names. Models do not inspect protocol block labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from covharness.models.exceptions import InvalidModelConfigurationError


class RollingCadence(str, Enum):
    """How a model advances between scheduled parameter refits."""

    ORIGIN_MAP = "origin_map"
    WINDOW_STATE = "window_state"
    RECURSIVE_STATE = "recursive_state"
    REFIT_HOLD = "refit_hold"


class FitInput(str, Enum):
    """Window consumed by ``fit`` at a scheduled refit origin."""

    REALIZED_COVARIANCE = "realized_covariance"
    REALIZED_COVARIANCE_AND_RQ = "realized_covariance_and_rq"
    DAILY_RETURN = "daily_return"


class UpdateObservable(str, Enum):
    """Observation consumed by a non-refit state advance, if any."""

    NONE = "none"
    REALIZED_COVARIANCE = "realized_covariance"
    REALIZED_COVARIANCE_WINDOW = "realized_covariance_window"
    REALIZED_COVARIANCE_AND_RQ_WINDOW = "realized_covariance_and_rq_window"
    DAILY_RETURN = "daily_return"


@dataclass(frozen=True)
class ModelCapabilities:
    """Typed rolling behavior used by the serial forecast runner."""

    rolling_cadence: RollingCadence
    fit_input: FitInput
    update_observable: UpdateObservable


def capabilities_of(model: object) -> ModelCapabilities:
    """Return the model's class-level rolling capabilities."""
    caps = getattr(type(model), "capabilities", None)
    if not isinstance(caps, ModelCapabilities):
        raise InvalidModelConfigurationError(
            f"{type(model).__name__} has no ModelCapabilities class attribute"
        )
    return caps
