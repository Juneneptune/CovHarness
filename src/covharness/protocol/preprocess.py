"""Train-only fit/transform contract.

Scalers and later graph builders may use information through the forecast
origin. They may transform VALIDATION observations. They may not fit on
VALIDATION, SCREEN, CONFIRM, the target date, or the full sample.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from covharness.protocol.exceptions import LookaheadError
from covharness.protocol.rolling import ForecastStep
from covharness.protocol.splits import TemporalProtocol


@runtime_checkable
class FitTransformEstimator(Protocol):
    """Smallest fit/transform surface later models need."""

    def fit(self, values: NDArray[np.floating]) -> FitTransformEstimator:
        """Estimate parameters from allowed training values only."""

    def transform(self, values: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply a previously fitted transform. Must not refit."""


class LocationScaleScaler:
    """Scalar mean/standard-deviation scaler. Fit never sees the full sample."""

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.scale_: float | None = None

    def fit(self, values: NDArray[np.floating]) -> LocationScaleScaler:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError("cannot fit a scaler on an empty window")
        if not np.isfinite(array).all():
            raise ValueError("scaler training values must be finite")
        self.mean_ = float(array.mean())
        scale = float(array.std(ddof=0))
        if scale == 0.0:
            scale = 1.0
        self.scale_ = scale
        return self

    def transform(self, values: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler has not been fit")
        array = np.asarray(values, dtype=float)
        return (array - self.mean_) / self.scale_


def require_information_through_origin(
    used_dates: pd.DatetimeIndex,
    origin: pd.Timestamp,
    target: pd.Timestamp | None = None,
) -> None:
    """Reject dates after the origin, or the target date itself.

    This is the contract later feature and graph construction must call.
    """
    dates = pd.DatetimeIndex(used_dates)
    origin = pd.Timestamp(origin)
    if len(dates) == 0:
        raise LookaheadError("an empty information set is not a valid estimation window")
    if target is not None and pd.Timestamp(target) in dates:
        raise LookaheadError("the target date may not enter fitting or graph construction")
    if (dates > origin).any():
        raise LookaheadError(
            "information after the origin is not permitted in fitting, "
            "scaling, or graph construction"
        )


def fit_on_estimation_window(
    estimator: FitTransformEstimator,
    series: pd.Series,
    protocol: TemporalProtocol,
    origin: pd.Timestamp,
    *,
    unlock_confirm: bool = False,
) -> FitTransformEstimator:
    """Fit ``estimator`` on the rolling window through ``origin`` only.

    Passing a series that also contains SCREEN or CONFIRM values does not
    leak. Those dates are excluded by the window. A naive
    ``estimator.fit(series.to_numpy())`` on the same series would leak.
    """
    window = protocol.estimation_window(origin, unlock_confirm=unlock_confirm)
    step = protocol.forecast_step(origin, unlock_confirm=unlock_confirm)
    require_information_through_origin(window, step.origin, target=step.target)
    aligned = series.reindex(window)
    if aligned.isna().any():
        raise LookaheadError("estimation-window values are missing after reindex")
    return estimator.fit(aligned.to_numpy(dtype=float))


def transform_on_dates(
    estimator: FitTransformEstimator,
    series: pd.Series,
    dates: pd.DatetimeIndex,
) -> NDArray[np.floating]:
    """Transform values at ``dates`` without refitting."""
    aligned = series.reindex(dates)
    if aligned.isna().any():
        raise LookaheadError("transform dates are missing after reindex")
    return estimator.transform(aligned.to_numpy(dtype=float))


def assert_step_allows_preprocess(step: ForecastStep, calendar: pd.DatetimeIndex) -> None:
    """Check that a scheduled step is safe for scaler or graph fitting."""
    require_information_through_origin(
        step.estimation_dates(calendar), step.origin, target=step.target
    )
