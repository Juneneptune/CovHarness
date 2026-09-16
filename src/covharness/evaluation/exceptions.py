"""Errors raised while aligning forecasts to evaluation targets."""

from __future__ import annotations


class EvaluationAlignmentError(ValueError):
    """Forecast records and target covariances could not be aligned by date.

    Missing targets, duplicate targets, mismatched model date support, and
    dimension disagreement are included. Arrays are not repaired.
    """
