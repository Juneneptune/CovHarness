"""Errors raised by covariance forecasting models."""

from __future__ import annotations


class InvalidModelInputError(ValueError):
    """A model input failed a documented numerical or shape contract.

    Non-square history, non-finite entries, asymmetry, a non-PSD realized
    covariance, an empty window, and an unset fitted state are included.
    No silent repair is applied.
    """


class InvalidModelConfigurationError(ValueError):
    """A model configuration is outside the documented parameter domain."""
