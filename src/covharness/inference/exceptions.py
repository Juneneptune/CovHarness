"""Errors raised by pairwise predictive inference."""

from __future__ import annotations


class DegenerateLossDifferentialError(ValueError):
    """HAC/DM is undefined for this loss-differential series.

    A constant series is rejected before demeaning. A non-finite or zero
    long-run variance is also rejected. No jitter is added.
    """


class ClarkWestScopeError(ValueError):
    """Clark-West was requested outside scalar squared-error nested comparisons."""


class DegenerateMCSDifferentialError(ValueError):
    """MCS cannot studentize a pairwise differential that is constant and nonzero.

    Identical loss columns are a tie and are not an error. A nonzero constant
    pair has zero variance, so the t-ratio is undefined. No jitter is added.
    """
