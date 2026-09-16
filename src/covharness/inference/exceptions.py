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


class RankDeficientGWInstrumentsError(ValueError):
    """GW instruments do not have full column rank.

    Duplicate or collinear test-function directions are rejected. Columns are
    not dropped and no ridge is added.
    """


class DegenerateGWCovarianceError(ValueError):
    """The GW outer-product covariance is singular or non-finite.

    All-zero moments are included. A nonzero constant differential with
    full-rank instruments is not this error.
    """


class RankDeficientMZDesignError(ValueError):
    """The pooled Mincer-Zarnowitz design matrix is rank deficient."""


class DegenerateFluctuationVarianceError(ValueError):
    """Giacomini-Rossi global long-run variance is zero, negative, or non-finite.

    The uncentered estimator is not repaired by jitter or clipping.
    """
