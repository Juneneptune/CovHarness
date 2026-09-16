"""Errors raised by origin-day state construction."""

from __future__ import annotations


class InvalidOriginStateError(ValueError):
    """An origin-day return panel failed a documented state-construction contract.

    Nonpositive aggregate realized quarticity, insufficient intervals, zero
    realized variance or bipower, and nonfinite or negative quadpower are
    included. No epsilon floor or silent replacement is applied.
    """
