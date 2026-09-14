"""Errors raised by the leak-proof temporal protocol."""

from __future__ import annotations


class ConfirmLockedError(RuntimeError):
    """CONFIRM observations were requested while the confirmatory block is locked.

    Pass ``unlock_confirm=True`` at the protocol boundary. The later evaluation
    runner will expose the same action as ``--unlock-confirm``. The default is
    locked.
    """


class ProtocolAllocationError(ValueError):
    """The calendar cannot support the committed SCREEN and CONFIRM minima."""


class TuningUnavailableError(RuntimeError):
    """VALIDATION has length zero, so data-driven tuning cannot be performed."""


class LookaheadError(ValueError):
    """Information at or after the target, or after the origin, was requested."""
