"""Origin-day measurement and stress state from synchronized intraday returns.

The constructions consume already-synchronized return matrices. They do not
touch raw TAQ or alter the realized-covariance pipeline.
"""

from covharness.features.exceptions import InvalidOriginStateError
from covharness.features.origin_state import (
    JUMP_ALPHA,
    JUMP_CRITICAL_VALUE,
    JUMP_MIN_INTERVALS,
    MEASUREMENT_STATE_SPECIFICATION,
    QUARTICITY_MIN_INTERVALS,
    BNSJumpResult,
    QuarticityResult,
    bns_market_jump,
    measurement_stress_instruments,
    realized_quarticity_aggregate,
)

__all__ = [
    "BNSJumpResult",
    "InvalidOriginStateError",
    "JUMP_ALPHA",
    "JUMP_CRITICAL_VALUE",
    "JUMP_MIN_INTERVALS",
    "MEASUREMENT_STATE_SPECIFICATION",
    "QUARTICITY_MIN_INTERVALS",
    "QuarticityResult",
    "bns_market_jump",
    "measurement_stress_instruments",
    "realized_quarticity_aggregate",
]
