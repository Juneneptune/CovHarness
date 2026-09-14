"""Demo protocol quantities, block roles, and frozen conventions.

These are target and minimum protocol quantities. VALIDATION 250 is a target
and may be shortened. SCREEN 500 and CONFIRM 500 are committed minima.
They are not numbers that individual experiments may silently violate to make
a short sample fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Default locked state. CONFIRM is never open unless an explicit unlock is passed.
LOCKED = True

# Rolling estimation window length in trading days. Part of the forecasting method.
DEFAULT_ROLLING_WINDOW = 250

# Monthly refit cadence in trading days. Identical for every later model family.
DEFAULT_REFIT_CADENCE = 21

# Preferred VALIDATION length. This is a target, not a hard minimum.
# The allocator may shorten VALIDATION, including to zero.
VALIDATION_TARGET_DAYS = 250
# Committed evaluation-block minima. These are never silently shortened.
SCREEN_MIN_DAYS = 500
CONFIRM_MIN_DAYS = 500

# Pre-specified stochastic-model seeds. Report all. Do not select a best seed.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# Equal configuration budget per model family. No search engine is implied.
DEFAULT_MAX_CONFIGURATIONS = 20

# Frozen seed-ensemble rule. The confirmatory object is the mean across seeds.
SEED_AGGREGATION = "mean_ensemble"


class BlockName(str, Enum):
    """Named chronological regions of the protocol calendar."""

    HISTORY = "HISTORY"
    VALIDATION = "VALIDATION"
    SCREEN = "SCREEN"
    CONFIRM = "CONFIRM"


# What each evaluation block is allowed to do. Encoded for audit, not as a tuner.
BLOCK_ROLES = {
    BlockName.VALIDATION: (
        "hyperparameter tuning",
        "early stopping",
        "architecture and configuration selection within the preregistered search space",
        "training-objective selection if preregistered",
    ),
    BlockName.SCREEN: (
        "comparison of frozen candidate configurations",
        "MCS / SPA screening in Block 3",
        "finalist selection under the frozen rule",
    ),
    BlockName.CONFIRM: (
        "final locked confirmatory comparisons only",
        "no tuning",
        "no architecture changes",
        "no proxy or loss changes",
        "no seed selection",
    ),
}


class OvernightChannel(str, Enum):
    """Statistical versus economic overnight convention, frozen before results."""

    STATISTICAL_OPEN_TO_CLOSE = "statistical_open_to_close"
    ECONOMIC_INCLUDE_OVERNIGHT = "economic_include_overnight"
    ECONOMIC_OPEN_TO_CLOSE_ROBUSTNESS = "economic_open_to_close_robustness"


@dataclass(frozen=True)
class OvernightConvention:
    """Measurement and economic overnight roles. GMV is not implemented here."""

    statistical_proxy: str = OvernightChannel.STATISTICAL_OPEN_TO_CLOSE.value
    economic_primary: str = OvernightChannel.ECONOMIC_INCLUDE_OVERNIGHT.value
    economic_robustness: str = OvernightChannel.ECONOMIC_OPEN_TO_CLOSE_ROBUSTNESS.value


@dataclass(frozen=True)
class SeedContract:
    """Fixed seed list and frozen aggregation. Best-seed selection is forbidden."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    aggregation: str = SEED_AGGREGATION
    report_all: bool = True
    allow_best_seed: bool = False

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("the seed list must be non-empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate seeds are not permitted")
        if self.allow_best_seed:
            raise ValueError("best-seed selection is not permitted")
        if not self.report_all:
            raise ValueError("all seeds must be reported")
        if self.aggregation != SEED_AGGREGATION:
            raise ValueError(
                f"aggregation must remain {SEED_AGGREGATION!r} until a "
                "preregistered change is recorded"
            )


@dataclass(frozen=True)
class TuningBudget:
    """Auditable configuration cap per model family. Not a tuning engine."""

    max_configurations: int = DEFAULT_MAX_CONFIGURATIONS
    family: str | None = None

    def __post_init__(self) -> None:
        if self.max_configurations < 1:
            raise ValueError("max_configurations must be at least 1")
