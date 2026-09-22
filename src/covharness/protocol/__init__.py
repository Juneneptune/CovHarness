"""Chronological temporal protocol.

Four chronological regions. HISTORY, VALIDATION, SCREEN, CONFIRM.
Half-open ``[start, end)`` spans on a strictly increasing trading-date index.
CONFIRM is locked unless ``unlock_confirm=True`` is passed.
"""

from covharness.protocol.constants import (
    BLOCK_ROLES,
    DEFAULT_MAX_CONFIGURATIONS,
    DEFAULT_REFIT_CADENCE,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_SEEDS,
    LOCKED,
    CONFIRM_MIN_DAYS,
    SCREEN_MIN_DAYS,
    VALIDATION_TARGET_DAYS,
    SEED_AGGREGATION,
    BlockName,
    OvernightChannel,
    OvernightConvention,
    SeedContract,
    TuningBudget,
)
from covharness.protocol.exceptions import (
    ConfirmLockedError,
    LookaheadError,
    ProtocolAllocationError,
    TuningUnavailableError,
)
from covharness.protocol.preprocess import (
    FitTransformEstimator,
    LocationScaleScaler,
    fit_on_estimation_window,
    require_information_through_origin,
    transform_on_dates,
)
from covharness.protocol.rolling import (
    ForecastStep,
    assert_no_target_leakage,
    build_schedule,
    estimation_window,
)
from covharness.protocol.runner import (
    OriginPayload,
    RollingAction,
    RollingForecastRecord,
    build_origin_payload,
    run_block_forecasts,
    run_rolling_forecasts,
)
from covharness.protocol.binance import (
    BinanceBlock,
    BinanceBlockSchedule,
    BinanceSegmentedProtocol,
    BinanceSelectionError,
    CandidateValidationRecord,
    InvalidBinanceProtocolError,
    build_binance_segmented_protocol,
    core_config_sha256,
    load_core_config,
    select_validation_configuration,
)
from covharness.protocol.splits import (
    BlockAllocation,
    IndexSpan,
    TemporalProtocol,
    allocate_blocks,
    build_temporal_protocol,
)

__all__ = [
    "BinanceBlock",
    "BinanceBlockSchedule",
    "BinanceSegmentedProtocol",
    "BinanceSelectionError",
    "CandidateValidationRecord",
    "InvalidBinanceProtocolError",
    "BLOCK_ROLES",
    "DEFAULT_MAX_CONFIGURATIONS",
    "DEFAULT_REFIT_CADENCE",
    "DEFAULT_ROLLING_WINDOW",
    "DEFAULT_SEEDS",
    "LOCKED",
    "SCREEN_MIN_DAYS",
    "CONFIRM_MIN_DAYS",
    "VALIDATION_TARGET_DAYS",
    "SEED_AGGREGATION",
    "BlockAllocation",
    "BlockName",
    "ConfirmLockedError",
    "FitTransformEstimator",
    "ForecastStep",
    "IndexSpan",
    "LocationScaleScaler",
    "LookaheadError",
    "OvernightChannel",
    "OriginPayload",
    "OvernightConvention",
    "ProtocolAllocationError",
    "RollingAction",
    "RollingForecastRecord",
    "TuningUnavailableError",
    "SeedContract",
    "TemporalProtocol",
    "TuningBudget",
    "allocate_blocks",
    "build_binance_segmented_protocol",
    "core_config_sha256",
    "load_core_config",
    "select_validation_configuration",
    "assert_no_target_leakage",
    "build_origin_payload",
    "build_schedule",
    "build_temporal_protocol",
    "estimation_window",
    "fit_on_estimation_window",
    "require_information_through_origin",
    "run_block_forecasts",
    "run_rolling_forecasts",
    "transform_on_dates",
]
