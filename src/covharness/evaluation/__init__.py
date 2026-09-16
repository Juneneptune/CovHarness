"""Connect rolling forecast records to existing covariance-space losses.

Point losses remain the Block 2A implementations. This package only aligns
``H_{t+1|t}`` to ``S_{t+1}`` and assembles descriptive panels.
"""

from covharness.evaluation.exceptions import EvaluationAlignmentError
from covharness.evaluation.score import (
    DEFAULT_LOSSES,
    LOSS_REDUCED_QLIKE,
    LOSS_SQUARED_FROBENIUS,
    MISMATCH_ERROR,
    MISMATCH_INTERSECTION,
    AlignedForecastTarget,
    EvaluationLoss,
    LossPanel,
    LossRecord,
    ModelLossSummary,
    TargetCovariancePanel,
    align_forecast_records,
    build_loss_panel,
    loss_differential_from_panel,
    score_aligned_pairs,
    score_forecast_records,
    summarize_loss_panel,
    target_covariance_panel,
)

__all__ = [
    "DEFAULT_LOSSES",
    "LOSS_REDUCED_QLIKE",
    "LOSS_SQUARED_FROBENIUS",
    "MISMATCH_ERROR",
    "MISMATCH_INTERSECTION",
    "AlignedForecastTarget",
    "EvaluationAlignmentError",
    "EvaluationLoss",
    "LossPanel",
    "LossRecord",
    "ModelLossSummary",
    "TargetCovariancePanel",
    "align_forecast_records",
    "build_loss_panel",
    "loss_differential_from_panel",
    "score_aligned_pairs",
    "score_forecast_records",
    "summarize_loss_panel",
    "target_covariance_panel",
]
