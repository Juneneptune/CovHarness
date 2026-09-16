"""Alignment, point-loss, panel, and differential tests for the evaluation adapter."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from covharness.evaluation import (
    LOSS_REDUCED_QLIKE,
    LOSS_SQUARED_FROBENIUS,
    MISMATCH_INTERSECTION,
    EvaluationAlignmentError,
    align_forecast_records,
    build_loss_panel,
    loss_differential_from_panel,
    score_forecast_records,
    summarize_loss_panel,
    target_covariance_panel,
)
from covharness.evaluation import score as score_module
from covharness.inference import diebold_mariano_from_losses, loss_differential
from covharness.losses import (
    ForecastNotPositiveDefiniteError,
    reduced_qlike_loss,
    squared_frobenius_loss,
)
from covharness.protocol.runner import RollingAction, RollingForecastRecord

SEED = 20260916
SPD_S = np.array([[2.0, 0.4], [0.4, 1.0]])
SPD_H = np.array([[1.5, 0.2], [0.2, 1.2]])
SPD_B = np.array([[1.8, 0.1], [0.1, 1.3]])
NOT_PD_H = np.array([[1.0, 2.0], [2.0, 1.0]])


def _calendar(n_times: int = 6) -> pd.DatetimeIndex:
    return pd.bdate_range("1990-01-01", periods=n_times)


def _cube(*matrices: np.ndarray) -> np.ndarray:
    return np.stack(matrices, axis=0)


def _record(
    name: str,
    origin: pd.Timestamp,
    target: pd.Timestamp,
    matrix: np.ndarray,
    *,
    refit: bool = True,
) -> RollingForecastRecord:
    return RollingForecastRecord(
        model_name=name,
        origin=origin,
        target=target,
        refit=refit,
        action=RollingAction.FIT if refit else RollingAction.UPDATE,
        covariance=np.array(matrix, dtype=float, copy=True),
    )


def test_origin_maps_to_next_calendar_target() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    origin = calendar[2]
    target = calendar[3]
    record = _record("random_walk_rcov", origin, target, SPD_H)
    pair = align_forecast_records([record], targets)[0]
    assert pair.origin == origin
    assert pair.target == target
    assert calendar.get_loc(pair.target) == calendar.get_loc(pair.origin) + 1


def test_evaluation_target_is_s_t_plus_one_not_s_t() -> None:
    calendar = _calendar()
    cube = _cube(SPD_S, SPD_H, SPD_B, SPD_S, SPD_H, SPD_B)
    original = cube.copy()
    targets = target_covariance_panel(calendar, cube)
    origin = calendar[1]
    target = calendar[2]
    pair = align_forecast_records(
        [_record("har_drd", origin, target, SPD_S)], targets
    )[0]
    np.testing.assert_array_equal(pair.proxy, original[2])
    assert not np.allclose(pair.proxy, original[1])
    np.testing.assert_array_equal(cube, original)


def test_calendar_key_alignment_reorders_shuffled_targets() -> None:
    calendar = _calendar(4)
    cube = _cube(SPD_S, SPD_H, SPD_B, SPD_S)
    order = np.array([2, 0, 3, 1])
    shuffled_calendar = calendar[order]
    shuffled_cube = cube[order]
    targets = target_covariance_panel(shuffled_calendar, shuffled_cube)
    origin = calendar[1]
    target = calendar[2]
    pair = align_forecast_records(
        [_record("ridge_drd", origin, target, SPD_S)], targets
    )[0]
    np.testing.assert_allclose(pair.proxy, cube[2])
    np.testing.assert_allclose(pair.proxy, SPD_B)


def test_missing_target_fails() -> None:
    calendar = _calendar(4)
    targets = target_covariance_panel(calendar, _cube(*([SPD_S] * 4)))
    missing = calendar[-1] + pd.offsets.BDay(5)
    record = _record("har_drd", calendar[1], missing, SPD_H)
    with pytest.raises(EvaluationAlignmentError, match="not on the target calendar"):
        align_forecast_records([record], targets)


def test_duplicate_target_fails() -> None:
    calendar = _calendar()
    targets = target_covariance_panel(calendar, _cube(*([SPD_S] * len(calendar))))
    first = _record("har_drd", calendar[1], calendar[2], SPD_H)
    second = _record("har_drd", calendar[1], calendar[2], SPD_B)
    with pytest.raises(EvaluationAlignmentError, match="duplicate target"):
        align_forecast_records([first, second], targets)


def test_dimension_mismatch_fails() -> None:
    calendar = _calendar()
    targets = target_covariance_panel(calendar, _cube(*([SPD_S] * len(calendar))))
    wide = np.eye(3)
    record = _record("har_drd", calendar[1], calendar[2], wide)
    with pytest.raises(EvaluationAlignmentError, match="dimensions must match"):
        align_forecast_records([record], targets)


def test_inputs_not_mutated() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    original_cube = cube.copy()
    record = _record("har_drd", calendar[1], calendar[2], SPD_H)
    original_h = record.covariance.copy()
    targets = target_covariance_panel(calendar, cube)
    pair = align_forecast_records([record], targets)[0]
    pair.forecast[0, 0] = -99.0
    pair.proxy[0, 0] = -99.0
    np.testing.assert_array_equal(cube, original_cube)
    np.testing.assert_array_equal(record.covariance, original_h)
    np.testing.assert_array_equal(targets.realized_covariances, original_cube)


def test_reduced_qlike_matches_existing_implementation() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    record = _record("ridge_drd", calendar[2], calendar[3], SPD_H)
    scored = score_forecast_records(
        [record], targets, losses=(LOSS_REDUCED_QLIKE,)
    )
    expected = reduced_qlike_loss(SPD_S, SPD_H)
    assert scored[0].loss_value == pytest.approx(expected)
    np.testing.assert_allclose(reduced_qlike_loss(np.eye(2), np.eye(2)), 2.0)


def test_squared_frobenius_matches_hand_calculation() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    record = _record("ridge_drd", calendar[2], calendar[3], SPD_H)
    scored = score_forecast_records(
        [record], targets, losses=(LOSS_SQUARED_FROBENIUS,)
    )
    residual = SPD_S - SPD_H
    expected = float(np.sum(residual * residual))
    assert scored[0].loss_value == pytest.approx(expected)
    assert scored[0].loss_value == pytest.approx(squared_frobenius_loss(SPD_S, SPD_H))


def test_equal_forecast_and_target_have_documented_values() -> None:
    calendar = _calendar()
    cube = _cube(*([np.eye(2)] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    record = _record("random_walk_rcov", calendar[1], calendar[2], np.eye(2))
    scored = score_forecast_records([record], targets)
    by_name = {row.loss_name: row.loss_value for row in scored}
    assert by_name[LOSS_SQUARED_FROBENIUS] == pytest.approx(0.0)
    assert by_name[LOSS_REDUCED_QLIKE] == pytest.approx(2.0)


def test_evaluation_does_not_repair_non_pd_forecast() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    record = _record("har_drd", calendar[1], calendar[2], NOT_PD_H)
    with pytest.raises(ForecastNotPositiveDefiniteError):
        score_forecast_records([record], targets, losses=(LOSS_REDUCED_QLIKE,))
    source = inspect.getsource(score_module.score_aligned_pairs)
    lowered = source.lower()
    for token in ("clip", "nearest", "jitter", "higham", "cov_nearest"):
        assert token not in lowered


def test_repeated_evaluation_is_deterministic() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    record = _record("ridge_drd", calendar[3], calendar[4], SPD_H)
    first = score_forecast_records([record], targets)
    second = score_forecast_records([record], targets)
    assert [row.loss_value for row in first] == [row.loss_value for row in second]


def test_loss_records_follow_chronological_targets() -> None:
    calendar = _calendar()
    cube = _cube(SPD_S, SPD_H, SPD_B, SPD_S, SPD_H, SPD_B)
    targets = target_covariance_panel(calendar, cube)
    later = _record("har_drd", calendar[3], calendar[4], SPD_S, refit=False)
    earlier = _record("har_drd", calendar[1], calendar[2], SPD_H)
    scored = score_forecast_records([later, earlier], targets)
    qlike = [row for row in scored if row.loss_name == LOSS_REDUCED_QLIKE]
    panel = build_loss_panel(qlike, loss_name=LOSS_REDUCED_QLIKE)
    np.testing.assert_array_equal(panel.targets, pd.DatetimeIndex([calendar[2], calendar[4]]))


def test_panel_has_one_entry_per_model_date_loss() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    records = [
        _record("a", calendar[1], calendar[2], SPD_H),
        _record("a", calendar[2], calendar[3], SPD_B, refit=False),
        _record("b", calendar[1], calendar[2], SPD_B),
        _record("b", calendar[2], calendar[3], SPD_H, refit=False),
    ]
    scored = score_forecast_records(records, targets)
    assert len(scored) == 8
    panel = build_loss_panel(
        scored,
        loss_name=LOSS_SQUARED_FROBENIUS,
        model_order=("a", "b"),
    )
    assert panel.values.shape == (2, 2)
    assert panel.model_names == ("a", "b")


def test_identical_target_support_is_required_by_default() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    records = [
        _record("a", calendar[1], calendar[2], SPD_H),
        _record("a", calendar[2], calendar[3], SPD_B, refit=False),
        _record("b", calendar[1], calendar[2], SPD_B),
    ]
    scored = score_forecast_records(records, targets)
    with pytest.raises(EvaluationAlignmentError, match="identical target dates"):
        build_loss_panel(scored, loss_name=LOSS_REDUCED_QLIKE)
    intersection = build_loss_panel(
        scored,
        loss_name=LOSS_REDUCED_QLIKE,
        date_support=MISMATCH_INTERSECTION,
        model_order=("a", "b"),
    )
    assert len(intersection.targets) == 1
    assert intersection.targets[0] == calendar[2]


def test_differential_matches_hand_calculation_and_sign_convention() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    records = [
        _record("a", calendar[1], calendar[2], SPD_H),
        _record("a", calendar[2], calendar[3], SPD_B, refit=False),
        _record("b", calendar[1], calendar[2], SPD_B),
        _record("b", calendar[2], calendar[3], SPD_H, refit=False),
    ]
    scored = score_forecast_records(
        records, targets, losses=(LOSS_SQUARED_FROBENIUS,)
    )
    panel = build_loss_panel(scored, loss_name=LOSS_SQUARED_FROBENIUS, model_order=("a", "b"))
    loss_a = panel.column("a")
    loss_b = panel.column("b")
    expected = loss_a - loss_b
    got = loss_differential_from_panel(panel, "a", "b")
    np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(got, loss_differential(loss_a, loss_b))
    swapped = loss_differential_from_panel(panel, "b", "a")
    np.testing.assert_allclose(swapped, -got)
    source = inspect.getsource(loss_differential_from_panel)
    assert "d_t = L_{A,t} - L_{B,t}" in source
    result = diebold_mariano_from_losses(loss_a, loss_b)
    assert result.n_observations == 2
    assert np.isfinite(result.statistic)


def test_panel_does_not_insert_nan_holes() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    records = [
        _record("a", calendar[1], calendar[2], SPD_H),
        _record("b", calendar[1], calendar[2], SPD_B),
    ]
    scored = score_forecast_records(records, targets)
    panel = build_loss_panel(scored, loss_name=LOSS_REDUCED_QLIKE)
    assert np.isfinite(panel.values).all()


def test_summary_is_descriptive_and_roster_ordered() -> None:
    calendar = _calendar()
    cube = _cube(*([SPD_S] * len(calendar)))
    targets = target_covariance_panel(calendar, cube)
    records = [
        _record("b", calendar[1], calendar[2], SPD_B),
        _record("a", calendar[1], calendar[2], SPD_H),
    ]
    scored = score_forecast_records(
        records, targets, losses=(LOSS_SQUARED_FROBENIUS,)
    )
    panel = build_loss_panel(
        scored, loss_name=LOSS_SQUARED_FROBENIUS, model_order=("a", "b")
    )
    summaries = summarize_loss_panel(panel)
    assert [row.model_name for row in summaries] == ["a", "b"]
    assert all(np.isfinite(row.mean_loss) for row in summaries)
    joined = " ".join(row.model_name for row in summaries)
    assert "winner" not in joined
    assert "best" not in inspect.getsource(summarize_loss_panel).lower()
