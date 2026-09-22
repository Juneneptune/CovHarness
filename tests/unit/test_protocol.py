"""Chronological temporal protocol tests on synthetic calendars only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covharness.protocol import (
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
    ConfirmLockedError,
    LocationScaleScaler,
    LookaheadError,
    OvernightChannel,
    ProtocolAllocationError,
    TuningUnavailableError,
    SeedContract,
    TemporalProtocol,
    TuningBudget,
    allocate_blocks,
    build_temporal_protocol,
    fit_on_estimation_window,
    require_information_through_origin,
    transform_on_dates,
)


def trading_calendar(n_days: int) -> pd.DatetimeIndex:
    """Business-day calendar with no market-data content."""
    return pd.bdate_range("1990-01-01", periods=n_days)


PREFERRED_N = (
    DEFAULT_ROLLING_WINDOW + VALIDATION_TARGET_DAYS + SCREEN_MIN_DAYS + CONFIRM_MIN_DAYS
)


def preferred_protocol() -> TemporalProtocol:
    return build_temporal_protocol(trading_calendar(PREFERRED_N))


def test_chronological_ordering() -> None:
    protocol = preferred_protocol()
    history = protocol.history_dates()
    validation = protocol.validation_targets()
    screen = protocol.screen_targets()
    confirm = protocol.confirm_targets(unlock_confirm=True)
    assert history.is_monotonic_increasing
    assert validation.is_monotonic_increasing
    assert screen.is_monotonic_increasing
    assert confirm.is_monotonic_increasing
    assert history[-1] < validation[0]
    assert validation[-1] < screen[0]
    assert screen[-1] < confirm[0]


def test_evaluation_blocks_do_not_overlap() -> None:
    protocol = preferred_protocol()
    validation = protocol.validation_targets()
    screen = protocol.screen_targets()
    confirm = protocol.confirm_targets(unlock_confirm=True)
    assert validation.intersection(screen).empty
    assert validation.intersection(confirm).empty
    assert screen.intersection(confirm).empty
    history = protocol.history_dates()
    assert history.intersection(validation).empty
    assert history.intersection(screen).empty
    assert history.intersection(confirm).empty


def test_target_block_size_validation_preferred() -> None:
    protocol = preferred_protocol()
    assert protocol.m == 250
    assert protocol.n_validation == VALIDATION_TARGET_DAYS
    assert protocol.n_screen == SCREEN_MIN_DAYS
    assert protocol.n_confirm == CONFIRM_MIN_DAYS
    assert protocol.validation_shortened is False
    assert protocol.n_evaluation_targets == 1250


def test_shortens_validation_before_confirm() -> None:
    n_dates = DEFAULT_ROLLING_WINDOW + 80 + SCREEN_MIN_DAYS + CONFIRM_MIN_DAYS
    protocol = build_temporal_protocol(trading_calendar(n_dates))
    assert protocol.n_validation == 80
    assert protocol.n_screen == SCREEN_MIN_DAYS
    assert protocol.n_confirm == CONFIRM_MIN_DAYS
    assert protocol.validation_shortened is True
    assert protocol.n_validation < protocol.validation_target_days
    assert protocol.data_driven_tuning_available is True
    protocol.require_data_driven_tuning()


def test_surplus_goes_to_validation_not_screen_or_confirm() -> None:
    n_dates = PREFERRED_N + 40
    protocol = build_temporal_protocol(trading_calendar(n_dates))
    assert protocol.n_validation == VALIDATION_TARGET_DAYS + 40
    assert protocol.n_screen == SCREEN_MIN_DAYS
    assert protocol.n_confirm == CONFIRM_MIN_DAYS


def test_zero_validation_when_only_screen_and_confirm_fit() -> None:
    n_dates = DEFAULT_ROLLING_WINDOW + SCREEN_MIN_DAYS + CONFIRM_MIN_DAYS
    protocol = build_temporal_protocol(trading_calendar(n_dates))
    assert protocol.n_validation == 0
    assert protocol.validation_targets().empty
    assert protocol.n_screen == SCREEN_MIN_DAYS
    assert protocol.n_confirm == CONFIRM_MIN_DAYS
    assert protocol.history_dates()[-1] < protocol.screen_targets()[0]
    assert protocol.validation_shortened is True
    assert protocol.data_driven_tuning_available is False
    with pytest.raises(TuningUnavailableError, match="unavailable"):
        protocol.require_data_driven_tuning()


def test_fails_when_confirm_requirement_cannot_be_met() -> None:
    n_dates = DEFAULT_ROLLING_WINDOW + SCREEN_MIN_DAYS + CONFIRM_MIN_DAYS - 1
    with pytest.raises(ProtocolAllocationError, match="infeasible"):
        build_temporal_protocol(trading_calendar(n_dates))


def test_does_not_silently_shrink_screen() -> None:
    allocation = allocate_blocks(PREFERRED_N)
    assert len(allocation.screen) == SCREEN_MIN_DAYS
    assert len(allocation.confirm) == CONFIRM_MIN_DAYS


def test_rejects_shuffled_dates() -> None:
    dates = trading_calendar(PREFERRED_N)
    shuffled = dates[::-1]
    with pytest.raises(ValueError, match="shuffling"):
        build_temporal_protocol(shuffled)


def test_rolling_window_boundaries_m_250() -> None:
    protocol = preferred_protocol()
    calendar = protocol.full_calendar(unlock_confirm=True)
    first_target = protocol.validation_targets()[0]
    origin = protocol.origin_for_target(first_target)
    window = protocol.estimation_window(origin)
    assert len(window) == 250
    assert window.equals(calendar[:250])
    assert window[-1] == origin
    assert first_target not in window
    assert origin == calendar[249]
    assert first_target == calendar[250]


def test_target_is_next_day_without_leakage() -> None:
    protocol = preferred_protocol()
    calendar = protocol.full_calendar(unlock_confirm=True)
    for block in (BlockName.VALIDATION, BlockName.SCREEN):
        schedule = protocol.forecast_schedule(block)
        for step in schedule:
            assert step.target_index == step.origin_index + 1
            assert step.window_end_index == step.target_index
            assert len(step.estimation_dates(calendar)) == 250
            assert step.target not in step.estimation_dates(calendar)
            assert step.estimation_dates(calendar)[-1] == step.origin
            assert (step.estimation_dates(calendar) > step.origin).sum() == 0


def test_monthly_refit_every_21_forecast_origins() -> None:
    protocol = preferred_protocol()
    schedule = protocol.forecast_schedule(BlockName.VALIDATION)
    assert protocol.refit_cadence == DEFAULT_REFIT_CADENCE == 21
    refit_flags = [step.refit for step in schedule]
    assert refit_flags[0] is True
    for position, flag in enumerate(refit_flags):
        assert flag is (position % 21 == 0)
    assert sum(refit_flags) == 12


def test_screen_schedule_uses_same_window_and_cadence() -> None:
    protocol = preferred_protocol()
    screen = protocol.forecast_schedule(BlockName.SCREEN)
    assert screen[0].m == 250
    assert screen[0].refit is True
    assert screen[21].refit is True
    assert screen[1].refit is False


def test_train_only_scaler_fitting() -> None:
    protocol = preferred_protocol()
    calendar = protocol.full_calendar(unlock_confirm=True)
    values = pd.Series(np.linspace(0.0, 1.0, len(calendar)), index=calendar)
    first_origin = protocol.origin_for_target(protocol.validation_targets()[0])
    scaler = LocationScaleScaler()
    fit_on_estimation_window(scaler, values, protocol, first_origin)
    window = protocol.estimation_window(first_origin)
    expected_mean = float(values.loc[window].mean())
    assert scaler.mean_ == pytest.approx(expected_mean)
    transformed = transform_on_dates(scaler, values, protocol.validation_targets()[:5])
    assert transformed.shape == (5,)


def test_no_global_scaling_leakage() -> None:
    protocol = preferred_protocol()
    calendar = protocol.full_calendar(unlock_confirm=True)
    values = np.ones(len(calendar), dtype=float)
    confirm = protocol.confirm_targets(unlock_confirm=True)
    values[calendar.get_indexer(confirm)] = 1_000_000.0
    series = pd.Series(values, index=calendar)
    naive = LocationScaleScaler().fit(series.to_numpy())
    first_origin = protocol.origin_for_target(protocol.validation_targets()[0])
    protocol_scaler = LocationScaleScaler()
    fit_on_estimation_window(protocol_scaler, series, protocol, first_origin)
    assert naive.mean_ == pytest.approx(series.mean())
    assert naive.mean_ > 1000.0
    assert protocol_scaler.mean_ == pytest.approx(1.0)
    assert protocol_scaler.mean_ != pytest.approx(naive.mean_)


def test_future_graph_dates_are_rejected() -> None:
    protocol = preferred_protocol()
    origin = protocol.origin_for_target(protocol.validation_targets()[0])
    target = protocol.validation_targets()[0]
    with pytest.raises(LookaheadError, match="target date"):
        require_information_through_origin(
            protocol.estimation_window(origin).append(pd.DatetimeIndex([target])),
            origin,
            target=target,
        )


def test_confirm_locked_by_default() -> None:
    protocol = preferred_protocol()
    assert LOCKED is True
    assert protocol.confirm_locked is True
    with pytest.raises(ConfirmLockedError, match="locked"):
        protocol.confirm_targets()
    with pytest.raises(ConfirmLockedError):
        protocol.forecast_schedule(BlockName.CONFIRM)
    visible = protocol.full_calendar(unlock_confirm=False)
    confirm = protocol.confirm_targets(unlock_confirm=True)
    assert visible.intersection(confirm).empty


def test_explicit_confirm_unlock() -> None:
    protocol = preferred_protocol()
    confirm = protocol.confirm_targets(unlock_confirm=True)
    assert len(confirm) == CONFIRM_MIN_DAYS
    schedule = protocol.forecast_schedule(BlockName.CONFIRM, unlock_confirm=True)
    assert len(schedule) == CONFIRM_MIN_DAYS
    assert schedule[0].target == confirm[0]
    calendar = protocol.full_calendar(unlock_confirm=True)
    assert calendar[-1] == confirm[-1]


def test_no_accidental_helper_bypasses_confirm_lock() -> None:
    protocol = preferred_protocol()
    confirm = protocol.confirm_targets(unlock_confirm=True)
    assert protocol.evaluation_targets().intersection(confirm).empty
    assert protocol.visible_calendar().intersection(confirm).empty
    assert protocol.all_evaluation_targets().intersection(confirm).empty
    assert protocol.screen_targets().intersection(confirm).empty
    assert protocol.validation_targets().intersection(confirm).empty
    with pytest.raises(ConfirmLockedError):
        getattr(protocol, "confirm")
    with pytest.raises(ConfirmLockedError):
        getattr(protocol, "confirm_dates")
    with pytest.raises(AttributeError):
        protocol.confirm_locked = False
    with pytest.raises(ConfirmLockedError):
        protocol.confirm_targets()
    with pytest.raises(ConfirmLockedError):
        build_temporal_protocol(trading_calendar(PREFERRED_N)).block_of(confirm[0])
    assert "1990" not in repr(protocol)


def test_screen_accessible_only_through_intended_interface() -> None:
    protocol = preferred_protocol()
    screen = protocol.screen_targets()
    assert protocol.block_targets(BlockName.SCREEN).equals(screen)
    assert protocol.block_of(screen[0]) is BlockName.SCREEN
    first_targets = pd.DatetimeIndex(
        [step.target for step in protocol.forecast_schedule(BlockName.SCREEN)]
    )
    assert first_targets.equals(screen)


def test_seed_set_preservation() -> None:
    protocol = preferred_protocol()
    assert protocol.seeds == DEFAULT_SEEDS == (0, 1, 2, 3, 4)
    assert protocol.seed_contract.aggregation == SEED_AGGREGATION
    assert protocol.seed_contract.report_all is True
    leaked = list(protocol.seeds)
    leaked.append(99)
    assert protocol.seeds == DEFAULT_SEEDS
    with pytest.raises(ValueError, match="best-seed"):
        SeedContract(allow_best_seed=True)
    with pytest.raises(ValueError, match="reported"):
        SeedContract(report_all=False)
    custom = build_temporal_protocol(
        trading_calendar(PREFERRED_N), seeds=(7, 11, 13)
    )
    assert custom.seeds == (7, 11, 13)


def test_tuning_budget_metadata() -> None:
    protocol = preferred_protocol()
    assert protocol.tuning_budget.max_configurations == DEFAULT_MAX_CONFIGURATIONS == 20
    assert protocol.tuning_budget.family is None
    ridge = build_temporal_protocol(
        trading_calendar(PREFERRED_N),
        max_configurations=20,
        family="Ridge-DRD",
    )
    har = build_temporal_protocol(
        trading_calendar(PREFERRED_N),
        max_configurations=20,
        family="HAR-DRD",
    )
    assert ridge.tuning_budget.max_configurations == har.tuning_budget.max_configurations
    assert ridge.tuning_budget.family == "Ridge-DRD"
    with pytest.raises(ValueError):
        TuningBudget(max_configurations=0)


def test_overnight_convention_recorded() -> None:
    protocol = preferred_protocol()
    assert (
        protocol.overnight.statistical_proxy
        == OvernightChannel.STATISTICAL_OPEN_TO_CLOSE.value
    )
    assert (
        protocol.overnight.economic_primary
        == OvernightChannel.ECONOMIC_INCLUDE_OVERNIGHT.value
    )
    assert (
        protocol.overnight.economic_robustness
        == OvernightChannel.ECONOMIC_OPEN_TO_CLOSE_ROBUSTNESS.value
    )


def test_block_roles_are_encoded() -> None:
    assert "hyperparameter tuning" in BLOCK_ROLES[BlockName.VALIDATION]
    assert "no tuning" in BLOCK_ROLES[BlockName.CONFIRM]
    assert "no seed selection" in BLOCK_ROLES[BlockName.CONFIRM]


def test_protocol_indexing_is_deterministic() -> None:
    dates = trading_calendar(PREFERRED_N)
    first = build_temporal_protocol(dates)
    second = build_temporal_protocol(dates)
    assert first.validation_targets().equals(second.validation_targets())
    assert first.screen_targets().equals(second.screen_targets())
    assert first.confirm_targets(unlock_confirm=True).equals(
        second.confirm_targets(unlock_confirm=True)
    )
    assert [s.origin for s in first.forecast_schedule(BlockName.VALIDATION)] == [
        s.origin for s in second.forecast_schedule(BlockName.VALIDATION)
    ]
    assert [s.refit for s in first.forecast_schedule(BlockName.SCREEN)] == [
        s.refit for s in second.forecast_schedule(BlockName.SCREEN)
    ]


def test_half_open_boundary_off_by_one() -> None:
    protocol = preferred_protocol()
    calendar = protocol.full_calendar(unlock_confirm=True)
    allocation = allocate_blocks(PREFERRED_N)
    assert allocation.history.end == allocation.validation.start
    assert allocation.validation.end == allocation.screen.start
    assert allocation.screen.end == allocation.confirm.start
    assert allocation.confirm.end == PREFERRED_N
    last_validation = protocol.validation_targets()[-1]
    first_screen = protocol.screen_targets()[0]
    last_screen = protocol.screen_targets()[-1]
    first_confirm = protocol.confirm_targets(unlock_confirm=True)[0]
    assert calendar.get_loc(first_screen) == calendar.get_loc(last_validation) + 1
    assert calendar.get_loc(first_confirm) == calendar.get_loc(last_screen) + 1
    last_screen_origin = protocol.origin_for_target(last_screen)
    last_screen_window = protocol.estimation_window(last_screen_origin)
    assert first_confirm not in last_screen_window
    last_step = protocol.forecast_schedule(BlockName.SCREEN)[-1]
    assert last_step.window_end_index == last_step.target_index
    with pytest.raises(LookaheadError):
        protocol.estimation_window(calendar[248])


def test_cannot_construct_unlocked_protocol() -> None:
    dates = trading_calendar(PREFERRED_N)
    allocation = allocate_blocks(PREFERRED_N)
    with pytest.raises(ConfirmLockedError, match="constructed locked"):
        TemporalProtocol(dates, allocation, confirm_locked=False)
