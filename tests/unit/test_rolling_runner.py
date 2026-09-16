"""Serial synthetic rolling-runner integration tests. No market data."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import torch

from covharness.models import (
    DCCCovariance,
    DCCNonlinearCovariance,
    EWMARealizedCovariance,
    HARDRDRealizedCovariance,
    HARQDRDRealizedCovariance,
    LSTMBEKKCovariance,
    LedoitWolfLinearCovariance,
    LedoitWolfNonlinearCovariance,
    RandomWalkRealizedCovariance,
    RidgeDRDRealizedCovariance,
    RollingCadence,
    XGBoostDRDRealizedCovariance,
    capabilities_of,
)
import covharness.models.lstm_bekk as lstm_bekk_module
from covharness.models.xgboost_drd import booster_raw_bytes
from covharness.protocol import (
    DEFAULT_REFIT_CADENCE,
    DEFAULT_ROLLING_WINDOW,
    BlockName,
    ConfirmLockedError,
    RollingAction,
    build_origin_payload,
    build_schedule,
    build_temporal_protocol,
    run_block_forecasts,
    run_rolling_forecasts,
)

SEED = 20260916
M = DEFAULT_ROLLING_WINDOW
N_TARGETS = 22
N_ASSETS = 2
N_DATES = M + N_TARGETS


def _calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("1990-01-01", periods=N_DATES)


def _schedule(calendar: pd.DatetimeIndex):
    targets = calendar[M:]
    return build_schedule(calendar, targets, m=M, refit_cadence=DEFAULT_REFIT_CADENCE)


def _garch_returns(n_times: int, n_assets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.empty((n_times, n_assets))
    variance = np.full(n_assets, 0.02)
    for time in range(n_times):
        shock = rng.standard_normal(n_assets)
        returns[time] = np.sqrt(variance) * shock
        variance = 0.05 + 0.15 * returns[time] ** 2 + 0.75 * variance
    return returns


def _spd_cube(n_times: int, n_assets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cube = np.empty((n_times, n_assets, n_assets))
    for time in range(n_times):
        factor = rng.normal(size=(n_assets, n_assets + 1))
        matrix = factor @ factor.T / factor.shape[1] + np.eye(n_assets)
        cube[time] = 0.5 * (matrix + matrix.T)
    return cube


def _rq_panel(n_times: int, n_assets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.08, 1.6, size=(n_times, n_assets))


def _panels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    returns = _garch_returns(N_DATES, N_ASSETS, SEED)
    rcov = _spd_cube(N_DATES, N_ASSETS, SEED + 1)
    rq = _rq_panel(N_DATES, N_ASSETS, SEED + 2)
    return returns, rcov, rq


def _instrument(model: object) -> dict[str, int]:
    counts = {"fit": 0, "update": 0, "update_window": 0}
    original_fit = model.fit

    def fit(*args, **kwargs):
        counts["fit"] += 1
        return original_fit(*args, **kwargs)

    model.fit = fit  # type: ignore[method-assign]
    if hasattr(model, "update"):
        original_update = model.update

        def update(*args, **kwargs):
            counts["update"] += 1
            return original_update(*args, **kwargs)

        model.update = update  # type: ignore[method-assign]
    if hasattr(model, "update_window"):
        original_refresh = model.update_window

        def update_window(*args, **kwargs):
            counts["update_window"] += 1
            return original_refresh(*args, **kwargs)

        model.update_window = update_window  # type: ignore[method-assign]
    return counts


def _run(model, calendar, schedule, returns, rcov, rq):
    return run_rolling_forecasts(
        model=model,
        schedule=schedule,
        calendar=calendar,
        daily_returns=returns,
        realized_covariances=rcov,
        realized_quarticity=rq,
    )


def _xgboost() -> XGBoostDRDRealizedCovariance:
    return XGBoostDRDRealizedCovariance(
        n_estimators=2,
        max_depth=1,
        learning_rate=0.3,
        min_child_weight=0.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        gamma=0.0,
    )


def _lstm() -> LSTMBEKKCovariance:
    return LSTMBEKKCovariance(
        seed=0,
        num_layers=3,
        dropout=0.1,
        learning_rate=0.05,
        gradient_clip_norm=1.0,
        max_epochs=1,
    )


def test_origin_target_and_refit_schedule() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    assert len(schedule) == N_TARGETS
    for position, step in enumerate(schedule):
        assert step.origin_index + 1 == step.target_index
        assert step.window_end_index == step.origin_index + 1
        assert step.window_end_index - step.window_start_index == M
        assert step.m == M
        assert step.origin == calendar[step.origin_index]
        assert step.target == calendar[step.target_index]
        assert step.refit is (position % DEFAULT_REFIT_CADENCE == 0)
    assert [step.origin_index for step in schedule] == list(range(M - 1, M - 1 + N_TARGETS))


def test_payloads_exclude_target_rows() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    for step in schedule:
        payload = build_origin_payload(
            step=step,
            calendar=calendar,
            daily_returns=returns,
            realized_covariances=rcov,
            realized_quarticity=rq,
        )
        assert payload.return_window is not None
        assert payload.rcov_window is not None
        assert payload.rq_window is not None
        assert payload.return_window.shape == (M, N_ASSETS)
        assert payload.rcov_window.shape == (M, N_ASSETS, N_ASSETS)
        assert payload.rq_window.shape == (M, N_ASSETS)
        np.testing.assert_array_equal(payload.return_window[-1], returns[step.origin_index])
        np.testing.assert_array_equal(payload.current_return, returns[step.origin_index])
        np.testing.assert_array_equal(payload.current_rcov, rcov[step.origin_index])
        np.testing.assert_array_equal(payload.current_rq, rq[step.origin_index])
        assert payload.return_window.shape[0] == M
        assert not np.shares_memory(payload.return_window, returns)
        leaked_return = returns[step.target_index]
        assert not np.array_equal(payload.current_return, leaked_return)
        leaked_rcov = rcov[step.target_index]
        assert not np.array_equal(payload.current_rcov, leaked_rcov)
        leaked_rq = rq[step.target_index]
        assert not np.array_equal(payload.current_rq, leaked_rq)


def test_call_counts_match_cadence() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    n_refit = sum(step.refit for step in schedule)
    n_hold = len(schedule) - n_refit
    roster: list[tuple[object, RollingCadence]] = [
        (RandomWalkRealizedCovariance(), RollingCadence.ORIGIN_MAP),
        (EWMARealizedCovariance(decay=0.94), RollingCadence.RECURSIVE_STATE),
        (HARDRDRealizedCovariance(), RollingCadence.WINDOW_STATE),
        (HARQDRDRealizedCovariance(), RollingCadence.WINDOW_STATE),
        (RidgeDRDRealizedCovariance(lambda_=0.5), RollingCadence.WINDOW_STATE),
        (_xgboost(), RollingCadence.WINDOW_STATE),
        (LedoitWolfLinearCovariance(), RollingCadence.REFIT_HOLD),
        (LedoitWolfNonlinearCovariance(), RollingCadence.REFIT_HOLD),
        (DCCCovariance(), RollingCadence.RECURSIVE_STATE),
        (DCCNonlinearCovariance(), RollingCadence.RECURSIVE_STATE),
    ]
    for model, cadence in roster:
        counts = _instrument(model)
        records = _run(model, calendar, schedule, returns, rcov, rq)
        assert counts["fit"] == n_refit
        if cadence is RollingCadence.REFIT_HOLD:
            assert counts["update"] == 0
            assert counts["update_window"] == 0
            assert all(
                record.action is (RollingAction.FIT if record.refit else RollingAction.HOLD)
                for record in records
            )
        elif cadence is RollingCadence.WINDOW_STATE:
            assert counts["update_window"] == n_hold
            assert counts["update"] == 0
        else:
            assert counts["update"] == n_hold
            assert counts["update_window"] == 0
        assert [record.refit for record in records] == [step.refit for step in schedule]


def test_forecast_records_common_invariants() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    models = {
        "random_walk_rcov": RandomWalkRealizedCovariance(),
        "ewma_rcov": EWMARealizedCovariance(decay=0.94),
        "har_drd": HARDRDRealizedCovariance(),
        "harq_drd": HARQDRDRealizedCovariance(),
        "ridge_drd": RidgeDRDRealizedCovariance(lambda_=0.25),
        "xgboost_drd": _xgboost(),
        "lw_linear": LedoitWolfLinearCovariance(),
        "lw_nl": LedoitWolfNonlinearCovariance(),
        "dcc": DCCCovariance(),
        "dcc_nl": DCCNonlinearCovariance(),
    }
    target_sets = []
    for name, model in models.items():
        records = _run(model, calendar, schedule, returns, rcov, rq)
        assert len(records) == len(schedule)
        target_sets.append(tuple(record.target for record in records))
        for record, step in zip(records, schedule, strict=True):
            assert record.model_name == name
            assert record.origin == step.origin
            assert record.target == step.target
            assert record.refit is step.refit
            assert record.covariance.shape == (N_ASSETS, N_ASSETS)
            probe = np.array(record.covariance, copy=True)
            probe[0, 0] = -99.0
            assert record.covariance[0, 0] != -99.0
    assert len(set(target_sets)) == 1


def test_rw_and_ewma_move_daily() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    rw = RandomWalkRealizedCovariance()
    ewma = EWMARealizedCovariance(decay=0.8)
    rw_records = _run(rw, calendar, schedule, returns, rcov, rq)
    ewma_records = _run(ewma, calendar, schedule, returns, rcov, rq)
    for index, step in enumerate(schedule):
        np.testing.assert_allclose(rw_records[index].covariance, rcov[step.origin_index])
        if index > 0:
            assert not np.allclose(
                rw_records[index].covariance, rw_records[index - 1].covariance
            )
            assert not np.allclose(
                ewma_records[index].covariance, ewma_records[index - 1].covariance
            )
    assert ewma.identity.configuration["decay"] == 0.8
    expected = rcov[schedule[0].window_start_index]
    decay = 0.8
    for row in rcov[schedule[0].window_start_index + 1 : schedule[0].window_end_index]:
        expected = decay * expected + (1.0 - decay) * row
    np.testing.assert_allclose(ewma_records[0].covariance, expected)
    for index in range(1, DEFAULT_REFIT_CADENCE):
        expected = decay * expected + (1.0 - decay) * rcov[schedule[index].origin_index]
        np.testing.assert_allclose(ewma_records[index].covariance, expected)


def test_har_harq_ridge_move_with_frozen_fit_and_current_fallback() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    har = HARDRDRealizedCovariance()
    harq = HARQDRDRealizedCovariance()
    ridge = RidgeDRDRealizedCovariance(lambda_=0.0)
    har_records = _run(har, calendar, schedule, returns, rcov, rq)
    harq_records = _run(harq, calendar, schedule, returns, rcov, rq)
    ridge_records = _run(ridge, calendar, schedule, returns, rcov, rq)
    first_refit = next(index for index, step in enumerate(schedule) if step.refit)
    second_refit = next(
        index for index, step in enumerate(schedule) if step.refit and index > 0
    )
    first_window = rcov[
        schedule[first_refit].window_start_index : schedule[first_refit].window_end_index
    ]
    first_rq = rq[
        schedule[first_refit].window_start_index : schedule[first_refit].window_end_index
    ]
    har_ref = HARDRDRealizedCovariance().fit(first_window)
    ridge_ref = RidgeDRDRealizedCovariance(lambda_=0.0).fit(first_window)
    harq_ref = HARQDRDRealizedCovariance().fit(first_window, first_rq)
    for index, record in enumerate(har_records):
        np.testing.assert_allclose(
            record.covariance, ridge_records[index].covariance, atol=1e-8
        )
    har_clone = HARDRDRealizedCovariance().fit(first_window)
    ridge_clone = RidgeDRDRealizedCovariance(lambda_=0.0).fit(first_window)
    harq_clone = HARQDRDRealizedCovariance().fit(first_window, first_rq)
    np.testing.assert_array_equal(ridge_clone.fit_state.scale_D, ridge_ref.fit_state.scale_D)
    for index in range(first_refit + 1, second_refit):
        assert not np.allclose(har_records[index].covariance, har_records[index - 1].covariance)
        assert not np.allclose(harq_records[index].covariance, harq_records[index - 1].covariance)
        window = rcov[schedule[index].window_start_index : schedule[index].window_end_index]
        rq_window = rq[schedule[index].window_start_index : schedule[index].window_end_index]
        har_clone.update_window(window)
        ridge_clone.update_window(window)
        harq_clone.update_window(window, rq_window)
        np.testing.assert_array_equal(har_clone.fit_state.beta_D, har_ref.fit_state.beta_D)
        np.testing.assert_array_equal(har_clone.fit_state.alpha_D, har_ref.fit_state.alpha_D)
        np.testing.assert_array_equal(ridge_clone.fit_state.scale_D, ridge_ref.fit_state.scale_D)
        np.testing.assert_array_equal(ridge_clone.fit_state.beta_D, ridge_ref.fit_state.beta_D)
        np.testing.assert_array_equal(harq_clone.fit_state.beta_Q, harq_ref.fit_state.beta_Q)
        np.testing.assert_allclose(har_clone.fit_state.window_mean, window.mean(axis=0))
        np.testing.assert_allclose(ridge_clone.fit_state.window_mean, window.mean(axis=0))
        np.testing.assert_allclose(harq_clone.fit_state.window_mean, window.mean(axis=0))
        np.testing.assert_allclose(har_clone.forecast().matrix, har_records[index].covariance)
        np.testing.assert_allclose(
            ridge_clone.forecast().matrix, ridge_records[index].covariance
        )
        np.testing.assert_allclose(
            harq_clone.forecast().matrix, harq_records[index].covariance
        )


def test_xgboost_window_state_matches_har_ridge_cadence() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    model = _xgboost()
    counts = _instrument(model)
    records = _run(model, calendar, schedule, returns, rcov, rq)
    n_refit = sum(step.refit for step in schedule)
    n_hold = len(schedule) - n_refit
    assert counts["fit"] == n_refit
    assert counts["update_window"] == n_hold
    assert counts["update"] == 0
    assert all(
        record.action is (RollingAction.FIT if record.refit else RollingAction.UPDATE_WINDOW)
        for record in records
    )
    har_records = _run(HARDRDRealizedCovariance(), calendar, schedule, returns, rcov, rq)
    ridge_records = _run(
        RidgeDRDRealizedCovariance(lambda_=0.25), calendar, schedule, returns, rcov, rq
    )
    assert [record.target for record in records] == [record.target for record in har_records]
    assert [record.origin for record in records] == [record.origin for record in ridge_records]
    first_refit = next(index for index, step in enumerate(schedule) if step.refit)
    second_refit = next(
        index for index, step in enumerate(schedule) if step.refit and index > 0
    )
    first_window = rcov[
        schedule[first_refit].window_start_index : schedule[first_refit].window_end_index
    ]
    clone = _xgboost().fit(first_window)
    ybar_d = clone.fit_state.ybar_D.copy()
    xbar_d = clone.fit_state.Xbar_D.copy()
    scale_d = clone.fit_state.scale_D.copy()
    bytes_d = booster_raw_bytes(clone.fit_state.variance_booster)
    bytes_r = booster_raw_bytes(clone.fit_state.correlation_booster)
    np.testing.assert_allclose(clone.forecast().matrix, records[first_refit].covariance)
    np.testing.assert_allclose(clone.fit_state.window_mean, first_window.mean(axis=0))
    moved = False
    for index in range(first_refit + 1, second_refit):
        window = rcov[schedule[index].window_start_index : schedule[index].window_end_index]
        clone.update_window(window)
        np.testing.assert_array_equal(clone.fit_state.ybar_D, ybar_d)
        np.testing.assert_array_equal(clone.fit_state.Xbar_D, xbar_d)
        np.testing.assert_array_equal(clone.fit_state.scale_D, scale_d)
        assert booster_raw_bytes(clone.fit_state.variance_booster) == bytes_d
        assert booster_raw_bytes(clone.fit_state.correlation_booster) == bytes_r
        np.testing.assert_allclose(clone.fit_state.window_mean, window.mean(axis=0))
        np.testing.assert_allclose(clone.forecast().matrix, records[index].covariance)
        if not np.allclose(records[index].covariance, records[index - 1].covariance):
            moved = True
    assert moved


def test_lw_held_between_refits() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    for factory in (LedoitWolfLinearCovariance, LedoitWolfNonlinearCovariance):
        model = factory()
        counts = _instrument(model)
        records = _run(model, calendar, schedule, returns, rcov, rq)
        assert counts["fit"] == 2
        assert counts["update"] == 0
        held = records[0].covariance
        for index in range(1, DEFAULT_REFIT_CADENCE):
            np.testing.assert_allclose(records[index].covariance, held)
            assert records[index].action is RollingAction.HOLD
        assert records[DEFAULT_REFIT_CADENCE].refit is True
        assert records[DEFAULT_REFIT_CADENCE].action is RollingAction.FIT
        assert not np.allclose(records[DEFAULT_REFIT_CADENCE].covariance, held)


def test_dcc_moves_daily_without_parameter_refit() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    for factory in (DCCCovariance, DCCNonlinearCovariance):
        model = factory()
        counts = _instrument(model)
        records = _run(model, calendar, schedule, returns, rcov, rq)
        assert counts["fit"] == 2
        assert counts["update"] == N_TARGETS - 2
        first_fit_window = returns[
            schedule[0].window_start_index : schedule[0].window_end_index
        ]
        clone = factory().fit(first_fit_window)
        alpha = clone.fit_state.dcc_alpha
        omega = clone.fit_state.omega.copy()
        fit_mean = clone.fit_state.fit_mean.copy()
        for index in range(1, DEFAULT_REFIT_CADENCE):
            clone.update(returns[schedule[index].origin_index])
            np.testing.assert_allclose(clone.forecast().matrix, records[index].covariance)
            assert clone.fit_state.dcc_alpha == alpha
            np.testing.assert_array_equal(clone.fit_state.omega, omega)
            np.testing.assert_array_equal(clone.fit_state.fit_mean, fit_mean)
            assert not np.allclose(records[index].covariance, records[index - 1].covariance)


def test_lstm_moves_daily_without_reseed_or_weight_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    seed_calls: list[int] = []
    original_seed = lstm_bekk_module.apply_cpu_seeds

    def counted_seed(seed: int) -> None:
        seed_calls.append(int(seed))
        original_seed(seed)

    monkeypatch.setattr(lstm_bekk_module, "apply_cpu_seeds", counted_seed)
    model = _lstm()
    counts = _instrument(model)
    records = _run(model, calendar, schedule, returns, rcov, rq)
    assert counts["fit"] == 2
    assert counts["update"] == N_TARGETS - 2
    assert seed_calls == [0, 0]
    first_fit_window = returns[
        schedule[0].window_start_index : schedule[0].window_end_index
    ]
    clone = _lstm().fit(first_fit_window)
    assert clone._fit is not None
    assert clone._network is not None
    weights = [tensor.detach().clone() for tensor in clone._network.parameters()]
    fit_mean = clone._fit.fit_mean.copy()
    for index in range(1, DEFAULT_REFIT_CADENCE):
        clone.update(returns[schedule[index].origin_index])
        np.testing.assert_allclose(clone.forecast().matrix, records[index].covariance)
        np.testing.assert_array_equal(clone._fit.fit_mean, fit_mean)
        for current, original in zip(clone._network.parameters(), weights, strict=True):
            np.testing.assert_array_equal(
                current.detach().cpu().numpy(), original.detach().cpu().numpy()
            )
        assert not np.allclose(records[index].covariance, records[index - 1].covariance)
    assert torch.get_num_threads() >= 1


def test_confirm_cannot_be_requested_while_locked() -> None:
    calendar = pd.bdate_range("1990-01-01", periods=1500)
    protocol = build_temporal_protocol(calendar)
    rcov = _spd_cube(len(calendar), N_ASSETS, SEED + 9)
    with pytest.raises(ConfirmLockedError):
        run_block_forecasts(
            model=RandomWalkRealizedCovariance(),
            protocol=protocol,
            block=BlockName.CONFIRM,
            realized_covariances=rcov,
        )


def test_deterministic_models_repeat() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    factories: list[Callable[[], object]] = [
        RandomWalkRealizedCovariance,
        lambda: EWMARealizedCovariance(decay=0.94),
        HARDRDRealizedCovariance,
        HARQDRDRealizedCovariance,
        lambda: RidgeDRDRealizedCovariance(lambda_=0.3),
        _xgboost,
        LedoitWolfLinearCovariance,
        LedoitWolfNonlinearCovariance,
        DCCCovariance,
        DCCNonlinearCovariance,
    ]
    for factory in factories:
        first = _run(factory(), calendar, schedule, returns, rcov, rq)
        second = _run(factory(), calendar, schedule, returns, rcov, rq)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_allclose(left.covariance, right.covariance)
            assert left.origin == right.origin
            assert left.target == right.target


def test_lstm_seed_reproducibility() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    first = _run(_lstm(), calendar, schedule, returns, rcov, rq)
    second = _run(_lstm(), calendar, schedule, returns, rcov, rq)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.covariance, right.covariance)


def test_adjacent_origins_ingest_one_new_observation() -> None:
    calendar = _calendar()
    schedule = _schedule(calendar)
    returns, rcov, rq = _panels()
    payloads = [
        build_origin_payload(
            step=step,
            calendar=calendar,
            daily_returns=returns,
            realized_covariances=rcov,
            realized_quarticity=rq,
        )
        for step in schedule
    ]
    for previous, current in zip(payloads, payloads[1:]):
        assert current.origin_index == previous.origin_index + 1
        assert previous.rcov_window is not None and current.rcov_window is not None
        np.testing.assert_array_equal(current.rcov_window[:-1], previous.rcov_window[1:])
        np.testing.assert_array_equal(current.rcov_window[-1], current.current_rcov)
        assert previous.return_window is not None and current.return_window is not None
        np.testing.assert_array_equal(
            current.return_window[:-1], previous.return_window[1:]
        )
        np.testing.assert_array_equal(current.return_window[-1], current.current_return)


def test_capabilities_are_explicit() -> None:
    assert capabilities_of(RandomWalkRealizedCovariance()).rolling_cadence is RollingCadence.ORIGIN_MAP
    assert capabilities_of(EWMARealizedCovariance(decay=0.5)).rolling_cadence is RollingCadence.RECURSIVE_STATE
    assert capabilities_of(HARDRDRealizedCovariance()).rolling_cadence is RollingCadence.WINDOW_STATE
    assert capabilities_of(_xgboost()).rolling_cadence is RollingCadence.WINDOW_STATE
    assert capabilities_of(LedoitWolfLinearCovariance()).rolling_cadence is RollingCadence.REFIT_HOLD
    assert capabilities_of(DCCCovariance()).rolling_cadence is RollingCadence.RECURSIVE_STATE
    assert capabilities_of(_lstm()).rolling_cadence is RollingCadence.RECURSIVE_STATE
