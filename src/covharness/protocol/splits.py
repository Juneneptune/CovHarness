"""Chronological VALIDATION / SCREEN / CONFIRM allocation with a CONFIRM lock.

Boundary convention. Trading dates form a strictly increasing unique index.
All protocol intervals are half-open on that index, written ``[start, end)``.
Evaluation blocks are defined on forecast *targets* (the day ``t+1`` being
scored). The first ``m`` dates are HISTORY used as burn-in for the first
origin. Ordering is HISTORY < VALIDATION < SCREEN < CONFIRM. Evaluation
target sets do not overlap. There is no shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    BlockName,
    OvernightConvention,
    SeedContract,
    TuningBudget,
)
from covharness.protocol.exceptions import (
    ConfirmLockedError,
    ProtocolAllocationError,
    TuningUnavailableError,
)
from covharness.protocol.rolling import (
    ForecastStep,
    build_schedule,
    estimation_window,
    origin_index_for_target,
    step_at_origin,
)


@dataclass(frozen=True)
class IndexSpan:
    """Half-open integer span ``[start, end)`` on the protocol calendar."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("span bounds must satisfy 0 <= start <= end")

    def __len__(self) -> int:
        return self.end - self.start

    def slice_of(self, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Return ``calendar[start:end]``."""
        return calendar[self.start : self.end]


@dataclass(frozen=True)
class BlockAllocation:
    """Integer target spans after applying the shortening rule."""

    history: IndexSpan
    validation: IndexSpan
    screen: IndexSpan
    confirm: IndexSpan
    validation_shortened: bool
    data_driven_tuning_available: bool
    validation_target_days: int
    m: int
    n_trading_days: int
    n_evaluation_targets: int


class TemporalProtocol:
    """Leak-proof split, rolling schedule, and CONFIRM lock.

    CONFIRM dates are not returned by public helpers unless
    ``unlock_confirm=True`` is passed to the method that retrieves them.
    The object is locked at construction. There is no mutating unlock.
    """

    def __init__(
        self,
        calendar: pd.DatetimeIndex,
        allocation: BlockAllocation,
        *,
        refit_cadence: int = DEFAULT_REFIT_CADENCE,
        seed_contract: SeedContract | None = None,
        tuning_budget: TuningBudget | None = None,
        overnight: OvernightConvention | None = None,
        confirm_locked: bool = LOCKED,
    ) -> None:
        self._calendar = pd.DatetimeIndex(calendar).copy()
        self._allocation = allocation
        self.refit_cadence = int(refit_cadence)
        self.seed_contract = seed_contract or SeedContract()
        self.tuning_budget = tuning_budget or TuningBudget()
        self.overnight = overnight or OvernightConvention()
        self.confirm_locked = bool(confirm_locked)
        if self.refit_cadence < 1:
            raise ValueError("refit_cadence must be at least 1")
        if not self.confirm_locked:
            raise ConfirmLockedError(
                "TemporalProtocol must be constructed locked. Pass "
                "unlock_confirm=True to confirm_targets or forecast_schedule."
            )
        self._frozen = True

    def __setattr__(self, name: str, value: object) -> None:
        if name != "_frozen" and self.__dict__.get("_frozen"):
            raise AttributeError("TemporalProtocol is frozen after construction")
        object.__setattr__(self, name, value)

    @property
    def m(self) -> int:
        return self._allocation.m

    @property
    def n_trading_days(self) -> int:
        return self._allocation.n_trading_days

    @property
    def n_evaluation_targets(self) -> int:
        return self._allocation.n_evaluation_targets

    @property
    def n_validation(self) -> int:
        return len(self._allocation.validation)

    @property
    def n_screen(self) -> int:
        return len(self._allocation.screen)

    @property
    def n_confirm(self) -> int:
        return len(self._allocation.confirm)

    @property
    def validation_shortened(self) -> bool:
        return self._allocation.validation_shortened

    @property
    def data_driven_tuning_available(self) -> bool:
        return self._allocation.data_driven_tuning_available

    @property
    def validation_target_days(self) -> int:
        return self._allocation.validation_target_days

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.seed_contract.seeds

    @property
    def roles(self) -> dict[BlockName, tuple[str, ...]]:
        return BLOCK_ROLES

    def require_data_driven_tuning(self) -> None:
        """Refuse to treat a zero-length VALIDATION block as completed tuning."""
        if not self.data_driven_tuning_available:
            raise TuningUnavailableError(
                "data-driven tuning is unavailable because VALIDATION has "
                f"length {self.n_validation}. Hyperparameter selection was "
                "not completed."
            )

    def history_dates(self) -> pd.DatetimeIndex:
        """Burn-in dates ``[0, m)``. Not an evaluation block."""
        return self._allocation.history.slice_of(self._calendar)

    def validation_targets(self) -> pd.DatetimeIndex:
        """VALIDATION target dates. Tuning and selection live here."""
        return self._allocation.validation.slice_of(self._calendar)

    def screen_targets(self) -> pd.DatetimeIndex:
        """SCREEN target dates. Frozen-candidate comparison only."""
        return self._allocation.screen.slice_of(self._calendar)

    def evaluation_targets(self) -> pd.DatetimeIndex:
        """VALIDATION then SCREEN. CONFIRM is omitted while locked."""
        return self.validation_targets().append(self.screen_targets())

    def visible_calendar(self) -> pd.DatetimeIndex:
        """Trading dates strictly before CONFIRM.

        The caller-supplied calendar is retained privately so that an unlocked
        confirmatory call can recover CONFIRM targets. Public helpers do not
        return those dates while locked.
        """
        confirm_start = self._allocation.confirm.start
        return self._calendar[:confirm_start]

    def confirm_targets(self, *, unlock_confirm: bool = False) -> pd.DatetimeIndex:
        """Return CONFIRM targets. Locked by default."""
        self._require_confirm_unlock(unlock_confirm)
        return self._allocation.confirm.slice_of(self._calendar)

    def all_evaluation_targets(self, *, unlock_confirm: bool = False) -> pd.DatetimeIndex:
        """VALIDATION, SCREEN, and CONFIRM if explicitly unlocked."""
        if not unlock_confirm:
            return self.evaluation_targets()
        return self.evaluation_targets().append(
            self.confirm_targets(unlock_confirm=True)
        )

    def origin_for_target(
        self, target: pd.Timestamp, *, unlock_confirm: bool = False
    ) -> pd.Timestamp:
        """Return the trading day immediately before the target."""
        self._guard_target(target, unlock_confirm=unlock_confirm)
        origin_index = origin_index_for_target(self._calendar, pd.Timestamp(target))
        return self._calendar[origin_index]

    def estimation_window(
        self, origin: pd.Timestamp, *, unlock_confirm: bool = False
    ) -> pd.DatetimeIndex:
        """Return the rolling ``m`` dates through ``origin``."""
        origin = pd.Timestamp(origin)
        self._guard_origin(origin, unlock_confirm=unlock_confirm)
        return estimation_window(self._calendar, origin, m=self.m)

    def forecast_step(
        self, origin: pd.Timestamp, *, unlock_confirm: bool = False, refit: bool = False
    ) -> ForecastStep:
        """Return the leak-free step whose origin is ``origin``."""
        origin = pd.Timestamp(origin)
        self._guard_origin(origin, unlock_confirm=unlock_confirm)
        return step_at_origin(self._calendar, origin, m=self.m, refit=refit)

    def forecast_schedule(
        self, block: BlockName | str, *, unlock_confirm: bool = False
    ) -> tuple[ForecastStep, ...]:
        """Rolling origin/target schedule for one evaluation block."""
        block_name = BlockName(block)
        targets = self.block_targets(block_name, unlock_confirm=unlock_confirm)
        return build_schedule(
            self._calendar,
            targets,
            m=self.m,
            refit_cadence=self.refit_cadence,
        )

    def block_targets(
        self, block: BlockName | str, *, unlock_confirm: bool = False
    ) -> pd.DatetimeIndex:
        """Return target dates for one named evaluation block."""
        block_name = BlockName(block)
        if block_name is BlockName.HISTORY:
            return self.history_dates()
        if block_name is BlockName.VALIDATION:
            return self.validation_targets()
        if block_name is BlockName.SCREEN:
            return self.screen_targets()
        if block_name is BlockName.CONFIRM:
            return self.confirm_targets(unlock_confirm=unlock_confirm)
        raise ValueError(f"unknown block {block_name}")

    def block_of(
        self, date: pd.Timestamp, *, unlock_confirm: bool = False
    ) -> BlockName:
        """Name the region containing ``date``. CONFIRM requires unlock."""
        index = self._locate_or_raise(date)
        if index in range(self._allocation.confirm.start, self._allocation.confirm.end):
            self._require_confirm_unlock(unlock_confirm)
            return BlockName.CONFIRM
        if index in range(self._allocation.screen.start, self._allocation.screen.end):
            return BlockName.SCREEN
        if index in range(
            self._allocation.validation.start, self._allocation.validation.end
        ):
            return BlockName.VALIDATION
        if index in range(self._allocation.history.start, self._allocation.history.end):
            return BlockName.HISTORY
        raise KeyError(f"{pd.Timestamp(date).date()} is not a protocol region date")

    def full_calendar(self, *, unlock_confirm: bool = False) -> pd.DatetimeIndex:
        """Return the constructor calendar. CONFIRM dates require unlock."""
        if unlock_confirm:
            return self._calendar.copy()
        return self.visible_calendar()

    def _require_confirm_unlock(self, unlock_confirm: bool) -> None:
        if not unlock_confirm:
            raise ConfirmLockedError(
                "CONFIRM is locked. Pass unlock_confirm=True at this protocol "
                "boundary. The later runner will use --unlock-confirm. The "
                "default is locked."
            )

    def _guard_target(self, target: pd.Timestamp, *, unlock_confirm: bool) -> None:
        target_index = self._locate_or_raise(target)
        if target_index >= self._allocation.confirm.start:
            self._require_confirm_unlock(unlock_confirm)

    def _guard_origin(self, origin: pd.Timestamp, *, unlock_confirm: bool) -> None:
        origin_index = self._locate_or_raise(origin)
        target_index = origin_index + 1
        if target_index >= self._allocation.confirm.start:
            self._require_confirm_unlock(unlock_confirm)

    def _locate_or_raise(self, stamp: pd.Timestamp) -> int:
        try:
            located = self._calendar.get_loc(pd.Timestamp(stamp))
        except KeyError as exc:
            raise KeyError(
                f"{pd.Timestamp(stamp).date()} is not on the protocol calendar"
            ) from exc
        if not isinstance(located, (int, np.integer)):
            raise KeyError("date must match exactly one calendar row")
        return int(located)

    def __getattr__(self, name: str) -> object:
        """Reject undeclared confirm helpers so they cannot bypass the lock."""
        if "confirm" in name.lower():
            raise ConfirmLockedError(
                f"{name!r} is not a public CONFIRM accessor. Use "
                "confirm_targets(unlock_confirm=True)."
            )
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __repr__(self) -> str:
        return (
            "TemporalProtocol("
            f"m={self.m}, "
            f"n_validation={self.n_validation}, "
            f"n_screen={self.n_screen}, "
            f"n_confirm={self.n_confirm}, "
            f"validation_shortened={self.validation_shortened}, "
            f"data_driven_tuning_available={self.data_driven_tuning_available}, "
            f"confirm_locked={self.confirm_locked})"
        )


def build_temporal_protocol(
    dates: pd.DatetimeIndex,
    *,
    m: int = DEFAULT_ROLLING_WINDOW,
    validation_target: int = VALIDATION_TARGET_DAYS,
    min_screen: int = SCREEN_MIN_DAYS,
    min_confirm: int = CONFIRM_MIN_DAYS,
    refit_cadence: int = DEFAULT_REFIT_CADENCE,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_configurations: int = DEFAULT_MAX_CONFIGURATIONS,
    family: str | None = None,
) -> TemporalProtocol:
    """Allocate HISTORY / VALIDATION / SCREEN / CONFIRM on ``dates``.

    ``validation_target`` is the preferred VALIDATION length, not a hard
    minimum. SCREEN and CONFIRM are never shortened. If the remaining
    evaluation length after locking those two blocks is below
    ``validation_target``, VALIDATION is shortened and that fact is reported.
    A zero-length VALIDATION block makes data-driven tuning unavailable.
    If SCREEN+CONFIRM cannot fit after the rolling burn-in, allocation fails.
    """
    calendar = _require_ordered_unique(dates)
    allocation = allocate_blocks(
        n_dates=len(calendar),
        m=m,
        validation_target=validation_target,
        min_screen=min_screen,
        min_confirm=min_confirm,
    )
    return TemporalProtocol(
        calendar,
        allocation,
        refit_cadence=refit_cadence,
        seed_contract=SeedContract(seeds=tuple(seeds)),
        tuning_budget=TuningBudget(
            max_configurations=max_configurations, family=family
        ),
        overnight=OvernightConvention(),
        confirm_locked=LOCKED,
    )


def allocate_blocks(
    n_dates: int,
    *,
    m: int = DEFAULT_ROLLING_WINDOW,
    validation_target: int = VALIDATION_TARGET_DAYS,
    min_screen: int = SCREEN_MIN_DAYS,
    min_confirm: int = CONFIRM_MIN_DAYS,
) -> BlockAllocation:
    """Central allocation. Experiments must not reimplement this rule."""
    if m < 1:
        raise ValueError("rolling window m must be at least 1")
    if min_screen < 1 or min_confirm < 1:
        raise ValueError("SCREEN and CONFIRM minima must be at least 1")
    if validation_target < 0:
        raise ValueError("VALIDATION target cannot be negative")
    if n_dates < m + min_screen + min_confirm:
        n_eval = max(n_dates - m, 0)
        raise ProtocolAllocationError(
            "confirmatory design is infeasible. "
            f"Need at least m+SCREEN+CONFIRM = {m + min_screen + min_confirm} "
            f"trading days; received {n_dates} dates "
            f"({n_eval} evaluation targets after m={m}). "
            "SCREEN and CONFIRM are not shortened to make a sample fit."
        )
    n_eval = n_dates - m
    screen_len = min_screen
    confirm_len = min_confirm
    validation_len = n_eval - screen_len - confirm_len
    validation_shortened = validation_len < validation_target
    history = IndexSpan(0, m)
    validation = IndexSpan(m, m + validation_len)
    screen = IndexSpan(m + validation_len, m + validation_len + screen_len)
    confirm = IndexSpan(m + validation_len + screen_len, n_dates)
    return BlockAllocation(
        history=history,
        validation=validation,
        screen=screen,
        confirm=confirm,
        validation_shortened=validation_shortened,
        data_driven_tuning_available=validation_len > 0,
        validation_target_days=validation_target,
        m=m,
        n_trading_days=n_dates,
        n_evaluation_targets=n_eval,
    )


def _require_ordered_unique(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Reject shuffled or duplicate calendars."""
    calendar = pd.DatetimeIndex(dates)
    if len(calendar) == 0:
        raise ValueError("the trading calendar is empty")
    if not calendar.is_unique:
        raise ValueError("duplicate trading dates are not permitted")
    if not calendar.is_monotonic_increasing:
        raise ValueError("shuffling is not permitted; dates must be strictly increasing")
    return calendar.copy()
