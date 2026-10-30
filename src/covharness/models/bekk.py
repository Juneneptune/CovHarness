"""Scalar, diagonal, and full BEKK(1,1) on daily returns.

The Gaussian NLL, percent scaling, and H0 rule are the LSTM-BEKK helpers.
Scalar steps call the imported recursion with a zero dynamic intercept.
Carry-forward is the primary failure policy. The interior map is a
robustness variant and is not headline-eligible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from covharness.models.base import CovarianceModel, ModelIdentity, pack_forecast
from covharness.models.capabilities import (
    FitInput,
    ModelCapabilities,
    RollingCadence,
    UpdateObservable,
)
from covharness.models.exceptions import (
    InvalidModelForecastError,
    InvalidModelInputError,
)
from covharness.models.lstm_bekk import (
    A0,
    B0,
    COVARIANCE_RESCALE,
    INTERNAL_RETURN_SCALE,
    LSTMBEKKCovariance,
    W0,
    _as_new_return,
    _lstm_bekk_return_window,
    bekk_recursion,
    construct_h0,
    gaussian_nll_cholesky_numpy,
    packed_static_c_from_h0,
    prepare_internal_returns,
    triangle_parameter_count,
    unpack_static_c_numpy,
)
from covharness.models.stationarity import (
    STATIONARITY_EPSILON,
    POLICY_CARRY_FORWARD,
    POLICY_INTERIOR_MAP,
    softmax_persistence,
)

MODEL_NAME = "bekk"
OPTIMIZER_NAME = "L-BFGS-B"
MAX_ITER = 100
FORMS = ("scalar", "diagonal", "full")
_PENALTY = 1.0e12


@dataclass
class BEKKFitState:
    """Frozen BEKK parameters and the one-step filter after one window."""

    fit_mean: NDArray[np.floating]
    h0: NDArray[np.floating]
    intercept: NDArray[np.floating]
    a_matrix: NDArray[np.floating]
    b_matrix: NDArray[np.floating]
    scalar_a: float
    scalar_b: float
    current_h: NDArray[np.floating]
    form: str
    targeting: bool
    failure_policy: str
    carry_count: int
    n_parameters: int
    final_nll: float


class BEKKCovariance(CovarianceModel):
    """One BEKK(1,1) specification. Refit cadence belongs to the runner."""

    model_name = MODEL_NAME
    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.RECURSIVE_STATE,
        fit_input=FitInput.DAILY_RETURN,
        update_observable=UpdateObservable.DAILY_RETURN,
    )

    def __init__(
        self,
        *,
        form: str,
        targeting: bool,
        failure_policy: str = POLICY_CARRY_FORWARD,
        specification_id: str,
    ) -> None:
        if form not in FORMS:
            raise InvalidModelInputError(f"unknown BEKK form {form}")
        if failure_policy not in (POLICY_CARRY_FORWARD, POLICY_INTERIOR_MAP):
            raise InvalidModelInputError(f"unknown BEKK failure policy {failure_policy}")
        self._form = form
        self._targeting = bool(targeting)
        self._failure_policy = failure_policy
        self._specification_id = specification_id
        self._state: BEKKFitState | None = None

    @property
    def identity(self) -> ModelIdentity:
        configuration: dict[str, object] = {
            "specification_id": self._specification_id,
            "form": self._form,
            "targeting": self._targeting,
            "failure_policy": self._failure_policy,
            "optimizer": OPTIMIZER_NAME,
            "max_iter": MAX_ITER,
            "internal_return_scale": INTERNAL_RETURN_SCALE,
            "covariance_rescale": COVARIANCE_RESCALE,
        }
        if self._state is not None:
            configuration.update(
                {
                    "scalar_a": self._state.scalar_a,
                    "scalar_b": self._state.scalar_b,
                    "n_parameters": self._state.n_parameters,
                    "carry_count": self._state.carry_count,
                    "final_nll": self._state.final_nll,
                }
            )
        return ModelIdentity(name=self.model_name, configuration=configuration)

    @property
    def carry_count(self) -> int:
        return 0 if self._state is None else self._state.carry_count

    def fit(self, returns: ArrayLike) -> BEKKCovariance:
        """QMLE on one origin window, or carry the previous parameters forward."""
        history = _lstm_bekk_return_window(returns)
        fit_mean, _native, internal = prepare_internal_returns(history)
        h0, _method = construct_h0(internal)
        try:
            state = _estimate(internal, h0, fit_mean, self._form, self._targeting, self._failure_policy)
        except (InvalidModelForecastError, InvalidModelInputError, np.linalg.LinAlgError):
            if self._failure_policy != POLICY_CARRY_FORWARD or self._state is None:
                raise
            state = _replay_carried(internal, h0, fit_mean, self._state)
        self._state = state
        return self

    def update(self, new_return: ArrayLike) -> BEKKCovariance:
        """Advance the filter by one return. Parameters stay frozen."""
        if self._state is None:
            raise InvalidModelInputError("BEKK update requires a fitted state")
        observed = _as_new_return(new_return, self._state.current_h.shape[0])
        shock = INTERNAL_RETURN_SCALE * (observed - self._state.fit_mean)
        nxt = bekk_transition(
            self._state.intercept,
            self._state.a_matrix,
            self._state.b_matrix,
            shock,
            self._state.current_h,
            self._form,
        )
        _require_pd(nxt)
        self._state.current_h = nxt
        return self

    def forecast(self):
        if self._state is None:
            raise InvalidModelInputError("BEKK forecast requires a fitted state")
        native = self._state.current_h / COVARIANCE_RESCALE
        _require_pd(native)
        return pack_forecast(native, self.identity)


def bekk_transition(
    intercept: NDArray[np.floating],
    a_matrix: NDArray[np.floating],
    b_matrix: NDArray[np.floating],
    shock: NDArray[np.floating],
    h_prev: NDArray[np.floating],
    form: str,
) -> NDArray[np.floating]:
    """One BEKK step. The scalar form calls the imported recursion."""
    if form == "scalar":
        a_value = float(a_matrix[0, 0])
        b_value = float(b_matrix[0, 0])
        factor = np.linalg.cholesky(intercept)
        zeros = np.zeros_like(intercept)
        return bekk_recursion(factor, zeros, a_value, b_value, shock, h_prev)
    # Diagonal and full use A x x' A' and B H B'.
    ax = a_matrix @ shock
    return intercept + np.outer(ax, ax) + b_matrix @ h_prev @ b_matrix.T


def scalar_parameters_stationary(a_value: float, b_value: float) -> bool:
    """Project rule for the scalar coefficients, including the shared epsilon."""
    return bool(
        np.isfinite(a_value)
        and np.isfinite(b_value)
        and a_value >= 0.0
        and b_value >= 0.0
        and a_value + b_value < 1.0 - STATIONARITY_EPSILON
    )


def matrix_parameters_stationary(
    a_matrix: NDArray[np.floating],
    b_matrix: NDArray[np.floating],
    form: str,
) -> bool:
    """Diagonal squares, or the full Kronecker spectral radius, stay below 1."""
    if form == "scalar":
        return scalar_parameters_stationary(float(a_matrix[0, 0]), float(b_matrix[0, 0]))
    if form == "diagonal":
        a_diag = np.diag(a_matrix)
        b_diag = np.diag(b_matrix)
        return bool(
            np.all(a_diag >= 0.0)
            and np.all(b_diag >= 0.0)
            and np.all(a_diag**2 + b_diag**2 < 1.0)
        )
    companion = np.kron(a_matrix, a_matrix) + np.kron(b_matrix, b_matrix)
    radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    return bool(np.isfinite(radius) and radius < 1.0)


def _estimate(
    internal: NDArray[np.floating],
    h0: NDArray[np.floating],
    fit_mean: NDArray[np.floating],
    form: str,
    targeting: bool,
    failure_policy: str,
) -> BEKKFitState:
    """L-BFGS-B QMLE. A boundary or failed terminal fit raises."""
    start = _start_vector(internal, form, targeting, failure_policy)
    bounds = _bounds(form, targeting, failure_policy, internal.shape[1])
    result = minimize(
        lambda theta: _objective(theta, internal, h0, form, targeting, failure_policy),
        start,
        method=OPTIMIZER_NAME,
        bounds=bounds,
        options={"maxiter": MAX_ITER},
    )
    if not np.isfinite(result.fun) or result.fun >= _PENALTY / 10.0:
        raise InvalidModelForecastError("BEKK QMLE did not return a finite NLL")
    return _state_from_theta(
        np.asarray(result.x, dtype=float),
        internal,
        h0,
        fit_mean,
        form,
        targeting,
        failure_policy,
        carry_count=0,
        final_nll=float(result.fun),
    )


def _objective(theta, internal, h0, form, targeting, failure_policy) -> float:
    try:
        intercept, a_matrix, b_matrix = _decode(theta, h0, form, targeting, failure_policy)
        if not matrix_parameters_stationary(a_matrix, b_matrix, form):
            return _PENALTY
        return _path_nll(internal, h0, intercept, a_matrix, b_matrix, form)
    except (InvalidModelForecastError, np.linalg.LinAlgError, ValueError):
        return _PENALTY


def _path_nll(internal, h0, intercept, a_matrix, b_matrix, form) -> float:
    """Sum the imported Gaussian NLL. The terminal forecast is not scored."""
    current = np.array(h0, dtype=float, copy=True)
    total = gaussian_nll_cholesky_numpy(current, internal[0])
    for time_index in range(internal.shape[0] - 1):
        current = bekk_transition(
            intercept, a_matrix, b_matrix, internal[time_index], current, form
        )
        total += gaussian_nll_cholesky_numpy(current, internal[time_index + 1])
    # Form the origin forecast so a non-PD terminal step fails inside the objective.
    bekk_transition(intercept, a_matrix, b_matrix, internal[-1], current, form)
    return float(total)


def _state_from_theta(
    theta, internal, h0, fit_mean, form, targeting, failure_policy, carry_count, final_nll
) -> BEKKFitState:
    intercept, a_matrix, b_matrix = _decode(theta, h0, form, targeting, failure_policy)
    if not matrix_parameters_stationary(a_matrix, b_matrix, form):
        raise InvalidModelForecastError("BEKK parameters are outside the stationarity region")
    current = np.array(h0, dtype=float, copy=True)
    for time_index in range(internal.shape[0]):
        current = bekk_transition(
            intercept, a_matrix, b_matrix, internal[time_index], current, form
        )
    _require_pd(current)
    return BEKKFitState(
        fit_mean=np.array(fit_mean, dtype=float, copy=True),
        h0=np.array(h0, dtype=float, copy=True),
        intercept=intercept,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        scalar_a=float(a_matrix[0, 0]) if form == "scalar" else float("nan"),
        scalar_b=float(b_matrix[0, 0]) if form == "scalar" else float("nan"),
        current_h=current,
        form=form,
        targeting=targeting,
        failure_policy=failure_policy,
        carry_count=carry_count,
        n_parameters=int(theta.shape[0]),
        final_nll=final_nll,
    )


def _replay_carried(internal, h0, fit_mean, previous: BEKKFitState) -> BEKKFitState:
    """Keep the last accepted parameters and rebuild the filter on this window."""
    current = np.array(h0, dtype=float, copy=True)
    for time_index in range(internal.shape[0]):
        current = bekk_transition(
            previous.intercept,
            previous.a_matrix,
            previous.b_matrix,
            internal[time_index],
            current,
            previous.form,
        )
    _require_pd(current)
    return BEKKFitState(
        fit_mean=np.array(fit_mean, dtype=float, copy=True),
        h0=np.array(h0, dtype=float, copy=True),
        intercept=np.array(previous.intercept, dtype=float, copy=True),
        a_matrix=np.array(previous.a_matrix, dtype=float, copy=True),
        b_matrix=np.array(previous.b_matrix, dtype=float, copy=True),
        scalar_a=previous.scalar_a,
        scalar_b=previous.scalar_b,
        current_h=current,
        form=previous.form,
        targeting=previous.targeting,
        failure_policy=previous.failure_policy,
        carry_count=previous.carry_count + 1,
        n_parameters=previous.n_parameters,
        final_nll=previous.final_nll,
    )


def _decode(theta, h0, form, targeting, failure_policy):
    """Map an optimizer vector into an intercept and the A and B matrices."""
    n_assets = h0.shape[0]
    cursor = 0
    if targeting:
        a_matrix, b_matrix, cursor = _read_ab(theta, cursor, n_assets, form, failure_policy)
        intercept = _targeted_intercept(h0, a_matrix, b_matrix, form)
    else:
        n_triangle = triangle_parameter_count(n_assets)
        factor = unpack_static_c_numpy(theta[cursor : cursor + n_triangle], n_assets)
        cursor += n_triangle
        intercept = factor @ factor.T
        a_matrix, b_matrix, cursor = _read_ab(theta, cursor, n_assets, form, failure_policy)
    if cursor != theta.shape[0]:
        raise InvalidModelInputError("BEKK parameter vector has the wrong length")
    return intercept, a_matrix, b_matrix


def _read_ab(theta, cursor, n_assets, form, failure_policy):
    if form == "scalar":
        a_value, b_value = _scalar_pair(theta[cursor : cursor + _scalar_width(failure_policy)], failure_policy)
        cursor += _scalar_width(failure_policy)
        return a_value * np.eye(n_assets), b_value * np.eye(n_assets), cursor
    if form == "diagonal":
        if failure_policy == POLICY_INTERIOR_MAP:
            a_diag = np.zeros(n_assets)
            b_diag = np.zeros(n_assets)
            for asset in range(n_assets):
                mapped = softmax_persistence(theta[cursor : cursor + 3])
                a_diag[asset] = mapped.alpha
                b_diag[asset] = mapped.beta
                cursor += 3
        else:
            a_diag = np.asarray(theta[cursor : cursor + n_assets], dtype=float)
            cursor += n_assets
            b_diag = np.asarray(theta[cursor : cursor + n_assets], dtype=float)
            cursor += n_assets
        return np.diag(a_diag), np.diag(b_diag), cursor
    n_full = n_assets * n_assets
    a_matrix = np.asarray(theta[cursor : cursor + n_full], dtype=float).reshape(n_assets, n_assets)
    cursor += n_full
    b_matrix = np.asarray(theta[cursor : cursor + n_full], dtype=float).reshape(n_assets, n_assets)
    cursor += n_full
    return a_matrix, b_matrix, cursor


def _scalar_pair(raw, failure_policy) -> tuple[float, float]:
    if failure_policy == POLICY_INTERIOR_MAP:
        mapped = softmax_persistence(raw)
        return mapped.alpha, mapped.beta
    return float(raw[0]), float(raw[1])


def _scalar_width(failure_policy: str) -> int:
    return 3 if failure_policy == POLICY_INTERIOR_MAP else 2


def _targeted_intercept(h0, a_matrix, b_matrix, form) -> NDArray[np.floating]:
    """Covariance targeting. A non-PD remainder is an estimation failure."""
    if form == "scalar":
        weight = 1.0 - float(a_matrix[0, 0]) - float(b_matrix[0, 0])
        intercept = weight * h0
    else:
        intercept = h0 - a_matrix @ h0 @ a_matrix.T - b_matrix @ h0 @ b_matrix.T
    _require_pd(intercept)
    return intercept


def _start_vector(internal, form, targeting, failure_policy) -> NDArray[np.floating]:
    """Start at the LSTM-BEKK scalar constants. Matrix forms put them on the diagonal."""
    n_assets = internal.shape[1]
    parts: list[NDArray[np.floating]] = []
    if not targeting:
        h0, _method = construct_h0(internal)
        parts.append(packed_static_c_from_h0(h0, W0))
    if form == "scalar":
        if failure_policy == POLICY_INTERIOR_MAP:
            parts.append(np.log(np.array([1.0 - A0 - B0, A0, B0], dtype=float)))
        else:
            parts.append(np.array([A0, B0], dtype=float))
    elif form == "diagonal":
        if failure_policy == POLICY_INTERIOR_MAP:
            logits = np.log(np.array([1.0 - A0 - B0, A0, B0], dtype=float))
            parts.extend(logits for _asset in range(n_assets))
        else:
            parts.append(np.full(n_assets, A0))
            parts.append(np.full(n_assets, B0))
    else:
        a_matrix = np.eye(n_assets) * A0
        b_matrix = np.eye(n_assets) * B0
        parts.append(a_matrix.reshape(-1))
        parts.append(b_matrix.reshape(-1))
    return np.concatenate(parts)


def _bounds(form, targeting, failure_policy, n_assets):
    if failure_policy == POLICY_INTERIOR_MAP or form == "full":
        return None
    width = 0 if targeting else triangle_parameter_count(n_assets)
    free = n_assets if form == "diagonal" else 1
    cap = 1.0 - STATIONARITY_EPSILON
    c_bounds = [(None, None)] * width
    ab_bounds = [(0.0, cap)] * (2 * free)
    return c_bounds + ab_bounds


class LSTMBEKKNoDynamicC(LSTMBEKKCovariance):
    """LSTM-BEKK ablation with the dynamic intercept forced to zero.

    Training settings stay those of the caller. Depth does not enter the
    recursion once the dynamic intercept is zero. This class is an ablation,
    not an eligible family.
    """

    model_name = "lstm_bekk_noc"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._zero_dynamic_c = True


def _require_pd(matrix: NDArray[np.floating]) -> None:
    try:
        np.linalg.cholesky(np.asarray(matrix, dtype=float))
    except np.linalg.LinAlgError as exc:
        raise InvalidModelForecastError("BEKK matrix is not positive definite") from exc
