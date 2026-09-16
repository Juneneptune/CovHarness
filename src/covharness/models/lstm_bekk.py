"""Faithful daily-return LSTM-BEKK one-day-ahead covariance model.

The recursion is Wang, Liu, Tran, and Wang (2025) scalar BEKK plus an
LSTM-generated dynamic intercept. Public forecasts are returned in
caller-native covariance units. Internal recursion and NLL use the
paper percent scale. Architecture details omitted by the paper are
labeled project choices in this module's constants and README.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from covharness.losses.contracts import (
    SYMMETRY_ATOL,
    InvalidCovarianceMatrixError,
    require_symmetric,
)
from covharness.models.base import (
    CovarianceForecast,
    CovarianceModel,
    ModelIdentity,
    as_daily_return_history,
    pack_forecast,
)
from covharness.models.capabilities import (
    FitInput,
    ModelCapabilities,
    RollingCadence,
    UpdateObservable,
)
from covharness.models.exceptions import (
    InvalidModelConfigurationError,
    InvalidModelForecastError,
    InvalidModelInputError,
)

MODEL_NAME = "lstm_bekk"
INTERNAL_RETURN_SCALE = 100.0
COVARIANCE_RESCALE = 10000.0
W0 = 0.05
A0 = 0.05
B0 = 0.90
SWISH_BETA0 = 1.0
MIN_NUM_LAYERS = 3
MAX_NUM_LAYERS = 5
MIN_DROPOUT = 0.1
MAX_DROPOUT = 0.2
OUTPUT_HEAD_BIAS = True
RMSPROP_ALPHA = 0.99
RMSPROP_EPS = 1e-8
RMSPROP_MOMENTUM = 0.0
RMSPROP_CENTERED = False
RMSPROP_WEIGHT_DECAY = 0.0
TORCH_DTYPE = torch.float64
TORCH_DEVICE = torch.device("cpu")
DTYPE_NAME = "float64"
DEVICE_NAME = "cpu"
H0_SAMPLE = "sample_covariance"
H0_DIAGONAL = "sample_variance_diagonal"


@dataclass(frozen=True)
class LSTMBEKKFitState:
    """Auditable fitted LSTM-BEKK filter after one origin-window train."""

    fit_mean: NDArray[np.floating]
    n_observations: int
    n_assets: int
    n_triangle: int
    internal_return_scale: float
    covariance_rescale: float
    h0_method: str
    h0: NDArray[np.floating]
    num_layers: int
    hidden_size: int
    dropout: float
    learning_rate: float
    gradient_clip_norm: float
    max_epochs: int
    optimizer_name: str
    rmsprop_alpha: float
    rmsprop_eps: float
    rmsprop_momentum: float
    weight_decay: float
    full_bptt: bool
    early_stopping: bool
    seed: int
    dtype_name: str
    device_name: str
    packed_static_c: NDArray[np.floating]
    static_c: NDArray[np.floating]
    stationarity_logits: NDArray[np.floating]
    stationarity_w: float
    bekk_a: float
    bekk_b: float
    swish_beta: float
    output_head_bias: bool
    hidden_state: NDArray[np.floating]
    cell_state: NDArray[np.floating]
    current_forecast_internal: NDArray[np.floating]
    last_observed_internal_return: NDArray[np.floating]
    training_loss_history: tuple[float, ...]
    initial_training_nll: float
    final_training_nll: float
    n_likelihood_terms: int


class LSTMBEKKNetwork(nn.Module):
    """Stacked LSTM, triangular head, static C, and stationarity logits."""

    def __init__(self, n_assets: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        n_triangle = triangle_parameter_count(n_assets)
        self.n_assets = n_assets
        self.n_triangle = n_triangle
        # Stacked LSTM. Hidden width equals the asset count.
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=n_assets,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=False,
        )
        # Linear N -> P head. This measurement map is a project completion.
        self.triangle_head = nn.Linear(n_assets, n_triangle, bias=OUTPUT_HEAD_BIAS)
        self.packed_static_c = nn.Parameter(torch.zeros(n_triangle, dtype=TORCH_DTYPE))
        self.stationarity_logits = nn.Parameter(torch.zeros(3, dtype=TORCH_DTYPE))
        self.swish_beta = nn.Parameter(torch.tensor(SWISH_BETA0, dtype=TORCH_DTYPE))
        self.to(device=TORCH_DEVICE, dtype=TORCH_DTYPE)

    def static_c_matrix(self) -> Tensor:
        """Unpack the lower triangle with a softplus diagonal."""
        return unpack_static_c_torch(self.packed_static_c, self.n_assets)

    def stationarity_weights(self) -> tuple[Tensor, Tensor, Tensor]:
        """Map unconstrained logits to the interior simplex (w, a, b)."""
        weights = torch.softmax(self.stationarity_logits, dim=0)
        return weights[0], weights[1], weights[2]


class LSTMBEKKCovariance(CovarianceModel):
    """One-day-ahead faithful LSTM-BEKK on a daily-return window."""

    model_name = MODEL_NAME
    capabilities = ModelCapabilities(
        rolling_cadence=RollingCadence.RECURSIVE_STATE,
        fit_input=FitInput.DAILY_RETURN,
        update_observable=UpdateObservable.DAILY_RETURN,
    )

    def __init__(
        self,
        *,
        seed: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        gradient_clip_norm: float,
        max_epochs: int,
    ) -> None:
        self._seed = _require_seed(seed)
        self._num_layers = _require_num_layers(num_layers)
        self._dropout = _require_dropout(dropout)
        self._learning_rate = _require_positive_float(learning_rate, "learning_rate")
        self._gradient_clip_norm = _require_positive_float(
            gradient_clip_norm, "gradient_clip_norm"
        )
        self._max_epochs = _require_max_epochs(max_epochs)
        self._network: LSTMBEKKNetwork | None = None
        self._fit: LSTMBEKKFitState | None = None

    @property
    def identity(self) -> ModelIdentity:
        configuration: dict[str, object] = {
            "seed": self._seed,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
            "learning_rate": self._learning_rate,
            "gradient_clip_norm": self._gradient_clip_norm,
            "max_epochs": self._max_epochs,
            "hidden_size": "n_assets",
            "internal_return_scale": INTERNAL_RETURN_SCALE,
            "covariance_rescale": COVARIANCE_RESCALE,
            "optimizer": "RMSprop",
            "full_bptt": True,
            "early_stopping": False,
            "dtype": DTYPE_NAME,
            "device": DEVICE_NAME,
            "output_head_bias": OUTPUT_HEAD_BIAS,
        }
        if self._fit is not None:
            configuration.update(
                {
                    "n_observations": self._fit.n_observations,
                    "n_assets": self._fit.n_assets,
                    "n_triangle": self._fit.n_triangle,
                    "hidden_size": self._fit.hidden_size,
                    "h0_method": self._fit.h0_method,
                    "bekk_a": self._fit.bekk_a,
                    "bekk_b": self._fit.bekk_b,
                    "stationarity_w": self._fit.stationarity_w,
                    "swish_beta": self._fit.swish_beta,
                    "final_training_nll": self._fit.final_training_nll,
                }
            )
        return ModelIdentity(name=self.model_name, configuration=configuration)

    def fit(self, returns: ArrayLike) -> LSTMBEKKCovariance:
        """Train on one origin window and store the one-step filter state."""
        history = _lstm_bekk_return_window(returns)
        n_observations, n_assets = history.shape
        # Center in caller units, then apply the paper percent scale.
        fit_mean, _native_residuals, internal = prepare_internal_returns(history)
        h0, h0_method = construct_h0(internal)
        packed_init = packed_static_c_from_h0(h0, W0)
        apply_cpu_seeds(self._seed)
        network = LSTMBEKKNetwork(n_assets, self._num_layers, self._dropout)
        initialize_lstm_bekk_network(network, packed_init)
        x_tensor = torch.tensor(internal, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        h0_tensor = torch.tensor(h0, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        optimizer = torch.optim.RMSprop(
            network.parameters(),
            lr=self._learning_rate,
            alpha=RMSPROP_ALPHA,
            eps=RMSPROP_EPS,
            momentum=RMSPROP_MOMENTUM,
            centered=RMSPROP_CENTERED,
            weight_decay=RMSPROP_WEIGHT_DECAY,
        )
        input_builder = self.build_lstm_input
        network.eval()
        with torch.no_grad():
            initial_nll_tensor = sequence_training_nll(
                network, x_tensor, h0_tensor, input_builder=input_builder
            )
        _require_finite_training_nll(initial_nll_tensor)
        initial_nll = float(initial_nll_tensor.cpu().item())
        network.train()
        history_losses: list[float] = []
        # Full-sequence BPTT. One epoch is one pass over the whole window.
        for _epoch in range(self._max_epochs):
            optimizer.zero_grad(set_to_none=True)
            nll = sequence_training_nll(
                network, x_tensor, h0_tensor, input_builder=input_builder
            )
            _require_finite_training_nll(nll)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=self._gradient_clip_norm
            )
            optimizer.step()
            history_losses.append(float(nll.detach().cpu().item()))
        # Replay in eval mode so dropout is off for the stored filter.
        network.eval()
        with torch.no_grad():
            replay = run_bekk_sequence(
                network, x_tensor, h0_tensor, input_builder=input_builder
            )
            final_nll_tensor = replay.nll
        _require_finite_training_nll(final_nll_tensor)
        final_nll = float(final_nll_tensor.cpu().item())
        forecast_internal = _numpy_matrix(replay.forecast_internal)
        _reject_invalid_forecast(forecast_internal, self.model_name)
        w_val, a_val, b_val = _numpy_stationarity(network)
        self._network = network
        self._fit = LSTMBEKKFitState(
            fit_mean=np.array(fit_mean, dtype=float, copy=True),
            n_observations=n_observations,
            n_assets=n_assets,
            n_triangle=triangle_parameter_count(n_assets),
            internal_return_scale=INTERNAL_RETURN_SCALE,
            covariance_rescale=COVARIANCE_RESCALE,
            h0_method=h0_method,
            h0=np.array(h0, dtype=float, copy=True),
            num_layers=self._num_layers,
            hidden_size=n_assets,
            dropout=self._dropout,
            learning_rate=self._learning_rate,
            gradient_clip_norm=self._gradient_clip_norm,
            max_epochs=self._max_epochs,
            optimizer_name="RMSprop",
            rmsprop_alpha=RMSPROP_ALPHA,
            rmsprop_eps=RMSPROP_EPS,
            rmsprop_momentum=RMSPROP_MOMENTUM,
            weight_decay=RMSPROP_WEIGHT_DECAY,
            full_bptt=True,
            early_stopping=False,
            seed=self._seed,
            dtype_name=DTYPE_NAME,
            device_name=DEVICE_NAME,
            packed_static_c=_numpy_vector(network.packed_static_c),
            static_c=_numpy_matrix(network.static_c_matrix()),
            stationarity_logits=_numpy_vector(network.stationarity_logits),
            stationarity_w=w_val,
            bekk_a=a_val,
            bekk_b=b_val,
            swish_beta=float(network.swish_beta.detach().cpu().item()),
            output_head_bias=OUTPUT_HEAD_BIAS,
            hidden_state=_numpy_rnn_state(replay.hidden),
            cell_state=_numpy_rnn_state(replay.cell),
            current_forecast_internal=forecast_internal,
            last_observed_internal_return=np.array(internal[-1], dtype=float, copy=True),
            training_loss_history=tuple(history_losses),
            initial_training_nll=initial_nll,
            final_training_nll=final_nll,
            n_likelihood_terms=n_observations,
        )
        return self

    def forecast(self) -> CovarianceForecast:
        """Return ``H_{t+1|t}`` in native units. The filter is not mutated."""
        state = _require_lstm_fit(self._fit, self.model_name)
        native = state.current_forecast_internal / state.covariance_rescale
        _reject_invalid_forecast(native, self.model_name)
        return pack_forecast(native, self.identity)

    def update(self, new_return: ArrayLike) -> LSTMBEKKCovariance:
        """Advance the filter by one observed return. Parameters stay frozen."""
        state = _require_lstm_fit(self._fit, self.model_name)
        network = _require_network(self._network, self.model_name)
        observed = _as_new_return(new_return, state.n_assets)
        # Center with the frozen fit-window mean and apply the same percent scale.
        x_new = INTERNAL_RETURN_SCALE * (observed - state.fit_mean)
        network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_new, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
            hidden = _torch_rnn_state(state.hidden_state)
            cell = _torch_rnn_state(state.cell_state)
            h_current = torch.tensor(
                state.current_forecast_internal, dtype=TORCH_DTYPE, device=TORCH_DEVICE
            )
            next_state = advance_one_observation(
                network,
                x_tensor,
                hidden,
                cell,
                h_current,
                input_builder=self.build_lstm_input,
            )
        forecast_internal = _numpy_matrix(next_state.forecast_internal)
        _reject_invalid_forecast(forecast_internal, self.model_name)
        self._fit = replace(
            state,
            hidden_state=_numpy_rnn_state(next_state.hidden),
            cell_state=_numpy_rnn_state(next_state.cell),
            current_forecast_internal=forecast_internal,
            last_observed_internal_return=np.array(x_new, dtype=float, copy=True),
        )
        return self

    def build_lstm_input(self, internal_return: Tensor) -> Tensor:
        """Map one internal residual into the LSTM input vector.

        Faithful LSTM-BEKK uses the centered scaled return only. A later
        RC model may concatenate origin-measurable realized features here
        without changing the BEKK recursion, NLL, or forecast contract.
        """
        return lstm_step_input(internal_return)


@dataclass(frozen=True)
class SequencePath:
    """Scored covariances, terminal forecast, and recurrent state."""

    nll: Tensor
    scored_covariances: tuple[Tensor, ...]
    forecast_internal: Tensor
    hidden: Tensor
    cell: Tensor


def triangle_parameter_count(n_assets: int) -> int:
    """Return ``N(N+1)/2`` packed lower-triangle parameters."""
    return n_assets * (n_assets + 1) // 2


def lower_triangle_indices(n_assets: int) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Row-major lower-triangle indices, including the diagonal."""
    rows, cols = np.tril_indices(n_assets)
    return rows, cols


def lstm_step_input(internal_return: Tensor) -> Tensor:
    """Faithful LSTM input is the internal residual. No RCov is appended."""
    return internal_return


def inverse_softplus(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Invert ``softplus`` for strictly positive targets."""
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.isfinite(array).all():
        raise InvalidModelInputError("inverse softplus requires finite strictly positive values")
    return array + np.log(-np.expm1(-array))


def inverse_softplus_torch(values: Tensor) -> Tensor:
    """Torch inverse of ``softplus`` for strictly positive targets."""
    return values + torch.log(-torch.expm1(-values))


def swish(values: Tensor, beta: Tensor) -> Tensor:
    """Swish ``x * sigmoid(beta * x)``. The output may be negative."""
    return values * torch.sigmoid(beta * values)


def unpack_static_c_torch(packed: Tensor, n_assets: int) -> Tensor:
    """Rebuild lower-triangular C with a softplus diagonal."""
    rows, cols = lower_triangle_indices(n_assets)
    matrix = packed.new_zeros((n_assets, n_assets))
    filled = packed
    matrix = matrix.index_put((torch.as_tensor(rows), torch.as_tensor(cols)), filled)
    diagonal = torch.diag(matrix)
    mapped = torch.nn.functional.softplus(diagonal)
    if torch.any(~torch.isfinite(mapped)) or torch.any(mapped <= 0.0):
        raise InvalidModelForecastError("static C diagonal must be finite and strictly positive")
    return matrix - torch.diag(diagonal) + torch.diag(mapped)


def unpack_static_c_numpy(packed: NDArray[np.floating], n_assets: int) -> NDArray[np.floating]:
    """NumPy reconstruction of lower-triangular C with a softplus diagonal."""
    rows, cols = lower_triangle_indices(n_assets)
    matrix = np.zeros((n_assets, n_assets), dtype=float)
    matrix[rows, cols] = np.asarray(packed, dtype=float)
    diagonal = np.diag(matrix).copy()
    mapped = np.log1p(np.exp(-np.abs(diagonal))) + np.maximum(diagonal, 0.0)
    if np.any(~np.isfinite(mapped)) or np.any(mapped <= 0.0):
        raise InvalidModelForecastError("static C diagonal must be finite and strictly positive")
    np.fill_diagonal(matrix, mapped)
    return matrix


def pack_lower_triangle(matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """Pack the lower triangle of an ``(N, N)`` matrix, including the diagonal."""
    rows, cols = lower_triangle_indices(matrix.shape[0])
    return np.asarray(matrix, dtype=float)[rows, cols].copy()


def unpack_lower_triangle(packed: NDArray[np.floating], n_assets: int) -> NDArray[np.floating]:
    """Unpack a length-``P`` vector into a strictly lower-plus-diagonal matrix."""
    rows, cols = lower_triangle_indices(n_assets)
    matrix = np.zeros((n_assets, n_assets), dtype=float)
    matrix[rows, cols] = np.asarray(packed, dtype=float)
    return matrix


def pack_dynamic_triangle(raw_triangle: Tensor, n_assets: int, beta: Tensor) -> Tensor:
    """Reshape a length-``P`` vector and apply Swish only on the diagonal."""
    rows, cols = lower_triangle_indices(n_assets)
    matrix = raw_triangle.new_zeros((n_assets, n_assets))
    matrix = matrix.index_put(
        (torch.as_tensor(rows, device=raw_triangle.device), torch.as_tensor(cols, device=raw_triangle.device)),
        raw_triangle,
    )
    diagonal = torch.diag(matrix)
    swished = swish(diagonal, beta)
    return matrix - torch.diag(diagonal) + torch.diag(swished)


def bekk_recursion(
    static_c: NDArray[np.floating],
    dynamic_c: NDArray[np.floating],
    a: float,
    b: float,
    shock: NDArray[np.floating],
    h_prev: NDArray[np.floating],
) -> NDArray[np.floating]:
    """NumPy BEKK step ``CC' + Ct Ct' + a xx' + b H_prev``."""
    shock = np.asarray(shock, dtype=float)
    return (
        static_c @ static_c.T
        + dynamic_c @ dynamic_c.T
        + a * np.outer(shock, shock)
        + b * h_prev
    )


def bekk_recursion_torch(
    static_c: Tensor,
    dynamic_c: Tensor,
    a: Tensor,
    b: Tensor,
    shock: Tensor,
    h_prev: Tensor,
) -> Tensor:
    """Differentiable BEKK step. There is no detach in this recursion."""
    return (
        static_c @ static_c.T
        + dynamic_c @ dynamic_c.T
        + a * torch.outer(shock, shock)
        + b * h_prev
    )


def gaussian_nll_cholesky_torch(covariance: Tensor, residual: Tensor) -> Tensor:
    """One-observation Gaussian NLL without ``N log(2pi)`` and without ``inv``."""
    try:
        factor = torch.linalg.cholesky(covariance)
    except RuntimeError as exc:
        raise InvalidModelForecastError("LSTM-BEKK Cholesky of H_t failed") from exc
    logdet = 2.0 * torch.sum(torch.log(torch.diag(factor)))
    solved = torch.linalg.solve_triangular(factor, residual.unsqueeze(-1), upper=False)
    quadratic = torch.sum(solved * solved)
    return 0.5 * (logdet + quadratic)


def gaussian_nll_cholesky_numpy(
    covariance: NDArray[np.floating], residual: NDArray[np.floating]
) -> float:
    """NumPy Cholesky NLL matching the training contribution."""
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise InvalidModelForecastError("LSTM-BEKK Cholesky of H_t failed") from exc
    logdet = 2.0 * float(np.sum(np.log(np.diag(factor))))
    solved = np.linalg.solve(factor, residual)
    quadratic = float(np.dot(solved, solved))
    return 0.5 * (logdet + quadratic)


def stationarity_logits_from_probabilities(
    w: float = W0, a: float = A0, b: float = B0
) -> NDArray[np.floating]:
    """Logits whose softmax recovers ``(w, a, b)`` up to floating-point error."""
    probabilities = np.asarray([w, a, b], dtype=float)
    if np.any(probabilities <= 0.0) or not np.isfinite(probabilities).all():
        raise InvalidModelConfigurationError("stationarity probabilities must be finite and positive")
    if not np.isclose(float(np.sum(probabilities)), 1.0):
        raise InvalidModelConfigurationError("stationarity probabilities must sum to 1")
    return np.log(probabilities)


def prepare_internal_returns(
    returns: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Demean the window and scale residuals by 100. Inputs are not mutated."""
    fit_mean = np.mean(returns, axis=0)
    native_residuals = returns - fit_mean
    sample_var = np.sum(native_residuals * native_residuals, axis=0) / (returns.shape[0] - 1)
    column_span = np.max(native_residuals, axis=0) - np.min(native_residuals, axis=0)
    if (
        np.any(~np.isfinite(sample_var))
        or np.any(sample_var <= 0.0)
        or np.any(column_span == 0.0)
    ):
        raise InvalidModelInputError(
            "LSTM-BEKK requires a finite strictly positive centered variance for every asset"
        )
    internal = INTERNAL_RETURN_SCALE * native_residuals
    return fit_mean, native_residuals, internal


def construct_h0(internal_returns: NDArray[np.floating]) -> tuple[NDArray[np.floating], str]:
    """Build a frozen PD ``H_0`` from the internal sample covariance."""
    n_observations, n_assets = internal_returns.shape
    sample = internal_returns.T @ internal_returns / (n_observations - 1)
    if not np.isfinite(sample).all():
        raise InvalidModelInputError("LSTM-BEKK H0 sample covariance is nonfinite")
    try:
        require_symmetric(sample, "LSTM-BEKK H0 sample covariance", atol=SYMMETRY_ATOL)
        np.linalg.cholesky(sample)
        return np.array(sample, dtype=float, copy=True), H0_SAMPLE
    except (InvalidCovarianceMatrixError, np.linalg.LinAlgError):
        diagonal = np.diag(sample).astype(float, copy=True)
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            raise InvalidModelInputError(
                "LSTM-BEKK H0 diagonal fallback requires finite strictly positive variances"
            )
        return np.diag(diagonal), H0_DIAGONAL


def packed_static_c_from_h0(h0: NDArray[np.floating], intercept_weight: float = W0) -> NDArray[np.floating]:
    """Initialize packed C so ``C C' = intercept_weight * H0``."""
    target = intercept_weight * h0
    try:
        cholesky = np.linalg.cholesky(target)
    except np.linalg.LinAlgError as exc:
        raise InvalidModelInputError("LSTM-BEKK static C initialization Cholesky failed") from exc
    rows, cols = lower_triangle_indices(h0.shape[0])
    packed = np.zeros(triangle_parameter_count(h0.shape[0]), dtype=float)
    diagonal_mask = rows == cols
    packed[diagonal_mask] = inverse_softplus(cholesky[rows[diagonal_mask], cols[diagonal_mask]])
    packed[~diagonal_mask] = cholesky[rows[~diagonal_mask], cols[~diagonal_mask]]
    return packed


def apply_cpu_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch. Execution stays on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def initialize_lstm_bekk_network(network: LSTMBEKKNetwork, packed_static_c: NDArray[np.floating]) -> None:
    """Replace PyTorch defaults with the frozen project initialization."""
    for name, parameter in network.lstm.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(parameter)
        elif "bias" in name:
            nn.init.zeros_(parameter)
    nn.init.xavier_uniform_(network.triangle_head.weight)
    if network.triangle_head.bias is not None:
        nn.init.zeros_(network.triangle_head.bias)
    with torch.no_grad():
        network.packed_static_c.copy_(
            torch.tensor(packed_static_c, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        )
        network.stationarity_logits.copy_(
            torch.tensor(stationarity_logits_from_probabilities(), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        )
        network.swish_beta.fill_(SWISH_BETA0)


def zero_recurrent_state(n_layers: int, n_assets: int) -> tuple[Tensor, Tensor]:
    """Return zero hidden and cell states with batch dimension 1."""
    shape = (n_layers, 1, n_assets)
    hidden = torch.zeros(shape, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    cell = torch.zeros(shape, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    return hidden, cell


def sequence_training_nll(
    network: LSTMBEKKNetwork,
    internal_returns: Tensor,
    h0: Tensor,
    input_builder: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Sum Gaussian NLL over all T observations. The forecast step is excluded."""
    path = run_bekk_sequence(
        network, internal_returns, h0, input_builder=input_builder
    )
    return path.nll


def run_bekk_sequence(
    network: LSTMBEKKNetwork,
    internal_returns: Tensor,
    h0: Tensor,
    input_builder: Callable[[Tensor], Tensor] | None = None,
) -> SequencePath:
    """Score ``x_0`` under ``H_0``, then recurse through the window and form ``H_T``."""
    builder = input_builder if input_builder is not None else lstm_step_input
    n_observations = internal_returns.shape[0]
    hidden, cell = zero_recurrent_state(network.lstm.num_layers, network.n_assets)
    static_c = network.static_c_matrix()
    _w, bekk_a, bekk_b = network.stationarity_weights()
    del _w
    current_h = h0
    nll = gaussian_nll_cholesky_torch(current_h, internal_returns[0])
    scored = [current_h]
    # Score remaining observations after LSTM-updated intercepts.
    for time_index in range(n_observations - 1):
        shock = builder(internal_returns[time_index])
        hidden, cell, dynamic_c = _lstm_dynamic_c(network, shock, hidden, cell)
        current_h = bekk_recursion_torch(static_c, dynamic_c, bekk_a, bekk_b, shock, current_h)
        nll = nll + gaussian_nll_cholesky_torch(current_h, internal_returns[time_index + 1])
        scored.append(current_h)
    # Process the last observed return to form the one-step forecast, not the NLL.
    last_shock = builder(internal_returns[-1])
    hidden, cell, forecast_c = _lstm_dynamic_c(network, last_shock, hidden, cell)
    forecast_h = bekk_recursion_torch(static_c, forecast_c, bekk_a, bekk_b, last_shock, current_h)
    return SequencePath(
        nll=nll,
        scored_covariances=tuple(scored),
        forecast_internal=forecast_h,
        hidden=hidden,
        cell=cell,
    )


def advance_one_observation(
    network: LSTMBEKKNetwork,
    internal_return: Tensor,
    hidden: Tensor,
    cell: Tensor,
    current_forecast: Tensor,
    input_builder: Callable[[Tensor], Tensor] | None = None,
) -> SequencePath:
    """Consume one new internal residual and form the next one-step covariance."""
    builder = input_builder if input_builder is not None else lstm_step_input
    static_c = network.static_c_matrix()
    _w, bekk_a, bekk_b = network.stationarity_weights()
    del _w
    shock = builder(internal_return)
    hidden, cell, dynamic_c = _lstm_dynamic_c(network, shock, hidden, cell)
    next_h = bekk_recursion_torch(static_c, dynamic_c, bekk_a, bekk_b, shock, current_forecast)
    return SequencePath(
        nll=current_forecast.new_zeros(()),
        scored_covariances=(current_forecast,),
        forecast_internal=next_h,
        hidden=hidden,
        cell=cell,
    )


def _lstm_dynamic_c(
    network: LSTMBEKKNetwork,
    shock: Tensor,
    hidden: Tensor,
    cell: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Advance the stacked LSTM one step and map the hidden state to ``C_t``."""
    lstm_input = shock.view(1, 1, network.n_assets)
    output, (hidden, cell) = network.lstm(lstm_input, (hidden, cell))
    hidden_vector = output.reshape(network.n_assets)
    raw_triangle = network.triangle_head(hidden_vector)
    dynamic_c = pack_dynamic_triangle(raw_triangle, network.n_assets, network.swish_beta)
    return hidden, cell, dynamic_c


def _lstm_bekk_return_window(returns: ArrayLike) -> NDArray[np.floating]:
    """Copy a finite ``(T, N)`` window and require at least two assets."""
    history = as_daily_return_history(returns)
    if history.shape[1] < 2:
        raise InvalidModelInputError(
            f"LSTM-BEKK requires at least two assets; got N={history.shape[1]}"
        )
    return history


def _as_new_return(new_return: ArrayLike, n_assets: int) -> NDArray[np.floating]:
    """Copy a finite length-``N`` update vector."""
    array = np.array(new_return, dtype=float, copy=True)
    if array.ndim != 1 or array.shape[0] != n_assets:
        raise InvalidModelInputError(
            f"LSTM-BEKK update requires shape (N,) with N={n_assets}; got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise InvalidModelInputError("LSTM-BEKK update return must be finite")
    return array


def _require_seed(seed: int) -> int:
    """Require an explicit integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InvalidModelConfigurationError("LSTM-BEKK seed must be an int")
    return int(seed)


def _require_num_layers(num_layers: int) -> int:
    """Require stacked LSTM depth in the paper range 3 to 5."""
    if isinstance(num_layers, bool) or not isinstance(num_layers, int):
        raise InvalidModelConfigurationError("LSTM-BEKK num_layers must be an int")
    if num_layers < MIN_NUM_LAYERS or num_layers > MAX_NUM_LAYERS:
        raise InvalidModelConfigurationError(
            f"LSTM-BEKK num_layers must be in [{MIN_NUM_LAYERS}, {MAX_NUM_LAYERS}]; got {num_layers}"
        )
    return num_layers


def _require_dropout(dropout: float) -> float:
    """Require dropout in the paper interval [0.1, 0.2]."""
    value = float(dropout)
    if not np.isfinite(value) or value < MIN_DROPOUT or value > MAX_DROPOUT:
        raise InvalidModelConfigurationError(
            f"LSTM-BEKK dropout must be in [{MIN_DROPOUT}, {MAX_DROPOUT}]; got {dropout}"
        )
    return value


def _require_positive_float(value: float, name: str) -> float:
    """Require a finite strictly positive hyperparameter."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise InvalidModelConfigurationError(f"LSTM-BEKK {name} must be finite and strictly positive")
    return number


def _require_max_epochs(max_epochs: int) -> int:
    """Require a fixed positive epoch count. There is no early stopping."""
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int):
        raise InvalidModelConfigurationError("LSTM-BEKK max_epochs must be an int")
    if max_epochs < 1:
        raise InvalidModelConfigurationError("LSTM-BEKK max_epochs must be at least 1")
    return max_epochs


def _require_lstm_fit(state: LSTMBEKKFitState | None, name: str) -> LSTMBEKKFitState:
    """Reject a filter read issued before ``fit``."""
    if state is None:
        raise InvalidModelInputError(
            f"{name} has no fitted origin-window state. Call fit before forecast or update."
        )
    return state


def _require_network(network: LSTMBEKKNetwork | None, name: str) -> LSTMBEKKNetwork:
    """Reject an update issued without a live network."""
    if network is None:
        raise InvalidModelInputError(
            f"{name} has no trained network. Call fit before forecast or update."
        )
    return network


def _require_finite_training_nll(nll: Tensor) -> None:
    """Reject a nonfinite training objective rather than stepping on NaNs."""
    if nll.ndim != 0 or not torch.isfinite(nll):
        raise InvalidModelForecastError("LSTM-BEKK training NLL is nonfinite")


def _reject_invalid_forecast(matrix: NDArray[np.floating], name: str) -> None:
    """Reject a nonfinite, asymmetric, or non-strictly-PD forecast. No repair."""
    if not np.isfinite(matrix).all():
        raise InvalidModelForecastError(f"{name} forecast is nonfinite")
    try:
        require_symmetric(matrix, f"{name} forecast", atol=SYMMETRY_ATOL)
    except InvalidCovarianceMatrixError as exc:
        raise InvalidModelForecastError(str(exc)) from exc
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise InvalidModelForecastError(
            f"{name} forecast is not strictly positive definite"
        ) from exc


def _numpy_matrix(tensor: Tensor) -> NDArray[np.floating]:
    """Detach a matrix copy onto CPU float64 NumPy memory."""
    return np.array(tensor.detach().cpu().numpy(), dtype=float, copy=True)


def _numpy_vector(tensor: Tensor) -> NDArray[np.floating]:
    """Detach a vector copy onto CPU float64 NumPy memory."""
    return np.array(tensor.detach().cpu().numpy(), dtype=float, copy=True)


def _numpy_stationarity(network: LSTMBEKKNetwork) -> tuple[float, float, float]:
    """Read ``(w, a, b)`` from the live network."""
    weights = torch.softmax(network.stationarity_logits.detach(), dim=0).cpu().numpy()
    return float(weights[0]), float(weights[1]), float(weights[2])


def _numpy_rnn_state(tensor: Tensor) -> NDArray[np.floating]:
    """Store LSTM state as ``(num_layers, N)`` without the batch axis."""
    array = np.array(tensor.detach().cpu().numpy(), dtype=float, copy=True)
    return array[:, 0, :]


def _torch_rnn_state(array: NDArray[np.floating]) -> Tensor:
    """Restore LSTM state with batch dimension 1."""
    return torch.tensor(array[:, None, :], dtype=TORCH_DTYPE, device=TORCH_DEVICE)
