"""Information-matched LSTM-BEKK variants. The faithful class is unchanged.

V10 appends origin-day realized features to the LSTM input. V01 trains on
reduced QLIKE against realized covariance. V11 does both. The BEKK shock
remains the internal daily return. V01 and V11 keep the LSTM22 learning
rate and epoch count.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from covharness.diagnostics.epps import covariance_to_correlation
from covharness.losses.qlike import reduced_qlike_loss
from covharness.models.base import ModelIdentity
from covharness.models.capabilities import (
    FitInput,
    ModelCapabilities,
    RollingCadence,
    UpdateObservable,
)
from covharness.models.exceptions import InvalidModelInputError
from covharness.models.har_drd import pair_vector_from_correlation
from covharness.models.lstm_bekk import (
    ARCHITECTURE_RANGE_POST_SCREEN_D1,
    COVARIANCE_RESCALE,
    INTERNAL_RETURN_SCALE,
    DEVICE_NAME,
    DTYPE_NAME,
    LSTMBEKKCovariance,
    LSTMBEKKFitState,
    LSTMBEKKNetwork,
    OUTPUT_HEAD_BIAS,
    RMSPROP_ALPHA,
    RMSPROP_CENTERED,
    RMSPROP_EPS,
    RMSPROP_MOMENTUM,
    RMSPROP_WEIGHT_DECAY,
    TORCH_DEVICE,
    TORCH_DTYPE,
    W0,
    SequencePath,
    _as_new_return,
    _lstm_bekk_return_window,
    _lstm_dynamic_c,
    _numpy_matrix,
    _numpy_rnn_state,
    _numpy_stationarity,
    _numpy_vector,
    _reject_invalid_forecast,
    _require_finite_training_nll,
    _require_lstm_fit,
    _require_network,
    _torch_rnn_state,
    apply_cpu_seeds,
    bekk_recursion_torch,
    construct_h0,
    gaussian_nll_cholesky_torch,
    initialize_lstm_bekk_network,
    packed_static_c_from_h0,
    prepare_internal_returns,
    triangle_parameter_count,
    zero_recurrent_state,
)

VARIANT_CAPABILITIES = ModelCapabilities(
    rolling_cadence=RollingCadence.RECURSIVE_STATE,
    fit_input=FitInput.DAILY_RETURN_REALIZED_CONTEXT,
    update_observable=UpdateObservable.DAILY_RETURN_AND_FEATURES,
)


def realized_feature_matrix(
    rs_positive: NDArray[np.floating],
    rs_negative: NDArray[np.floating],
    realized_covariance: NDArray[np.floating],
    realized_quarticity: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Build ``[log RS+, log RS-, vech correlation, log RQ]`` for each day.

    Row ``t`` uses only day ``t``. A nonpositive semivariance or quarticity
    is an input failure. Nothing is floored.
    """
    positive = np.asarray(rs_positive, dtype=float)
    negative = np.asarray(rs_negative, dtype=float)
    rcov = np.asarray(realized_covariance, dtype=float)
    quarticity = np.asarray(realized_quarticity, dtype=float)
    # Reject a nonpositive semivariance before the logarithm.
    if np.any(positive <= 0.0) or np.any(negative <= 0.0):
        raise InvalidModelInputError("a nonpositive semivariance is an input failure")
    if np.any(quarticity <= 0.0) or not np.isfinite(quarticity).all():
        raise InvalidModelInputError("a nonpositive realized quarticity is an input failure")
    correlations = np.stack(
        [
            pair_vector_from_correlation(covariance_to_correlation(rcov[index]))
            for index in range(rcov.shape[0])
        ],
        axis=0,
    )
    features = np.concatenate(
        [np.log(positive), np.log(negative), correlations, np.log(quarticity)],
        axis=1,
    )
    if not np.isfinite(features).all():
        raise InvalidModelInputError("realized features must be finite")
    return features


def fit_feature_scaler(
    feature_window: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Mean and population standard deviation of one estimation window."""
    window = np.asarray(feature_window, dtype=float)
    mean = np.mean(window, axis=0)
    scale = np.std(window, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale == 0.0):
        raise InvalidModelInputError("feature scaler is undefined on this window")
    return mean, scale


def apply_feature_scaler(
    features: NDArray[np.floating],
    mean: NDArray[np.floating],
    scale: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply a frozen window scaler. The scaler is not refit here."""
    return (np.asarray(features, dtype=float) - mean) / scale


def reduced_qlike_torch(forecast: Tensor, proxy: Tensor) -> Tensor:
    """Reduced QLIKE ``logdet(H) + trace(H^{-1} S)`` with one Cholesky."""
    factor = torch.linalg.cholesky(forecast)
    logdet = 2.0 * torch.sum(torch.log(torch.diag(factor)))
    lower = torch.linalg.solve_triangular(factor, proxy, upper=False)
    solved = torch.linalg.solve_triangular(factor.transpose(0, 1), lower, upper=True)
    return logdet + torch.trace(solved)


def _score_observation(
    forecast: Tensor,
    residual: Tensor,
    proxy: Tensor | None,
) -> Tensor:
    """Gaussian NLL, or reduced QLIKE when a same-day proxy is supplied."""
    if proxy is None:
        return gaussian_nll_cholesky_torch(forecast, residual)
    return reduced_qlike_torch(forecast, proxy)


def _step_shock(
    internal_return: Tensor,
    feature_row: Tensor | None,
) -> Tensor:
    """Return the BEKK shock, with standardized features appended for the LSTM."""
    if feature_row is None:
        return internal_return
    return torch.cat((internal_return, feature_row), dim=0)


def run_information_sequence(
    network: LSTMBEKKNetwork,
    internal_returns: Tensor,
    h0: Tensor,
    feature_rows: Tensor | None,
    proxy_rows: Tensor | None,
) -> SequencePath:
    """Score the window. Features and the QLIKE proxy are optional."""
    n_observations = int(internal_returns.shape[0])
    n_assets = network.n_assets
    hidden, cell = zero_recurrent_state(network.lstm.num_layers, n_assets)
    static_c = network.static_c_matrix()
    _weight, bekk_a, bekk_b = network.stationarity_weights()
    del _weight
    current_h = h0
    first_proxy = None if proxy_rows is None else proxy_rows[0]
    objective = _score_observation(current_h, internal_returns[0], first_proxy)
    # Advance after each scored residual. The last step forms H_{T}.
    for time_index in range(n_observations - 1):
        feature_row = None if feature_rows is None else feature_rows[time_index]
        shock = _step_shock(internal_returns[time_index], feature_row)
        return_shock = shock if shock.shape[0] == n_assets else shock[:n_assets]
        hidden, cell, dynamic_c = _lstm_dynamic_c(network, shock, hidden, cell)
        current_h = bekk_recursion_torch(
            static_c, dynamic_c, bekk_a, bekk_b, return_shock, current_h
        )
        proxy = None if proxy_rows is None else proxy_rows[time_index + 1]
        objective = objective + _score_observation(
            current_h, internal_returns[time_index + 1], proxy
        )
    last_feature = None if feature_rows is None else feature_rows[-1]
    last_shock = _step_shock(internal_returns[-1], last_feature)
    last_return = last_shock if last_shock.shape[0] == n_assets else last_shock[:n_assets]
    hidden, cell, forecast_c = _lstm_dynamic_c(network, last_shock, hidden, cell)
    forecast_h = bekk_recursion_torch(
        static_c, forecast_c, bekk_a, bekk_b, last_return, current_h
    )
    return SequencePath(
        nll=objective,
        scored_covariances=(),
        forecast_internal=forecast_h,
        hidden=hidden,
        cell=cell,
    )


class InformationMatchedLSTMBEKK(LSTMBEKKCovariance):
    """LSTM22 settings with an optional realized input and an optional QLIKE loss."""

    capabilities = VARIANT_CAPABILITIES
    variant_id = ""
    uses_realized_input = False
    uses_qlike = False

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._feature_mean: NDArray[np.floating] | None = None
        self._feature_scale: NDArray[np.floating] | None = None

    @property
    def identity(self) -> ModelIdentity:
        base = super().identity
        configuration = dict(base.configuration)
        configuration["variant"] = self.variant_id
        configuration["realized_input"] = self.uses_realized_input
        configuration["training_objective"] = (
            "reduced_qlike" if self.uses_qlike else "gaussian_nll"
        )
        configuration["architecture_range"] = ARCHITECTURE_RANGE_POST_SCREEN_D1
        return ModelIdentity(name=self.model_name, configuration=configuration)

    def fit(
        self,
        returns: ArrayLike,
        features: ArrayLike,
        proxy: ArrayLike,
    ) -> InformationMatchedLSTMBEKK:
        """Train on one origin window. The scaler uses that window only."""
        history = _lstm_bekk_return_window(returns)
        n_observations, n_assets = history.shape
        feature_rows, proxy_rows = self._window_tensors(history, features, proxy)
        fit_mean, _native, internal = prepare_internal_returns(history)
        h0, h0_method = construct_h0(internal)
        packed_init = packed_static_c_from_h0(h0, W0)
        apply_cpu_seeds(self._seed)
        input_size = n_assets if feature_rows is None else int(feature_rows.shape[1] + n_assets)
        network = LSTMBEKKNetwork(
            n_assets, self._num_layers, self._dropout, input_size=input_size
        )
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
        network.eval()
        with torch.no_grad():
            initial = run_information_sequence(
                network, x_tensor, h0_tensor, feature_rows, proxy_rows
            ).nll
        _require_finite_training_nll(initial)
        initial_loss = float(initial.cpu().item())
        network.train()
        history_losses: list[float] = []
        # Full-sequence BPTT for the declared objective. The epoch count is not retuned.
        for _epoch in range(self._max_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = run_information_sequence(
                network, x_tensor, h0_tensor, feature_rows, proxy_rows
            ).nll
            _require_finite_training_nll(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=self._gradient_clip_norm
            )
            optimizer.step()
            history_losses.append(float(loss.detach().cpu().item()))
        network.eval()
        with torch.no_grad():
            replay = run_information_sequence(
                network, x_tensor, h0_tensor, feature_rows, proxy_rows
            )
        _require_finite_training_nll(replay.nll)
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
            initial_training_nll=initial_loss,
            final_training_nll=float(replay.nll.detach().cpu().item()),
            n_likelihood_terms=n_observations,
        )
        return self

    def update(self, new_return: ArrayLike, feature_row: ArrayLike | None = None) -> InformationMatchedLSTMBEKK:
        """Advance one origin. V01 ignores the feature row."""
        if not self.uses_realized_input:
            super().update(new_return)
            return self
        state = _require_lstm_fit(self._fit, self.model_name)
        network = _require_network(self._network, self.model_name)
        if self._feature_mean is None or self._feature_scale is None or feature_row is None:
            raise InvalidModelInputError("a realized-input update requires the frozen scaler")
        observed = _as_new_return(new_return, state.n_assets)
        x_new = INTERNAL_RETURN_SCALE * (observed - state.fit_mean)
        scaled = apply_feature_scaler(
            np.asarray(feature_row, dtype=float).reshape(1, -1),
            self._feature_mean,
            self._feature_scale,
        )[0]
        network.eval()
        with torch.no_grad():
            shock = _step_shock(
                torch.tensor(x_new, dtype=TORCH_DTYPE, device=TORCH_DEVICE),
                torch.tensor(scaled, dtype=TORCH_DTYPE, device=TORCH_DEVICE),
            )
            hidden, cell, dynamic_c = _lstm_dynamic_c(
                network,
                shock,
                _torch_rnn_state(state.hidden_state),
                _torch_rnn_state(state.cell_state),
            )
            _weight, bekk_a, bekk_b = network.stationarity_weights()
            del _weight
            current = torch.tensor(
                state.current_forecast_internal, dtype=TORCH_DTYPE, device=TORCH_DEVICE
            )
            next_h = bekk_recursion_torch(
                network.static_c_matrix(),
                dynamic_c,
                bekk_a,
                bekk_b,
                shock[: state.n_assets],
                current,
            )
        forecast_internal = _numpy_matrix(next_h)
        _reject_invalid_forecast(forecast_internal, self.model_name)
        self._fit = replace(
            state,
            hidden_state=_numpy_rnn_state(hidden),
            cell_state=_numpy_rnn_state(cell),
            current_forecast_internal=forecast_internal,
            last_observed_internal_return=np.array(x_new, dtype=float, copy=True),
        )
        return self

    def _window_tensors(
        self,
        history: NDArray[np.floating],
        features: ArrayLike,
        proxy: ArrayLike,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Standardize features on this window and scale the proxy by 10^4."""
        feature_rows = None
        if self.uses_realized_input:
            raw = np.asarray(features, dtype=float)
            if raw.shape[0] != history.shape[0]:
                raise InvalidModelInputError("feature window length must match the return window")
            self._feature_mean, self._feature_scale = fit_feature_scaler(raw)
            scaled = apply_feature_scaler(raw, self._feature_mean, self._feature_scale)
            feature_rows = torch.tensor(scaled, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        proxy_rows = None
        if self.uses_qlike:
            cube = np.asarray(proxy, dtype=float)
            if cube.shape[0] != history.shape[0]:
                raise InvalidModelInputError("proxy window length must match the return window")
            proxy_rows = torch.tensor(
                cube * COVARIANCE_RESCALE, dtype=TORCH_DTYPE, device=TORCH_DEVICE
            )
        return feature_rows, proxy_rows


class LSTMBEKKRCInput(InformationMatchedLSTMBEKK):
    """V10. Realized features enter the LSTM. The return NLL is unchanged."""

    model_name = "lstm_bekk_v10"
    variant_id = "V10"
    uses_realized_input = True
    uses_qlike = False


class LSTMBEKKRCLoss(InformationMatchedLSTMBEKK):
    """V01. Return-only input, trained on reduced QLIKE."""

    model_name = "lstm_bekk_v01"
    variant_id = "V01"
    uses_realized_input = False
    uses_qlike = True


class LSTMBEKKRCBoth(InformationMatchedLSTMBEKK):
    """V11. Realized features and reduced QLIKE together."""

    model_name = "lstm_bekk_v11"
    variant_id = "V11"
    uses_realized_input = True
    uses_qlike = True


def variant_class(variant_id: str) -> type[InformationMatchedLSTMBEKK]:
    """Return the class for one named variant."""
    classes = {"V10": LSTMBEKKRCInput, "V01": LSTMBEKKRCLoss, "V11": LSTMBEKKRCBoth}
    try:
        return classes[variant_id]
    except KeyError as exc:
        raise InvalidModelInputError(f"unknown information-matched variant {variant_id}") from exc
