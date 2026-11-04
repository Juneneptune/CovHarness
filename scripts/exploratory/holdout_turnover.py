"""Mean holdout turnover of unconstrained global-minimum-variance books.

Exploratory, computed after the holdout from saved forecasts. No model is refit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from covharness.portfolio.gmv import (
    LongOnlySolverError,
    drift_weights,
    turnover,
    unconstrained_gmv_weights,
)
from covharness.protocol.binance_validation import ensemble_covariances

REPO = Path(__file__).resolve().parents[2]
JOBS = REPO / "results/confirm/jobs"
PANEL = REPO / "data/processed/binance_five_asset_panel.npz"
OUT = REPO / "results/exploratory/holdout_turnover/turnover.csv"
SEEDED = ("LSTM19", "LSTM22", "LSTMBEKK-noC", "V10", "V01", "V11")
ROSTER = (
    "RW01",
    "EWMA01",
    "EWMA21",
    "HARDRD01",
    "HARQDRD01",
    "SHARDRD01",
    "LWLIN01",
    "LWNL01",
    "RIDGE01",
    "XGB08",
    "DCC02",
    "DCCNL02",
    "DCC03",
    "DCCNL03",
    "BEKKS01",
    "LSTMBEKK-noC",
    "LSTM19",
    "LSTM22",
    "V10",
    "V01",
    "V11",
)


def forecast_path(model_id: str, seed: int | None = None) -> Path:
    """Path of one saved forecast file."""
    if seed is None:
        return JOBS / model_id / "forecast.npz"
    return JOBS / model_id / f"seed_{seed}" / "forecast.npz"


def load_forecasts(model_id: str) -> np.ndarray:
    """Saved forecast cube, shape ``(T, N, N)``.

    A seeded model returns the equal-weight mean of its five seed cubes.
    """
    if model_id in SEEDED:
        cubes = [
            np.array(np.load(forecast_path(model_id, seed))["forecasts"], dtype=float)
            for seed in range(5)
        ]
        return ensemble_covariances(cubes)
    return np.array(np.load(forecast_path(model_id))["forecasts"], dtype=float)


def holdout_simple_returns(targets: np.ndarray) -> np.ndarray:
    """Simple returns on the saved holdout targets. Shape ``(T, N)``."""
    panel = np.load(PANEL)
    dates = [str(value) for value in panel["dates"].tolist()]
    lookup = {day: index for index, day in enumerate(dates)}
    try:
        rows = [lookup[str(day)] for day in targets.tolist()]
    except KeyError as exc:
        raise SystemExit(f"holdout target {exc} is not in the return panel") from exc
    # The panel stores log returns. The portfolio code uses simple returns.
    return np.expm1(np.array(panel["daily_returns"][rows], dtype=float))


def mean_daily_turnover(forecasts: np.ndarray, simple: np.ndarray) -> float:
    """Mean L1 turnover of the unconstrained book, first day excluded.

    ``forecasts`` has shape ``(T, N, N)`` and ``simple`` has shape ``(T, N)``.
    Day ``t`` compares the new weights with the day ``t-1`` weights after
    they drift with that day's simple returns.
    """
    n_rows, n_assets = simple.shape
    if forecasts.shape[0] != n_rows or forecasts.shape[1:] != (n_assets, n_assets):
        raise SystemExit(f"forecast shape {forecasts.shape} does not match returns {simple.shape}")
    weights = np.full((n_rows, n_assets), np.nan)
    traded = np.full(n_rows, np.nan)
    # Same weight, drift, and L1 steps as the holdout economic evaluation.
    for index in range(n_rows):
        try:
            current = unconstrained_gmv_weights(forecasts[index])
        except (np.linalg.LinAlgError, LongOnlySolverError, ValueError):
            continue
        weights[index] = current
        if index == 0:
            traded[index] = 0.0
        elif np.isfinite(weights[index - 1]).all():
            drifted = drift_weights(weights[index - 1], simple[index - 1])
            traded[index] = turnover(current, drifted)
    used = traded[1:]
    if not np.isfinite(used).all():
        raise SystemExit(f"turnover is missing on {int(np.sum(~np.isfinite(used)))} days after the first")
    return float(np.mean(used))


def main() -> None:
    reference = np.load(JOBS / "HARQDRD01" / "forecast.npz")
    targets = np.asarray(reference["targets"].astype(str))
    if targets.shape[0] != 500:
        raise SystemExit(f"expected 500 holdout targets, found {targets.shape[0]}")
    simple = holdout_simple_returns(targets)
    rows = []
    means: dict[str, float] = {}
    for model_id in ROSTER:
        cube = load_forecasts(model_id)
        stored_path = forecast_path(model_id, 0 if model_id in SEEDED else None)
        stored_targets = np.load(stored_path)["targets"].astype(str)
        if not np.array_equal(stored_targets, targets):
            raise SystemExit(f"{model_id} targets do not match HARQDRD01")
        value = mean_daily_turnover(cube, simple)
        means[model_id] = value
        rows.append({"model": model_id, "mean_daily_turnover": value})
    ratio = means["HARQDRD01"] / means["V01"]
    rows.append({"model": "HARQDRD01_to_V01", "mean_daily_turnover": ratio})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"HARQDRD01 {means['HARQDRD01']:.10f}")
    print(f"V01 {means['V01']:.10f}")
    print(f"ratio {ratio:.10f}")


if __name__ == "__main__":
    main()
