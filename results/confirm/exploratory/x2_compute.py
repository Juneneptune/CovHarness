"""Exploratory, after CONFIRM; not a confirmatory claim.

Evaluates saved forecast cubes. No model is refit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from covharness.features.origin_state import measurement_stress_instruments
from covharness.inference.gw import market_state_instruments
from covharness.protocol.binance_validation import ensemble_covariances, score_forecast_cube

LABEL = "exploratory, after CONFIRM; not a confirmatory claim"
REPO = Path(__file__).resolve().parents[3]
PANEL = REPO / "data/processed/binance_five_asset_panel.npz"
MEASURED = REPO / "results/confirm/inputs/confirm_measurements.npz"
JOBS = REPO / "results/confirm/jobs"
X1 = REPO / "results/confirm/exploratory/x1_numbers.json"
OUT = REPO / "results/confirm/exploratory"
PAIRS = (("V01", "LSTMBEKK-noC"), ("LSTM22", "LSTMBEKK-noC"))
SEEDED = {"V01", "LSTM22", "LSTMBEKK-noC"}


def _qlike(model_id: str, proxy: np.ndarray) -> np.ndarray:
    cubes = [
        np.array(np.load(JOBS / model_id / f"seed_{seed}" / "forecast.npz")["forecasts"], dtype=float)
        for seed in range(5)
    ]
    if model_id not in SEEDED or len(cubes) != 5:
        raise RuntimeError(f"{model_id} does not have five saved seed cubes")
    forecasts = ensemble_covariances(cubes)
    qlike, _frobenius, failures, _frobenius_failures = score_forecast_cube(forecasts, proxy)
    if failures or not np.isfinite(qlike).all():
        raise RuntimeError(f"{model_id} reduced QLIKE is incomplete")
    return qlike


def _labels(panel, origins: list[str], origin_returns: np.ndarray) -> dict[str, np.ndarray]:
    stored = json.loads(X1.read_text(encoding="utf-8"))
    dates = [str(value) for value in panel["dates"].tolist()]
    lookup = {day: index for index, day in enumerate(dates)}
    origin_rcov = np.array(panel["rcov_5min"][[lookup[day] for day in origins]], dtype=float)
    market = market_state_instruments(origin_rcov)
    stress = measurement_stress_instruments(origin_returns)
    variance_low = float(stored["variance_cutoffs"]["screen_tercile_low"])
    variance_high = float(stored["variance_cutoffs"]["screen_tercile_high"])
    correlation_low = float(stored["correlation_cutoffs"]["screen_tercile_low"])
    correlation_high = float(stored["correlation_cutoffs"]["screen_tercile_high"])
    variance = np.full(len(origins), "middle", dtype=object)
    variance[market[:, 1] <= variance_low] = "low"
    variance[market[:, 1] > variance_high] = "high"
    correlation = np.full(len(origins), "middle", dtype=object)
    correlation[market[:, 2] <= correlation_low] = "low"
    correlation[market[:, 2] > correlation_high] = "high"
    jump = np.where(stress[:, 2] >= 0.5, "yes", "no")
    return {"variance": variance, "correlation": correlation, "jump": jump}


def _summary(gap: np.ndarray) -> dict[str, object]:
    order = np.argsort(-np.abs(gap), kind="mergesort")
    trimmed = {}
    for count in (5, 10, 20):
        keep = np.ones(gap.shape[0], dtype=bool)
        keep[order[:count]] = False
        trimmed[str(count)] = float(np.mean(gap[keep]))
    return {
        "first_lower_loss_share": float(np.mean(gap < 0.0)),
        "tie_share": float(np.mean(gap == 0.0)),
        "median_gap": float(np.median(gap)),
        "mean_gap": float(np.mean(gap)),
        "mean_gap_without_largest_absolute": trimmed,
        "largest_absolute_order": order,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = np.load(PANEL)
    measured = np.load(MEASURED)
    targets = [str(value) for value in measured["target_dates"].tolist()]
    origins = [str(value) for value in measured["origin_dates"].tolist()]
    lookup = {str(value): index for index, value in enumerate(panel["dates"].tolist())}
    proxy = np.array(panel["rcov_5min"][[lookup[day] for day in targets]], dtype=float)
    labels = _labels(panel, origins, measured["origin_returns_5min"])
    payload = {
        "label": LABEL,
        "loss": "raw reduced QLIKE of the saved equal-weight covariance ensemble on the five-minute proxy",
        "note": "Seed files store each seed loss. The ensemble loss is the loss of the saved mean covariance, not the mean of the seed losses.",
        "pairs": {},
    }
    for left, right in PAIRS:
        gap = _qlike(left, proxy) - _qlike(right, proxy)
        summary = _summary(gap)
        order = summary.pop("largest_absolute_order")
        days = []
        for index in order[:20]:
            days.append(
                {
                    "date": targets[int(index)],
                    "origin": origins[int(index)],
                    "gap": float(gap[int(index)]),
                    "absolute_gap": float(abs(gap[int(index)])),
                    "variance_bucket": str(labels["variance"][int(index)]),
                    "correlation_bucket": str(labels["correlation"][int(index)]),
                    "jump_bucket": str(labels["jump"][int(index)]),
                }
            )
        payload["pairs"][f"{left} - {right}"] = {**summary, "largest_20_absolute_days": days}
    (OUT / "x2_numbers.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {item: value[item] for item in ("first_lower_loss_share", "median_gap", "mean_gap", "mean_gap_without_largest_absolute")} for key, value in payload["pairs"].items()}, indent=2))


if __name__ == "__main__":
    main()
