"""Exploratory, after CONFIRM; not a confirmatory claim.

Reads saved CONFIRM forecasts and the production panel. Does not rewrite them.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from covharness.features.origin_state import measurement_stress_instruments
from covharness.inference.gw import giacomini_white, market_state_instruments
from covharness.protocol.binance import build_binance_segmented_protocol
from covharness.protocol.binance_screen import request_screen_schedule
from covharness.protocol.binance_validation import ensemble_covariances, score_forecast_cube

LABEL = "exploratory, after CONFIRM; not a confirmatory claim"
REPO = Path(__file__).resolve().parents[3]
PANEL = REPO / "data/processed/binance_five_asset_panel.npz"
MEASURED = REPO / "results/confirm/inputs/confirm_measurements.npz"
JOBS = REPO / "results/confirm/jobs"
DIAGNOSTICS = REPO / "results/confirm/diagnostics.json"
OUT = REPO / "results/confirm/exploratory"
SELL_OFF = (np.datetime64("2025-10-10"), np.datetime64("2025-10-11"))
PAIRS = (
    ("HARQDRD01", "V01"),
    ("HARQDRD01", "BEKKS01"),
    ("V01", "LSTMBEKK-noC"),
    ("LSTM22", "LSTMBEKK-noC"),
)
SEEDED = {"V01", "LSTM22", "LSTMBEKK-noC"}
MARKET_NAMES = ("constant", "log_mean_realized_variance", "mean_realized_correlation")
STRESS_NAMES = ("constant", "log_aggregate_realized_quarticity", "bns_jump_indicator")


def _qlike(model_id: str, proxy: np.ndarray) -> np.ndarray:
    if model_id in SEEDED:
        cubes = [
            np.array(np.load(JOBS / model_id / f"seed_{seed}" / "forecast.npz")["forecasts"], dtype=float)
            for seed in range(5)
        ]
        forecasts = ensemble_covariances(cubes)
    else:
        forecasts = np.array(np.load(JOBS / model_id / "forecast.npz")["forecasts"], dtype=float)
    qlike, _frobenius, failures, _frobenius_failures = score_forecast_cube(forecasts, proxy)
    if failures or not np.isfinite(qlike).all():
        raise RuntimeError(f"{model_id} reduced QLIKE is incomplete")
    return qlike


def _origin_rcov(panel, dates: list[str]) -> np.ndarray:
    stored = [str(value) for value in panel["dates"].tolist()]
    lookup = {day: index for index, day in enumerate(stored)}
    rows = [lookup[day] for day in dates]
    return np.array(panel["rcov_5min"][rows], dtype=float)


def _terciles(screen: np.ndarray, confirm: np.ndarray) -> dict[str, object]:
    low, high = np.quantile(screen, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.full(confirm.shape[0], "middle", dtype=object)
    labels[confirm <= low] = "low"
    labels[confirm > high] = "high"
    return {
        "screen_tercile_low": float(low),
        "screen_tercile_high": float(high),
        "rule": "low at or below the SCREEN one-third quantile; high above the SCREEN two-thirds quantile",
        "labels": labels,
    }


def _bucket(name: str, mask: np.ndarray, gaps: dict[str, np.ndarray]) -> dict[str, object]:
    row: dict[str, object] = {"bucket": name, "n_days": int(mask.sum())}
    for pair, gap in gaps.items():
        used = gap[mask]
        second_wins = float(np.mean(used > 0.0)) if used.size else None
        first_wins = float(np.mean(used < 0.0)) if used.size else None
        ties = float(np.mean(used == 0.0)) if used.size else None
        first, second = pair.split(" - ")
        v01_share = None
        if first == "V01":
            v01_share = first_wins
        elif second == "V01":
            v01_share = second_wins
        row[pair] = {
            "mean_gap": None if used.size == 0 else float(np.mean(used)),
            "first_model_lower_loss_share": first_wins,
            "second_model_lower_loss_share": second_wins,
            "tie_share": ties,
            "v01_lower_loss_share": v01_share,
        }
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = np.load(PANEL)
    measured = np.load(MEASURED)
    targets = [str(value) for value in measured["target_dates"].tolist()]
    origins = [str(value) for value in measured["origin_dates"].tolist()]
    proxy = _origin_rcov(panel, targets)
    losses = {model_id: _qlike(model_id, proxy) for model_id in ("HARQDRD01", "V01", "BEKKS01", "LSTM22", "LSTMBEKK-noC")}
    gaps = {f"{left} - {right}": losses[left] - losses[right] for left, right in PAIRS}
    headline = gaps["HARQDRD01 - V01"]

    # Same origin-day instruments as the committed Giacomini-White statistics.
    market = market_state_instruments(_origin_rcov(panel, origins))
    stress = measurement_stress_instruments(measured["origin_returns_5min"])
    stored = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))["gw"]
    market_result = giacomini_white(losses["HARQDRD01"], losses["V01"], market)
    stress_result = giacomini_white(losses["HARQDRD01"], losses["V01"], stress)
    if abs(market_result.statistic - stored["market_statistic"]) > 1e-8:
        raise RuntimeError("market-state Giacomini-White statistic does not match diagnostics.json")
    if abs(stress_result.statistic - stored["measurement_statistic"]) > 1e-8:
        raise RuntimeError("measurement Giacomini-White statistic does not match diagnostics.json")

    coefficients = {}
    for set_name, instruments, names in (
        ("market_state", market, MARKET_NAMES),
        ("measurement_stress", stress, STRESS_NAMES),
    ):
        beta, *_rest = np.linalg.lstsq(instruments, headline, rcond=None)
        coefficients[set_name] = [
            {"instrument": name, "coefficient": float(value), "sign": int(np.sign(value))}
            for name, value in zip(names, beta)
        ]

    screen = request_screen_schedule(build_binance_segmented_protocol())
    screen_origins = [step.origin.date().isoformat() for step in screen.steps]
    screen_market = market_state_instruments(_origin_rcov(panel, screen_origins))
    variance = _terciles(screen_market[:, 1], market[:, 1])
    correlation = _terciles(screen_market[:, 2], market[:, 2])
    jump = stress[:, 2].astype(int)
    buckets = []
    for label in ("low", "middle", "high"):
        buckets.append(_bucket(f"log_mean_realized_variance {label}", variance["labels"] == label, gaps))
        buckets.append(_bucket(f"mean_realized_correlation {label}", correlation["labels"] == label, gaps))
    buckets.append(_bucket("bns_jump no", jump == 0, gaps))
    buckets.append(_bucket("bns_jump yes", jump == 1, gaps))

    target_days = np.array(targets, dtype="datetime64[D]")
    inside = (target_days >= SELL_OFF[0]) & (target_days <= SELL_OFF[1])
    total = float(np.sum(headline))
    window_sum = float(np.sum(headline[inside]))
    figure = OUT / "x1_cumulative_gap.png"
    plt.figure(figsize=(8, 4.2))
    plt.plot(target_days, np.cumsum(headline), color="black", linewidth=1.0)
    plt.axvspan(SELL_OFF[0], SELL_OFF[1] + np.timedelta64(1, "D"), color="0.75", alpha=0.8, label="2025-10-10 to 2025-10-11")
    plt.xlabel("CONFIRM target date")
    plt.ylabel("Cumulative reduced QLIKE gap")
    plt.title(LABEL)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure)
    plt.close()

    payload = {
        "label": LABEL,
        "loss": "raw reduced QLIKE on the five-minute proxy",
        "gap_definition": "loss of the first named model minus loss of the second; a positive value favours the second",
        "gw_source": "coefficients are least squares of the headline gap on the instrument sets whose joint statistics are in diagnostics.json",
        "stored_gw": stored,
        "coefficients": coefficients,
        "screen_origin_start": screen_origins[0],
        "screen_origin_end": screen_origins[-1],
        "screen_n": len(screen_origins),
        "variance_cutoffs": {key: variance[key] for key in ("screen_tercile_low", "screen_tercile_high", "rule")},
        "correlation_cutoffs": {key: correlation[key] for key in ("screen_tercile_low", "screen_tercile_high", "rule")},
        "buckets": buckets,
        "selloff_window": ["2025-10-10", "2025-10-11"],
        "selloff_source": "UTC dates of the 10 October 2025 liquidation event recorded by Reuters on 13 October 2025",
        "selloff_n_days": int(inside.sum()),
        "selloff_gap_sum": window_sum,
        "total_gap_sum": total,
        "selloff_share_of_total_gap": window_sum / total,
        "headline_mean_gap": float(np.mean(headline)),
    }
    (OUT / "x1_numbers.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("coefficients", "variance_cutoffs", "correlation_cutoffs", "selloff_share_of_total_gap", "selloff_n_days")}, indent=2))
    print("buckets", [(row["bucket"], row["n_days"]) for row in buckets])


if __name__ == "__main__":
    main()
