"""Equal-weight ensembles for the D2 variants. No model is refit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from covharness.protocol.binance import build_binance_segmented_protocol
from covharness.protocol.binance_screen import load_screen_view, request_screen_schedule
from covharness.protocol.binance_validation import (
    ensemble_covariances,
    load_validation_view,
    request_validation_schedule,
    score_forecast_cube,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "results/post_screen/D2"
PANEL = REPO_ROOT / "data/processed/binance_five_asset_panel.npz"
CORE = REPO_ROOT / "configs/binance_open_data_core.yaml"
SCREEN_CONFIG = REPO_ROOT / "configs/binance_open_data_screen.yaml"
D1 = json.loads((REPO_ROOT / "results/post_screen/D1/d1_summary.json").read_text(encoding="utf-8"))
HARQ_SCREEN_QLIKE = -31.86206280070645


def _proxies(view, schedule) -> np.ndarray:
    lookup = {day: index for index, day in enumerate(view.dates)}
    return np.stack([view.rcov[lookup[step.target.date()]] for step in schedule.steps], axis=0)


def _period(variant: str, period: str, proxies: np.ndarray) -> dict[str, object]:
    cubes = []
    seed_qlike = []
    seed_frobenius = []
    runtimes = []
    for seed in range(5):
        directory = ROOT / variant / f"seed_{seed}"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary[period]["complete_support"] is not True:
            return {"variant": variant, "complete": False, "reason": f"seed {seed} failed"}
        checkpoint = directory / f"{period}_checkpoints" / f"{variant}_LSTM22_seed{seed}.npz"
        cubes.append(np.load(checkpoint)["forecasts"])
        seed_qlike.append(summary[period]["mean_qlike"])
        seed_frobenius.append(summary[period]["mean_frobenius"])
        runtimes.append(summary[period]["runtime_seconds"])
    averaged = ensemble_covariances(cubes)
    qlike, frobenius, qlike_fail, frobenius_fail = score_forecast_cube(averaged, proxies)
    return {
        "variant": variant,
        "complete": qlike_fail == 0 and frobenius_fail == 0,
        "seed_mean_qlike": seed_qlike,
        "seed_mean": float(np.mean(seed_qlike)),
        "seed_sd": float(np.std(seed_qlike, ddof=1)),
        "seed_mean_frobenius": seed_frobenius,
        "seed_frobenius_sd": float(np.std(seed_frobenius, ddof=1)),
        "ensemble_mean_qlike": float(np.mean(qlike)),
        "ensemble_mean_frobenius": float(np.mean(frobenius)),
        "runtime_seconds": runtimes,
    }


def _v00() -> dict[str, object]:
    validation = next(row for row in D1["lstm"] if row["config_id"] == "LSTM22")
    screen = D1["screen"]["LSTM-BEKK"]
    return {
        "variant": "V00",
        "refit": False,
        "validation": {
            "seed_mean_qlike": validation["seed_mean_qlike"],
            "seed_mean": validation["seed_mean"],
            "seed_sd": validation["seed_sd"],
            "ensemble_mean_qlike": validation["ensemble_mean_qlike"],
            "ensemble_mean_frobenius": validation["ensemble_mean_frobenius"],
            "screen_seed_distribution": "not retained in the D1 summary",
        },
        "screen": {
            "ensemble_mean_qlike": screen["mean_qlike"],
            "ensemble_mean_frobenius": screen["mean_frobenius"],
            "seed_mean": None,
            "seed_sd": None,
        },
    }


def main() -> None:
    protocol = build_binance_segmented_protocol()
    validation = load_validation_view(PANEL, CORE, protocol)
    screen = load_screen_view(PANEL, CORE, SCREEN_CONFIG, protocol)
    validation_proxy = _proxies(validation, request_validation_schedule(protocol))
    screen_proxy = _proxies(screen, request_screen_schedule(protocol))
    rows = {}
    for variant in ("V10", "V01", "V11"):
        rows[variant] = {
            "validation": _period(variant, "validation", validation_proxy),
            "screen": _period(variant, "screen", screen_proxy),
        }
    v00 = _v00()
    v11_screen = rows["V11"]["screen"]["ensemble_mean_qlike"]
    payload = {
        "task": "D2",
        "spec_commit": "60b240bae32577e0159df8cc494019b3db4935d4",
        "nproc": 48,
        "parallelism": 44,
        "v00": v00,
        "variants": rows,
        "screen_gaps_descriptive": {
            "V11_minus_V00_qlike": v11_screen - v00["screen"]["ensemble_mean_qlike"],
            "HARQDRD01_minus_V11_qlike": HARQ_SCREEN_QLIKE - v11_screen,
            "harq_screen_qlike": HARQ_SCREEN_QLIKE,
        },
    }
    (ROOT / "d2_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["screen_gaps_descriptive"], indent=2))
    for variant, row in rows.items():
        print(
            variant,
            row["validation"]["ensemble_mean_qlike"],
            row["screen"]["ensemble_mean_qlike"],
        )


if __name__ == "__main__":
    main()
