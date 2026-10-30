"""VALIDATION and SCREEN for BEKK and the zero-intercept LSTM ablation.

Task C2. CONFIRM is not read. No SPA or MCS is run.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from covharness.models.bekk import BEKKCovariance, LSTMBEKKNoDynamicC
from covharness.models.lstm_bekk import ARCHITECTURE_RANGE_POST_SCREEN_D1
from covharness.models.stationarity import POLICY_CARRY_FORWARD, POLICY_INTERIOR_MAP
from covharness.protocol.binance import build_binance_segmented_protocol
from covharness.protocol.binance_screen import (
    load_screen_view,
    request_screen_schedule,
    run_one_screen_configuration,
)
from covharness.protocol.binance_validation import (
    ensemble_covariances,
    load_validation_view,
    request_validation_schedule,
    run_one_configuration,
    score_forecast_cube,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "23e728a5ebb8c2c3481a19637b9b8e0e8659b3c5"
PANEL = REPO_ROOT / "data/processed/binance_five_asset_panel.npz"
CORE = REPO_ROOT / "configs/binance_open_data_core.yaml"
SCREEN_CONFIG = REPO_ROOT / "configs/binance_open_data_screen.yaml"
OUTPUT = REPO_ROOT / "results/post_screen/C2"
CONFIGS = (
    {"id": "BEKKS01", "form": "scalar", "targeting": False},
    {"id": "BEKKS02", "form": "scalar", "targeting": True},
    {"id": "BEKKD01", "form": "diagonal", "targeting": False},
    {"id": "BEKKD02", "form": "diagonal", "targeting": True},
    {"id": "BEKKF01", "form": "full", "targeting": False},
    {"id": "BEKKF02", "form": "full", "targeting": True},
)
ABLATION = {
    "id": "LSTMBEKK-noC",
    "num_layers": 7,
    "dropout": 0.2,
    "max_epochs": 75,
    "learning_rate": 0.001,
    "gradient_clip_norm": 1.0,
    "architecture_range": ARCHITECTURE_RANGE_POST_SCREEN_D1,
}


def _log(line: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _row_from_result(result, carry_count: int) -> dict[str, object]:
    return {
        "id": result.config_id,
        "mean_qlike": result.mean_qlike,
        "mean_frobenius": result.mean_frobenius,
        "complete_support": result.complete_support,
        "repair_count": result.repair_count,
        "n_targets": result.n_targets,
        "runtime_seconds": result.runtime_seconds,
        "failure_reason": result.failure_reason,
        "carry_count": carry_count,
        "n_parameters": None,
    }


def _run_bekk(period, view, schedule, runner, row, policy: str) -> dict[str, object]:
    spec_id = row["id"] if policy == POLICY_CARRY_FORWARD else f"{row['id']}-INT"
    carry = {"count": 0}

    def factory(_family, _config, _seed):
        return BEKKCovariance(
            form=str(row["form"]),
            targeting=bool(row["targeting"]),
            failure_policy=policy,
            specification_id=spec_id,
        )

    def on_origin(origin, refit, model) -> None:
        carry["count"] = int(model.carry_count)
        if refit:
            _log(f"C2 {period} {origin.isoformat()} {spec_id} refit carry {model.carry_count}")

    try:
        result = runner(
            family="BEKK",
            row={"id": spec_id},
            seed=None,
            view=view,
            schedule=schedule,
            checkpoint_dir=OUTPUT / f"{period.lower()}_checkpoints",
            model_factory=factory,
            on_origin=on_origin,
        )
        payload = _row_from_result(result, carry["count"])
        payload["form"] = row["form"]
        payload["targeting"] = row["targeting"]
        payload["failure_policy"] = policy
        payload["headline_eligible"] = policy == POLICY_CARRY_FORWARD
        if result.complete_support:
            _log(f"C2 {period} {spec_id} done {result.mean_qlike}")
        return payload
    except Exception as exc:
        _log(f"C2 {period} {spec_id} failed {exc}")
        (OUTPUT / f"{spec_id}_{period.lower()}_traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        return {
            "id": spec_id,
            "form": row["form"],
            "targeting": row["targeting"],
            "failure_policy": policy,
            "headline_eligible": policy == POLICY_CARRY_FORWARD,
            "mean_qlike": None,
            "mean_frobenius": None,
            "complete_support": False,
            "repair_count": 0,
            "n_targets": 0,
            "runtime_seconds": None,
            "failure_reason": str(exc),
            "carry_count": carry["count"],
        }


def _select(rows: list[dict[str, object]]) -> dict[str, object] | None:
    eligible = [row for row in rows if row["complete_support"] and row["mean_qlike"] is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row["mean_qlike"]))


def _ablation(period, view, schedule, runner, score) -> dict[str, object]:
    seeds = []

    def factory(_family, row, seed_value):
        return LSTMBEKKNoDynamicC(
            seed=int(seed_value),
            num_layers=int(row["num_layers"]),
            dropout=float(row["dropout"]),
            learning_rate=float(row["learning_rate"]),
            gradient_clip_norm=float(row["gradient_clip_norm"]),
            max_epochs=int(row["max_epochs"]),
            architecture_range=str(row["architecture_range"]),
        )

    def on_origin(origin, refit, _model) -> None:
        if refit:
            _log(f"C2 {period} {origin.isoformat()} LSTMBEKK-noC refit")

    for seed in range(5):
        result = runner(
            family="LSTMBEKK-noC",
            row=ABLATION,
            seed=seed,
            view=view,
            schedule=schedule,
            checkpoint_dir=OUTPUT / f"{period.lower()}_ablation_checkpoints",
            model_factory=factory,
            on_origin=on_origin,
        )
        seeds.append(result)
        _log(f"C2 {period} LSTMBEKK-noC seed {seed} done")
    if any(not item.complete_support or item.forecasts is None for item in seeds):
        return {
            "id": "LSTMBEKK-noC",
            "complete_support": False,
            "failure_reason": "failed seed invalidates the ablation ensemble",
            "seed_qlike": [item.mean_qlike for item in seeds],
        }
    averaged = ensemble_covariances([item.forecasts for item in seeds])
    qlike, frobenius, qlike_fail, _frobenius_fail = score(averaged, _proxies(view, schedule))
    return {
        "id": "LSTMBEKK-noC",
        "complete_support": qlike_fail == 0,
        "mean_qlike": None if qlike_fail else float(np.nanmean(qlike)),
        "mean_frobenius": float(np.nanmean(frobenius)),
        "seed_qlike": [item.mean_qlike for item in seeds],
        "seed_qlike_sd": float(np.std([item.mean_qlike for item in seeds], ddof=1)),
        "n_targets": int(averaged.shape[0]),
        "headline_eligible": False,
    }


def _proxies(view, schedule):
    lookup = {day: index for index, day in enumerate(view.dates)}
    return np.stack([view.rcov[lookup[step.target.date()]] for step in schedule.steps], axis=0)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = build_binance_segmented_protocol()
    validation = load_validation_view(PANEL, CORE, protocol)
    screen = load_screen_view(PANEL, CORE, SCREEN_CONFIG, protocol)
    validation_schedule = request_validation_schedule(protocol)
    screen_schedule = request_screen_schedule(protocol)
    validation_rows = [
        _run_bekk("VALIDATION", validation, validation_schedule, run_one_configuration, row, POLICY_CARRY_FORWARD)
        for row in CONFIGS
    ]
    selected = _select(validation_rows)
    screen_rows = [
        _run_bekk("SCREEN", screen, screen_schedule, run_one_screen_configuration, row, POLICY_CARRY_FORWARD)
        for row in CONFIGS
    ]
    interior = {"status": "not_run"}
    if selected is None:
        interior = {"status": "blocked", "reason": "no configuration completed VALIDATION"}
    elif selected["form"] == "full":
        interior = {
            "status": "blocked",
            "reason": "C1 interior map is scalar. A full-matrix winner has no settled map.",
            "selected_id": selected["id"],
        }
    else:
        source = next(row for row in CONFIGS if row["id"] == selected["id"])
        interior = {
            "status": "run",
            "selected_id": selected["id"],
            "validation": _run_bekk(
                "VALIDATION", validation, validation_schedule, run_one_configuration, source, POLICY_INTERIOR_MAP
            ),
            "screen": _run_bekk(
                "SCREEN", screen, screen_schedule, run_one_screen_configuration, source, POLICY_INTERIOR_MAP
            ),
        }
    ablation = {
        "validation": _ablation(
            "VALIDATION", validation, validation_schedule, run_one_configuration, score_forecast_cube
        ),
        "screen": _ablation(
            "SCREEN", screen, screen_schedule, run_one_screen_configuration, score_forecast_cube
        ),
    }
    payload = {
        "task": "C2",
        "spec_commit": SPEC_COMMIT,
        "selected_validation_id": None if selected is None else selected["id"],
        "validation": validation_rows,
        "screen": screen_rows,
        "interior": interior,
        "ablation": ablation,
        "full_parameter_count_note": "Full BEKK at N=5 has 65 free parameters on a 250-day window.",
    }
    (OUTPUT / "c2_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("C2", payload["selected_validation_id"], flush=True)


if __name__ == "__main__":
    main()
