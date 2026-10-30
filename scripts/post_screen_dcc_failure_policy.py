"""VALIDATION and SCREEN for the DCC carry-forward and interior-map policies.

Task C1. DCC01 and DCCNL01 stay failed. No SPA, MCS, or finalist selection.
CONFIRM stays locked. The 2026-09-21 artifacts are not rewritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from covharness.models.dcc import DCCCovariance, DCCNonlinearCovariance
from covharness.protocol.binance import build_binance_segmented_protocol
from covharness.protocol.binance_screen import (
    load_screen_view,
    request_screen_schedule,
    run_one_screen_configuration,
)
from covharness.protocol.binance_validation import (
    load_validation_view,
    request_validation_schedule,
    run_one_configuration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "ed7d2a4761fa5702e7c18e55c48eb2d7cc0c50b7"
PANEL = REPO_ROOT / "data/processed/binance_five_asset_panel.npz"
CORE = REPO_ROOT / "configs/binance_open_data_core.yaml"
SCREEN_CONFIG = REPO_ROOT / "configs/binance_open_data_screen.yaml"
DIAGNOSTICS = REPO_ROOT / "results/binance_screen_diagnostics.json"
OUTPUT = REPO_ROOT / "results/post_screen/C1"
MODELS = (
    ("DCC", "DCC02", "carry_forward"),
    ("DCC-NL", "DCCNL02", "carry_forward"),
    ("DCC", "DCC03", "interior_map"),
    ("DCC-NL", "DCCNL03", "interior_map"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log(period: str, origin: str, model_id: str, detail: str) -> None:
    path = OUTPUT / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"C1 {period} {origin} {model_id} {detail}\n")


def _run_period(period: str, view, schedule, runner, checkpoint_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, config_id, policy in MODELS:
        events: list[dict[str, object]] = []
        seen = {"n": 0}

        def on_origin(origin, refit, model, config_id=config_id, events=events, seen=seen) -> None:
            _log(period, origin.isoformat(), config_id, f"refit={refit}")
            # Attach the calendar origin to events created on this step.
            while seen["n"] < len(model.policy_events):
                event = dict(model.policy_events[seen["n"]])
                event["origin"] = origin.isoformat()
                event["model_id"] = config_id
                event["period"] = period
                events.append(event)
                seen["n"] += 1

        def factory(factory_family, row, seed, policy=policy):
            if factory_family == "DCC":
                return DCCCovariance(failure_policy=policy)
            if factory_family == "DCC-NL":
                return DCCNonlinearCovariance(failure_policy=policy)
            raise ValueError(f"C1 does not fit {factory_family}")

        _log(period, "start", config_id, policy)
        result = runner(
            family=family,
            row={"id": config_id, "failure_policy": policy},
            seed=None,
            view=view,
            schedule=schedule,
            checkpoint_dir=checkpoint_dir,
            model_factory=factory,
            on_origin=on_origin,
        )
        _log(period, "done", config_id, f"support={result.complete_support}")
        rows.append(
            {
                "family": family,
                "config_id": config_id,
                "failure_policy": policy,
                "headline_eligible": policy == "carry_forward",
                "complete_support": result.complete_support,
                "n_targets": result.n_targets,
                "mean_qlike": result.mean_qlike,
                "mean_frobenius": result.mean_frobenius,
                "repair_count": result.repair_count,
                "fit_failures": result.fit_failures,
                "failure_reason": result.failure_reason,
                "runtime_seconds": result.runtime_seconds,
                "qlike_failures": result.qlike_failures,
                "frobenius_failures": result.frobenius_failures,
                "policy_events": events,
                "carry_forward_count": sum(
                    1 for event in events if event.get("policy") == "carry_forward"
                ),
                "near_boundary_count": sum(
                    1 for event in events if event.get("near_boundary") is True
                ),
            }
        )
    return rows


def _existing_columns() -> list[dict[str, object]]:
    payload = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    columns = []
    for row in payload["models"]:
        columns.append(
            {
                "config_id": row["configuration"],
                "status": row["status"],
                "mean_qlike": row["mean_qlike"],
                "mean_frobenius": row["mean_frobenius"],
                "source": "results/binance_screen_diagnostics.json",
            }
        )
    return columns


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = build_binance_segmented_protocol()
    validation_view = load_validation_view(PANEL, CORE, protocol)
    validation_schedule = request_validation_schedule(protocol)
    screen_view = load_screen_view(PANEL, CORE, SCREEN_CONFIG, protocol)
    screen_schedule = request_screen_schedule(protocol)
    validation_rows = _run_period(
        "VALIDATION",
        validation_view,
        validation_schedule,
        run_one_configuration,
        OUTPUT / "validation_checkpoints",
    )
    screen_rows = _run_period(
        "SCREEN",
        screen_view,
        screen_schedule,
        run_one_screen_configuration,
        OUTPUT / "screen_checkpoints",
    )
    # Hash checkpoint forecasts. The NPZ files stay local.
    artifacts = {}
    for directory in (OUTPUT / "validation_checkpoints", OUTPUT / "screen_checkpoints"):
        for path in sorted(directory.glob("*.npz")):
            artifacts[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    summary = {
        "task": "C1",
        "spec_commit": SPEC_COMMIT,
        "label": "robustness/selection",
        "confirm_locked": True,
        "stationarity_epsilon": 1e-6,
        "near_boundary_sum": 0.999,
        "validation": validation_rows,
        "screen": screen_rows,
        "existing_screen_columns": _existing_columns(),
        "artifact_sha256": artifacts,
        "untouched": [
            "results/binance_screen_failures.json",
            "results/binance_screen_diagnostics.json",
            "configs/binance_open_data_core.yaml",
        ],
    }
    (OUTPUT / "c1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("C1 complete", flush=True)


if __name__ == "__main__":
    main()
