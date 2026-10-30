"""VALIDATION edge search for EWMA and LSTM-BEKK. SCREEN only if the ID changes.

Task D1. The frozen core YAML is not edited. CONFIRM stays locked.
No SPA, MCS, or finalist reselection is run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from covharness.protocol.binance import (
    CandidateValidationRecord,
    build_binance_segmented_protocol,
    select_validation_configuration,
)
from covharness.protocol.binance_screen import (
    load_screen_view,
    lstm_screen_ensemble,
    request_screen_schedule,
    run_one_screen_configuration,
)
from covharness.protocol.binance_validation import (
    _proxy_cube,
    load_validation_view,
    lstm_ensemble_result,
    request_validation_schedule,
    run_one_configuration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "PENDING"
PANEL = REPO_ROOT / "data/processed/binance_five_asset_panel.npz"
CORE = REPO_ROOT / "configs/binance_open_data_core.yaml"
SCREEN_CONFIG = REPO_ROOT / "configs/binance_open_data_screen.yaml"
D1_CONFIG = REPO_ROOT / "configs/binance_post_screen_d1.yaml"
SUMMARY = REPO_ROOT / "results/binance_validation_candidate_summary.csv"
OUTPUT = REPO_ROOT / "results/post_screen/D1"
SEEDS = (0, 1, 2, 3, 4)


def _log(period: str, origin: str, model_id: str, detail: str) -> None:
    path = OUTPUT / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"D1 {period} {origin} {model_id} {detail}\n")


def _stored_score(family: str, config_id: str, scope: str) -> float:
    import csv

    with SUMMARY.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["family"] == family and row["configuration_id"] == config_id and row["seed_scope"] == scope:
                return float(row["mean_qlike"])
    raise RuntimeError(f"stored score missing for {family} {config_id} {scope}")


def _on_origin(period: str, model_id: str):
    def callback(origin, refit, _model) -> None:
        if refit:
            _log(period, origin.isoformat(), model_id, "refit")

    return callback


def _run_ewma(view, schedule, rows: list[dict]) -> list[dict]:
    results = []
    checkpoint = OUTPUT / "validation_checkpoints"
    for row in rows:
        started = time.perf_counter()
        _log("VALIDATION", "start", row["id"], f"decay={row['decay']}")
        result = run_one_configuration(
            family="EWMA",
            row=row,
            seed=None,
            view=view,
            schedule=schedule,
            checkpoint_dir=checkpoint,
            on_origin=_on_origin("VALIDATION", row["id"]),
        )
        results.append(
            {
                "config_id": row["id"],
                "decay": row["decay"],
                "mean_qlike": result.mean_qlike,
                "mean_frobenius": result.mean_frobenius,
                "complete_support": result.complete_support,
                "runtime_seconds": time.perf_counter() - started,
                "repair_count": result.repair_count,
                "failure_reason": result.failure_reason,
            }
        )
        _log("VALIDATION", "done", row["id"], f"qlike={result.mean_qlike}")
    return results


def _run_lstm(view, schedule, rows: list[dict], proxies) -> list[dict]:
    checkpoint = OUTPUT / "validation_checkpoints"
    summaries = []
    for row in rows:
        started = time.perf_counter()
        seed_results = []
        seed_scores = []
        for seed in SEEDS:
            _log("VALIDATION", "start", row["id"], f"seed={seed}")
            result = run_one_configuration(
                family="LSTM-BEKK",
                row=row,
                seed=seed,
                view=view,
                schedule=schedule,
                checkpoint_dir=checkpoint,
                on_origin=_on_origin("VALIDATION", f"{row['id']}-seed{seed}"),
            )
            seed_results.append(result)
            seed_scores.append(result.mean_qlike)
            _log("VALIDATION", "done", row["id"], f"seed={seed} qlike={result.mean_qlike}")
        ensemble = lstm_ensemble_result(seed_results, proxies)
        finite = [score for score in seed_scores if score is not None]
        mean = float(sum(finite) / len(finite)) if finite else float("nan")
        variance = float(sum((score - mean) ** 2 for score in finite) / len(finite)) if len(finite) > 1 else float("nan")
        summaries.append(
            {
                "config_id": row["id"],
                "axis": row.get("axis"),
                "num_layers": row["num_layers"],
                "dropout": row["dropout"],
                "max_epochs": row["max_epochs"],
                "learning_rate": row["learning_rate"],
                "seed_mean_qlike": seed_scores,
                "seed_mean": mean,
                "seed_sd": variance ** 0.5,
                "ensemble_mean_qlike": ensemble.mean_qlike,
                "ensemble_mean_frobenius": ensemble.mean_frobenius,
                "complete_support": ensemble.complete_support,
                "runtime_seconds": time.perf_counter() - started,
                "repair_count": ensemble.repair_count,
                "failure_reason": ensemble.failure_reason,
            }
        )
    return summaries


def _winner(family: str, pairs: list[tuple[str, float | None]]) -> str:
    records = [
        CandidateValidationRecord(config_id, score is not None, score)
        for config_id, score in pairs
    ]
    return select_validation_configuration(records)


def _screen_one(view, schedule, family: str, row: dict) -> dict:
    checkpoint = OUTPUT / "screen_checkpoints"
    proxies = _proxy_cube(view, schedule)
    if family == "EWMA":
        result = run_one_screen_configuration(
            family=family,
            row=row,
            seed=None,
            view=view,
            schedule=schedule,
            checkpoint_dir=checkpoint,
            on_origin=_on_origin("SCREEN", row["id"]),
        )
        return {
            "config_id": row["id"],
            "mean_qlike": result.mean_qlike,
            "mean_frobenius": result.mean_frobenius,
            "complete_support": result.complete_support,
            "runtime_seconds": result.runtime_seconds,
        }
    seed_results = []
    for seed in SEEDS:
        _log("SCREEN", "start", row["id"], f"seed={seed}")
        seed_results.append(
            run_one_screen_configuration(
                family=family,
                row=row,
                seed=seed,
                view=view,
                schedule=schedule,
                checkpoint_dir=checkpoint,
                on_origin=_on_origin("SCREEN", f"{row['id']}-seed{seed}"),
            )
        )
    ensemble = lstm_screen_ensemble(seed_results, proxies)
    return {
        "config_id": row["id"],
        "mean_qlike": ensemble.mean_qlike,
        "mean_frobenius": ensemble.mean_frobenius,
        "complete_support": ensemble.complete_support,
        "runtime_seconds": ensemble.runtime_seconds,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    document = yaml.safe_load(D1_CONFIG.read_text(encoding="utf-8"))
    protocol = build_binance_segmented_protocol()
    view = load_validation_view(PANEL, CORE, protocol)
    schedule = request_validation_schedule(protocol)
    proxies = _proxy_cube(view, schedule)
    ewma_rows = _run_ewma(view, schedule, list(document["ewma"]))
    lstm_rows = _run_lstm(view, schedule, list(document["lstm"]), proxies)
    ewma_baseline = _stored_score("EWMA", "EWMA01", "deterministic")
    lstm_baseline = _stored_score("LSTM-BEKK", "LSTM19", "ensemble")
    ewma_winner = _winner("EWMA", [("EWMA01", ewma_baseline)] + [(row["config_id"], row["mean_qlike"]) for row in ewma_rows])
    lstm_winner = _winner(
        "LSTM-BEKK",
        [("LSTM19", lstm_baseline)] + [(row["config_id"], row["ensemble_mean_qlike"]) for row in lstm_rows],
    )
    extension = document["edge_extension"]
    open_question = None
    extra_ran = []
    if ewma_winner == extension["ewma"]["edge_id"]:
        extra = {"id": extension["ewma"]["next_id"], "decay": extension["ewma"]["decay"]}
        ewma_rows.extend(_run_ewma(view, schedule, [extra]))
        extra_ran.append(extra["id"])
        ewma_winner = _winner("EWMA", [("EWMA01", ewma_baseline)] + [(row["config_id"], row["mean_qlike"]) for row in ewma_rows])
    if lstm_winner == extension["learning_rate"]["edge_id"]:
        open_question = "LSTM27 is the learning-rate edge. A further learning rate is not specified."
    else:
        for axis, spec in extension.items():
            if axis in {"ewma", "learning_rate"}:
                continue
            if lstm_winner == spec["edge_id"]:
                extra = dict(spec)
                extra["id"] = spec["next_id"]
                extra["axis"] = axis
                added = _run_lstm(view, schedule, [extra], proxies)
                lstm_rows.extend(added)
                extra_ran.append(extra["id"])
                lstm_winner = _winner(
                    "LSTM-BEKK",
                    [("LSTM19", lstm_baseline)] + [(row["config_id"], row["ensemble_mean_qlike"]) for row in lstm_rows],
                )
                break
    screen = {}
    if ewma_winner != "EWMA01":
        row = next(item for item in document["ewma"] if item["id"] == ewma_winner)
        if ewma_winner == extension["ewma"]["next_id"]:
            row = {"id": ewma_winner, "decay": extension["ewma"]["decay"]}
        screen_view = load_screen_view(PANEL, CORE, SCREEN_CONFIG, protocol)
        screen["EWMA"] = _screen_one(screen_view, request_screen_schedule(protocol), "EWMA", row)
    if lstm_winner != "LSTM19":
        pool = list(document["lstm"])
        for axis, spec in extension.items():
            if spec.get("next_id"):
                pool.append({**spec, "id": spec["next_id"], "axis": axis})
        row = next(item for item in pool if item["id"] == lstm_winner)
        screen_view = load_screen_view(PANEL, CORE, SCREEN_CONFIG, protocol)
        screen["LSTM-BEKK"] = _screen_one(screen_view, request_screen_schedule(protocol), "LSTM-BEKK", row)
    payload = {
        "task": "D1",
        "spec_commit": SPEC_COMMIT,
        "ewma_baseline_qlike": ewma_baseline,
        "lstm_baseline_ensemble_qlike": lstm_baseline,
        "ewma": ewma_rows,
        "lstm": lstm_rows,
        "ewma_winner": ewma_winner,
        "lstm_winner": lstm_winner,
        "extra_configurations": extra_ran,
        "open_question": open_question,
        "ewma_interior": ewma_winner not in {"EWMA01", "EWMA21", extension["ewma"]["next_id"]},
        "lstm_interior": lstm_winner not in {"LSTM19", "LSTM22", "LSTM24", "LSTM26", "LSTM27", "LSTM28", "LSTM29", "LSTM30"},
        "screen": screen,
        "searched_totals": {
            "EWMA": 20 + len(ewma_rows),
            "LSTM-BEKK": 20 + len(lstm_rows),
        },
    }
    (OUTPUT / "d1_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("D1", ewma_winner, lstm_winner, open_question, flush=True)


if __name__ == "__main__":
    main()
