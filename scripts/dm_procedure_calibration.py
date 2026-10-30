"""Calibrate confirmatory pairwise procedures on simulated nulls.

Task B1. The selection rule is fixed in the register. This script does not
choose a procedure by which one favours a fitted model. CONFIRM is not read.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from covharness.inference.bootstrap import stationary_bootstrap_indices
from covharness.inference.bootstrap_mean import PAIRWISE_N_BOOT, stationary_bootstrap_mean_test
from covharness.inference.dm import diebold_mariano
from covharness.inference.hac import llsw_sqrt_lag
from covharness.inference.size import (
    CALIBRATION_N_REPS,
    CALIBRATION_SEED,
    ar1_true_long_run_variance,
    stationary_ar1_paths,
    student_t_ar1_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "3ccffdbd84398a82879cd16d7da8d41c6ba8c5b6"
OUTPUT = REPO_ROOT / "results" / "post_screen" / "B1"
QLIKE_PATH = REPO_ROOT / "results" / "binance_screen_losses_qlike.npz"
QLIKE_SHA256 = "3fa68921871277b52e60412bae9fb23f31e7b26e39bec446d1051f7f6e8af61d"
PROCEDURES = ("P1", "P2", "P3", "P4", "P5", "P6")
SAMPLE_SIZES = (500, 250)
RHOS = (0.0, 0.3, 0.6, 0.8, 0.9)
QUALIFYING_RHOS = (0.0, 0.3, 0.6)
D3_BLOCKS = (7, 29)
ALPHA = 0.05
POWER_SHIFT = 0.2


def _log(message: str) -> None:
    path = OUTPUT / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _cell_seed(key: str) -> int:
    digest = hashlib.sha256(f"{CALIBRATION_SEED}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _rejects(series: np.ndarray, procedure: str, seed: int) -> bool:
    if procedure == "P1":
        result = diebold_mariano(series, lag_rule="newey_west_1994", reference="normal")
        return result.p_value < ALPHA
    if procedure == "P2":
        result = diebold_mariano(series, lag_rule="four_newey_west_1994", reference="normal")
        return result.p_value < ALPHA
    if procedure == "P3":
        result = diebold_mariano(series, lag_rule="llsw_sqrt", reference="fixed_b")
        return abs(result.statistic) > float(result.critical_value)
    if procedure == "P4":
        result = diebold_mariano(series, lag_rule="andrews_1991", reference="fixed_b")
        return abs(result.statistic) > float(result.critical_value)
    if procedure == "P5":
        result = diebold_mariano(series, reference="ewc")
        return abs(result.statistic) > float(result.critical_value)
    if procedure == "P6":
        result = stationary_bootstrap_mean_test(
            series,
            n_boot=PAIRWISE_N_BOOT,
            block_length=llsw_sqrt_lag(series.shape[0]),
            seed=seed,
        )
        return result.p_value < ALPHA
    raise ValueError(procedure)


def _rate(paths: np.ndarray, procedure: str, seed: int) -> dict[str, float | int]:
    rejects = 0
    failures = 0
    for index, series in enumerate(paths):
        try:
            rejects += int(_rejects(series, procedure, seed + index))
        except Exception:
            failures += 1
    valid = int(paths.shape[0] - failures)
    rate = float(rejects / valid) if valid else float("nan")
    mcse = float(np.sqrt(rate * (1.0 - rate) / valid)) if valid else float("nan")
    return {
        "replications": int(paths.shape[0]),
        "valid_replications": valid,
        "failures": failures,
        "rejection_rate": rate,
        "mcse": mcse,
    }


def _paths(design: str, n_obs: int, rho: float, seed: int, shift: float) -> np.ndarray:
    if design == "D1":
        paths = stationary_ar1_paths(
            n_reps=CALIBRATION_N_REPS, n_obs=n_obs, rho=rho, seed=seed
        )
    elif design == "D2":
        paths = student_t_ar1_paths(
            n_reps=CALIBRATION_N_REPS, n_obs=n_obs, rho=rho, seed=seed
        )
    else:
        raise ValueError(design)
    if shift != 0.0:
        paths = paths + shift * np.sqrt(ar1_true_long_run_variance(rho))
    return paths


def run_cell(payload: dict[str, object]) -> dict[str, object]:
    """Evaluate one design cell. The payload is the checkpoint key."""
    design = str(payload["design"])
    kind = str(payload["kind"])
    procedure = str(payload["procedure"])
    key = str(payload["key"])
    seed = _cell_seed(key)
    if design == "D3":
        paths = np.asarray(payload["paths"], dtype=float)
    else:
        shift = POWER_SHIFT if kind == "power" else 0.0
        paths = _paths(str(design), int(payload["T"]), float(payload["rho"]), seed, shift)
    summary = _rate(paths, procedure, seed)
    summary.update(
        {
            "design": design,
            "kind": kind,
            "procedure": procedure,
            "T": payload.get("T"),
            "rho": payload.get("rho"),
            "block_length": payload.get("block_length"),
            "seed": seed,
            "key": key,
        }
    )
    return summary


def _d3_paths(
    block_length: int,
    n_reps: int = CALIBRATION_N_REPS,
    seed_key: str | None = None,
) -> np.ndarray:
    digest = hashlib.sha256(QLIKE_PATH.read_bytes()).hexdigest()
    if digest != QLIKE_SHA256:
        raise RuntimeError("SCREEN QLIKE loss hash drifted")
    blob = np.load(QLIKE_PATH)
    labels = [str(value) for value in blob["labels"].tolist()]
    losses = np.asarray(blob["losses"], dtype=float)
    gap = losses[:, labels.index("HARQ-DRD")] - losses[:, labels.index("LSTM-BEKK")]
    centered = gap - float(gap.mean())
    # Keep the B1 draw unless the caller names a new seed key and replication count.
    key = f"D3-source-{block_length}" if seed_key is None else seed_key
    seed = _cell_seed(key)
    indices = stationary_bootstrap_indices(
        centered.shape[0],
        n_reps,
        block_length=block_length,
        seed=seed,
    )
    return centered[indices]


def _tasks() -> list[dict[str, object]]:
    tasks = []
    for design in ("D1", "D2"):
        for kind in ("size", "power"):
            for n_obs in SAMPLE_SIZES:
                for rho in RHOS:
                    for procedure in PROCEDURES:
                        key = f"{design}-{kind}-T{n_obs}-rho{rho}-{procedure}"
                        tasks.append(
                            {
                                "design": design,
                                "kind": kind,
                                "procedure": procedure,
                                "T": n_obs,
                                "rho": rho,
                                "block_length": None,
                                "key": key,
                            }
                        )
    for block_length in D3_BLOCKS:
        paths = _d3_paths(block_length)
        for procedure in PROCEDURES:
            key = f"D3-size-block{block_length}-{procedure}"
            tasks.append(
                {
                    "design": "D3",
                    "kind": "size",
                    "procedure": procedure,
                    "T": int(paths.shape[1]),
                    "rho": None,
                    "block_length": block_length,
                    "key": key,
                    "paths": paths,
                }
            )
    return tasks


def _select(rows: list[dict[str, object]]) -> dict[str, object]:
    def cell(procedure: str, design: str, kind: str, n_obs: int | None, rho: float | None, block: int | None):
        for row in rows:
            if (
                row["procedure"] == procedure
                and row["design"] == design
                and row["kind"] == kind
                and row.get("T") == n_obs
                and row.get("rho") == rho
                and row.get("block_length") == block
            ):
                return float(row["rejection_rate"])
        raise KeyError((procedure, design, kind, n_obs, rho, block))

    ranking = []
    for procedure in PROCEDURES:
        size_rates = []
        for design in ("D1", "D2"):
            for rho in QUALIFYING_RHOS:
                size_rates.append(cell(procedure, design, "size", 500, rho, None))
        for block in D3_BLOCKS:
            size_rates.append(cell(procedure, "D3", "size", 500, None, block))
        power_rates = [
            cell(procedure, design, "power", n_obs, rho, None)
            for design in ("D1", "D2")
            for n_obs in SAMPLE_SIZES
            for rho in RHOS
        ]
        qualifies = all(0.03 <= rate <= 0.075 for rate in size_rates)
        ranking.append(
            {
                "procedure": procedure,
                "qualifies": qualifies,
                "max_abs_distortion": max(abs(rate - ALPHA) for rate in size_rates),
                "average_power": float(np.mean(power_rates)),
                "size_rates": size_rates,
            }
        )
    qualified = [row for row in ranking if row["qualifies"]]
    if qualified:
        chosen = sorted(qualified, key=lambda row: (-row["average_power"], PROCEDURES.index(row["procedure"])))[0]
        rule = "highest average power among procedures inside [0.03, 0.075]"
        limitation = None
    else:
        chosen = sorted(ranking, key=lambda row: (row["max_abs_distortion"], PROCEDURES.index(row["procedure"])))[0]
        rule = "smallest maximum distortion; no procedure qualified"
        limitation = "no procedure met the size band at every required cell"
    return {"rule": rule, "chosen": chosen["procedure"], "limitation": limitation, "ranking": ranking}


PAIRS = (
    ("headline", "HARQ-DRD", "LSTM-BEKK"),
    ("quarticity", "HARQ-DRD", "HAR-DRD"),
    ("nonlinear", "HAR-DRD", "XGBoost-DRD"),
    ("qlike_mcs", "HARQ-DRD", "EWMA"),
)
LOSS_FILES = {
    "reduced_qlike": (
        REPO_ROOT / "results" / "binance_screen_losses_qlike.npz",
        QLIKE_SHA256,
    ),
    "squared_frobenius": (
        REPO_ROOT / "results" / "binance_screen_losses_frobenius.npz",
        "63a4de10db499b5527e5f1c276c17460481d85a7c3699a7a6226a97819729713",
    ),
}
ROLLING_WINDOW = 21


def demonstrate() -> None:
    """Apply frozen P5 to the four A1 pairs. The choice is not recomputed."""
    decision = json.loads((OUTPUT / "b1_calibration.json").read_text(encoding="utf-8"))
    chosen = decision["decision"]["chosen"]
    if chosen != "P5":
        raise RuntimeError(f"demonstration expected frozen P5, found {chosen}")
    rows = []
    for loss_name, (path, expected) in LOSS_FILES.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"hash mismatch for {path}")
        blob = np.load(path)
        labels = [str(value) for value in blob["labels"].tolist()]
        dates = [str(value) for value in blob["dates"].tolist()]
        losses = np.asarray(blob["losses"], dtype=float)
        for pair_id, left, right in PAIRS:
            differential = losses[:, labels.index(left)] - losses[:, labels.index(right)]
            result = diebold_mariano(differential, reference="ewc")
            cumulative = np.cumsum(differential) / np.arange(1, differential.shape[0] + 1)
            rolling = np.full(differential.shape[0], np.nan)
            # Rolling mean uses the frozen 21-origin refit cadence as the window.
            for index in range(ROLLING_WINDOW - 1, differential.shape[0]):
                rolling[index] = float(np.mean(differential[index - ROLLING_WINDOW + 1 : index + 1]))
            wins = int(np.sum(differential < 0.0))
            ties = int(np.sum(differential == 0.0))
            rows.append(
                {
                    "loss": loss_name,
                    "pair_id": pair_id,
                    "left": left,
                    "right": right,
                    "procedure": "P5",
                    "statistic": result.statistic,
                    "critical_value": result.critical_value,
                    "reject_5pct": abs(result.statistic) > float(result.critical_value),
                    "p_value": result.p_value,
                    "nu": result.hac_maxlags,
                    "cumulative_mean_end": float(cumulative[-1]),
                    "rolling_mean_end": float(rolling[-1]),
                    "rolling_window": ROLLING_WINDOW,
                    "win_rate": wins / differential.shape[0],
                    "tie_count": ties,
                    "n_observations": int(differential.shape[0]),
                    "date_start": dates[0],
                    "date_end": dates[-1],
                }
            )
            _log(f"B1 SCREEN demonstration {loss_name} {pair_id} {left}-{right}")
    (OUTPUT / "b1_demonstration.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("demonstration", len(rows), flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cell_dir = OUTPUT / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    finished = []
    for task in _tasks():
        path = cell_dir / f"{task['key']}.json"
        if path.is_file():
            finished.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        pending.append(task)
    _log(f"B1 calibrate pending {len(pending)} finished {len(finished)}")
    if pending:
        workers = min(8, len(pending))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_cell, task): task["key"] for task in pending}
            for future in as_completed(futures):
                key = futures[future]
                row = future.result()
                (cell_dir / f"{key}.json").write_text(json.dumps(row) + "\n", encoding="utf-8")
                finished.append(row)
                _log(
                    f"B1 {row['design']} {row['kind']} T={row['T']} rho={row['rho']} "
                    f"block={row['block_length']} {row['procedure']} rate={row['rejection_rate']}"
                )
    decision = _select(finished)
    payload = {
        "task": "B1",
        "spec_commit": SPEC_COMMIT,
        "seed": CALIBRATION_SEED,
        "replications": CALIBRATION_N_REPS,
        "rows": finished,
        "decision": decision,
    }
    (OUTPUT / "b1_calibration.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    print(decision["chosen"], decision["rule"], flush=True)


if __name__ == "__main__":
    import sys

    if "--demonstrate" in sys.argv:
        demonstrate()
    else:
        main()
