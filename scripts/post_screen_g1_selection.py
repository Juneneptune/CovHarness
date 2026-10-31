"""SCREEN finalist ranks and descriptive SPA/MCS. CONFIRM is not read.

Task G1. Saved raw losses only. No model is refit. The 2026-09-21
SPA and MCS files are not rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from covharness.protocol.binance_screen import (
    SUBBLOCK_BOUNDS,
    average_ranks,
    econ_median_rank_finalist,
    screen_mcs,
    screen_spa,
)
from covharness.protocol.binance_validation import score_forecast_cube

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "cd68fd556add3eb03df9f933e736c11bce4580b8"
OUTPUT = REPO_ROOT / "results/post_screen/G1"
C3_FORECASTS = REPO_ROOT / "results/post_screen/C3/recalibrated_forecasts.npz"
C3_SUMMARY = REPO_ROOT / "results/post_screen/C3/c3_summary.json"
FROZEN_FINALISTS = REPO_ROOT / "results/binance_screen_finalists.json"
QLIKE_NPZ = REPO_ROOT / "results/binance_screen_losses_qlike.npz"
INT_CHECKPOINT = (
    REPO_ROOT
    / "results/post_screen/C2/screen_checkpoints/BEKK_BEKKS01-INT.npz"
)
SCREEN_START = "2023-11-30"
SCREEN_END = "2025-04-12"
CONFIRM_START = "2025-04-13"
N_BOOT = 5000
SEED = 20260913
BLOCK_LENGTHS = (7, 29)
ECON_POOL = (
    "RW01",
    "EWMA21",
    "HARDRD01",
    "HARQDRD01",
    "SHARDRD01",
    "LWLIN01",
    "LWNL01",
    "DCC02",
    "DCCNL02",
    "BEKKS01",
)
DEEP_POOL = ("LSTM22", "V10", "V01", "V11")
FAMILY_TO_STORED = {
    "RW": "RW01",
    "EWMA": "EWMA01",
    "HAR-DRD": "HARDRD01",
    "HARQ-DRD": "HARQDRD01",
    "LW-linear": "LWLIN01",
    "LW-NL": "LWNL01",
    "Ridge-DRD": "RIDGE01",
    "XGBoost-DRD": "XGB08",
    "LSTM-BEKK": "LSTM19",
}


def _dates(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def _rank_pool(names: tuple[str, ...], qlike: dict[str, np.ndarray]) -> dict[str, object]:
    panel = np.column_stack([qlike[name] for name in names])
    block_means: list[np.ndarray] = []
    block_rank_rows: list[np.ndarray] = []
    for start, stop in SUBBLOCK_BOUNDS:
        # Rank this 125-day block. Lower QLIKE receives rank 1.
        means = panel[start:stop].mean(axis=0)
        block_means.append(means)
        block_rank_rows.append(average_ranks(means))
    ranks = np.stack(block_rank_rows, axis=1)
    median_ranks = np.median(ranks, axis=1)
    mean_ranks = ranks.mean(axis=1)
    order = sorted(
        range(len(names)),
        key=lambda index: (
            float(median_ranks[index]),
            float(mean_ranks[index]),
            names[index],
        ),
    )
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "model_id": name,
                "block_means": [float(block[index]) for block in block_means],
                "block_ranks": [float(block[index]) for block in block_rank_rows],
                "median_rank": float(median_ranks[index]),
                "mean_rank": float(mean_ranks[index]),
                "full_mean_qlike": float(panel[:, index].mean()),
            }
        )
    return {"winner": names[order[0]], "order": [names[index] for index in order], "rows": rows}


def _score_cube(forecasts: np.ndarray, proxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qlike, frobenius, qlike_fail, frobenius_fail = score_forecast_cube(forecasts, proxy)
    if qlike_fail or frobenius_fail or not np.isfinite(qlike).all() or not np.isfinite(frobenius).all():
        raise RuntimeError("a raw SCREEN column is not complete")
    return qlike, frobenius


def _load_series() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    bundle = np.load(C3_FORECASTS)
    targets = _dates(bundle["target_dates"])
    if targets[0] != SCREEN_START or targets[-1] != SCREEN_END or len(targets) != 500:
        raise RuntimeError("C3 cube is not the 500 SCREEN targets")
    if any(day >= CONFIRM_START for day in targets):
        raise RuntimeError("a CONFIRM target entered the selection cube")
    proxy = np.array(bundle["proxy_rcov"], dtype=float)
    model_ids = _dates(bundle["model_ids"])
    qlike: dict[str, np.ndarray] = {}
    frobenius: dict[str, np.ndarray] = {}
    for position, model_id in enumerate(model_ids):
        qlike[model_id], frobenius[model_id] = _score_cube(bundle["raw"][position], proxy)
    bundle.close()
    stored = json.loads(C3_SUMMARY.read_text())
    for row in stored["models"]:
        model_id = str(row["model_id"])
        expected = float(row["scores"]["raw"]["mean_qlike"])
        got = float(qlike[model_id].mean())
        if abs(got - expected) > 1e-8:
            raise RuntimeError(f"{model_id} raw QLIKE mean {got} disagrees with C3 {expected}")
    if INT_CHECKPOINT.is_file():
        checkpoint = np.load(INT_CHECKPOINT)
        checkpoint_targets = _dates(checkpoint["targets"])
        if checkpoint_targets != targets:
            raise RuntimeError("BEKKS01-INT dates do not match SCREEN")
        qlike["BEKKS01-INT"], frobenius["BEKKS01-INT"] = _score_cube(
            np.array(checkpoint["forecasts"], dtype=float), proxy
        )
        checkpoint.close()
    return qlike, frobenius, targets


def _frozen_rule_check() -> dict[str, object]:
    blob = np.load(QLIKE_NPZ)
    labels = _dates(blob["complete_labels"])
    all_labels = _dates(blob["labels"])
    index = [all_labels.index(name) for name in labels]
    losses = np.asarray(blob["losses"][:, index], dtype=float)
    dates = _dates(blob["dates"])
    blob.close()
    if dates[0] != SCREEN_START or dates[-1] != SCREEN_END or any(day >= CONFIRM_START for day in dates):
        raise RuntimeError("stored QLIKE file is not the locked SCREEN window")
    fresh = econ_median_rank_finalist(losses, labels)
    frozen = json.loads(FROZEN_FINALISTS.read_text())
    if fresh["econ_finalist"] != frozen["econ_finalist"]:
        raise RuntimeError("replayed finalist rule does not return HARQ-DRD")
    for name, value in fresh["median_ranks"].items():
        if float(value) != float(frozen["median_ranks"][name]):
            raise RuntimeError(f"replayed median rank for {name} disagrees with the frozen file")
    return {"econ_finalist": fresh["econ_finalist"], "median_ranks": fresh["median_ranks"]}


def _panel(names: list[str], series: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([series[name] for name in names])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    qlike, frobenius, targets = _load_series()
    frozen_check = _frozen_rule_check()
    for name, stored_name in FAMILY_TO_STORED.items():
        blob = np.load(QLIKE_NPZ)
        labels = _dates(blob["labels"])
        column = np.asarray(blob["losses"][:, labels.index(name)], dtype=float)
        blob.close()
        if not np.array_equal(column, qlike[stored_name]):
            gap = float(np.max(np.abs(column - qlike[stored_name])))
            if gap > 1e-8:
                raise RuntimeError(f"{stored_name} disagrees with the frozen loss file by {gap}")
    missing = [name for name in ECON_POOL + DEEP_POOL if name not in qlike]
    if missing:
        raise RuntimeError(f"missing selection columns: {missing}")
    econ = _rank_pool(ECON_POOL, qlike)
    deep = _rank_pool(DEEP_POOL, qlike)
    lstm19_means = [
        float(qlike["LSTM19"][start:stop].mean()) for start, stop in SUBBLOCK_BOUNDS
    ]
    universe = sorted(qlike)
    if "HARDRD01" not in universe:
        raise RuntimeError("SPA benchmark HARDRD01 is absent")
    qlike_panel = _panel(universe, qlike)
    frobenius_panel = _panel(universe, frobenius)
    spa = {}
    mcs = {}
    for loss_name, panel in (("qlike", qlike_panel), ("frobenius", frobenius_panel)):
        for block_length in BLOCK_LENGTHS:
            key = f"{loss_name}_block_{block_length}"
            print(f"spa {key}", flush=True)
            spa[key] = screen_spa(
                panel,
                universe,
                benchmark="HARDRD01",
                n_boot=N_BOOT,
                seed=SEED,
                block_length=block_length,
            )
            print(f"mcs {key}", flush=True)
            mcs[key] = screen_mcs(
                panel,
                universe,
                n_boot=N_BOOT,
                seed=SEED,
                block_length=block_length,
            )
    payload = {
        "task": "G1",
        "spec_commit": SPEC_COMMIT,
        "confirm_locked": True,
        "targets_scored": [targets[0], targets[-1], len(targets)],
        "frozen_rule_replay": frozen_check,
        "old_finalists": {
            "econometric": "HARQDRD01",
            "deep": "LSTM19",
            "source": "results/binance_screen_finalists.json",
        },
        "econometric_pool": econ,
        "deep_tournament": deep,
        "lstm19_outside_tournament": {
            "model_id": "LSTM19",
            "block_means": lstm19_means,
            "full_mean_qlike": float(qlike["LSTM19"].mean()),
            "tournament_rank": None,
        },
        "spa_mcs_universe": universe,
        "spa": spa,
        "mcs": mcs,
        "does_not_change_finalists": True,
    }
    (OUTPUT / "g1_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"econ {econ['winner']} deep {deep['winner']} universe {len(universe)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
