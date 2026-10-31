"""SCREEN recalibration from saved forecasts. No model is refit.

Task C3. CONFIRM is not read. The headline forecast stays raw.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from covharness.data.binance_calendar import segment_label
from covharness.evaluation.recalibration import recalibrate_forecast_sequence
from covharness.inference.mz import mincer_zarnowitz_pooled_vech
from covharness.protocol.binance_validation import (
    ensemble_covariances,
    score_forecast_cube,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "4b69ed387c2057abd7917dd77ae2a053109976dd"
SCREEN_NPZ = REPO_ROOT / "results/binance_screen_forecasts.npz"
OUTPUT = REPO_ROOT / "results/post_screen/C3"
MZ_MODELS = ("LSTM19", "LSTM22", "V11")

FAMILY_KEYS = {
    "RW01": "RW_forecasts",
    "EWMA01": "EWMA_forecasts",
    "HARDRD01": "HAR-DRD_forecasts",
    "HARQDRD01": "HARQ-DRD_forecasts",
    "LWLIN01": "LW-linear_forecasts",
    "LWNL01": "LW-NL_forecasts",
    "RIDGE01": "Ridge-DRD_forecasts",
    "XGB08": "XGBoost-DRD_forecasts",
    "LSTM19": "LSTM-BEKK_forecasts",
}


def _dates(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def _protocol_refit(n_rows: int) -> np.ndarray:
    flags = np.zeros(n_rows, dtype=bool)
    flags[::21] = True
    return flags


def _load_checkpoint(path: Path) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    with np.load(path) as bundle:
        forecasts = np.array(bundle["forecasts"], dtype=float)
        origins = _dates(bundle["origins"])
        targets = _dates(bundle["targets"])
        refit = np.array(bundle["refit"], dtype=bool)
    return forecasts, origins, targets, refit


def _ensemble(paths: list[Path]) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    cubes = []
    origins: list[str] | None = None
    targets: list[str] | None = None
    refit: np.ndarray | None = None
    for path in paths:
        forecasts, path_origins, path_targets, path_refit = _load_checkpoint(path)
        cubes.append(forecasts)
        if origins is None:
            origins, targets, refit = path_origins, path_targets, path_refit
        elif path_origins != origins or path_targets != targets:
            raise ValueError(f"seed dates disagree in {path}")
        elif not np.array_equal(path_refit, refit):
            raise ValueError(f"seed refit flags disagree in {path}")
    assert origins is not None and targets is not None and refit is not None
    return ensemble_covariances(cubes), origins, targets, refit


def _seed_paths(directory: Path, pattern: str) -> list[Path]:
    paths = [directory / pattern.format(seed=seed) for seed in range(5)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def _catalog() -> list[dict[str, object]]:
    d1 = REPO_ROOT / "results/post_screen/D1/screen_checkpoints"
    c1 = REPO_ROOT / "results/post_screen/C1/screen_checkpoints"
    c2 = REPO_ROOT / "results/post_screen/C2/screen_checkpoints"
    c4 = REPO_ROOT / "results/post_screen/C4/screen_checkpoints"
    ablation = REPO_ROOT / "results/post_screen/C2/screen_ablation_checkpoints"
    rows: list[dict[str, object]] = []
    for model_id, key in FAMILY_KEYS.items():
        rows.append(
            {
                "model_id": model_id,
                "role": "original_grid",
                "appendix": False,
                "source": "screen_npz",
                "key": key,
            }
        )
    rows.extend(
        [
            {
                "model_id": "EWMA21",
                "role": "family_configuration",
                "appendix": False,
                "source": "checkpoint",
                "path": d1 / "EWMA_EWMA21.npz",
            },
            {
                "model_id": "LSTM22",
                "role": "family_configuration",
                "appendix": False,
                "source": "ensemble",
                "paths": _seed_paths(d1, "LSTM-BEKK_LSTM22_seed{seed}.npz"),
            },
            {
                "model_id": "DCC02",
                "role": "primary",
                "appendix": False,
                "source": "checkpoint",
                "path": c1 / "DCC_DCC02.npz",
            },
            {
                "model_id": "DCCNL02",
                "role": "primary",
                "appendix": False,
                "source": "checkpoint",
                "path": c1 / "DCC-NL_DCCNL02.npz",
            },
            {
                "model_id": "DCC03",
                "role": "robustness",
                "appendix": False,
                "source": "checkpoint",
                "path": c1 / "DCC_DCC03.npz",
            },
            {
                "model_id": "DCCNL03",
                "role": "robustness",
                "appendix": False,
                "source": "checkpoint",
                "path": c1 / "DCC-NL_DCCNL03.npz",
            },
            {
                "model_id": "SHARDRD01",
                "role": "econometric_extension",
                "appendix": False,
                "source": "checkpoint",
                "path": c4 / "SHAR-DRD_SHARDRD01.npz",
            },
            {
                "model_id": "BEKKS01",
                "role": "family_selection",
                "appendix": False,
                "source": "checkpoint",
                "path": c2 / "BEKK_BEKKS01.npz",
            },
            {
                "model_id": "LSTMBEKK-noC",
                "role": "ablation",
                "appendix": False,
                "source": "ensemble",
                "paths": _seed_paths(
                    ablation, "LSTMBEKK-noC_LSTMBEKK-noC_seed{seed}.npz"
                ),
            },
        ]
    )
    for model_id, role in (
        ("V10", "input_only"),
        ("V01", "objective_only"),
        ("V11", "input_and_objective"),
    ):
        root = REPO_ROOT / "results/post_screen/D2" / model_id
        paths = [
            root / f"seed_{seed}/screen_checkpoints/{model_id}_LSTM22_seed{seed}.npz"
            for seed in range(5)
        ]
        rows.append(
            {
                "model_id": model_id,
                "role": role,
                "appendix": False,
                "source": "ensemble",
                "paths": paths,
            }
        )
    for model_id in ("BEKKS02", "BEKKD01", "BEKKD02", "BEKKF01", "BEKKF02"):
        rows.append(
            {
                "model_id": model_id,
                "role": "bekk_appendix",
                "appendix": True,
                "source": "checkpoint",
                "path": c2 / f"BEKK_{model_id}.npz",
            }
        )
    return rows


def _load_row(
    row: dict[str, object],
    screen: np.lib.npyio.NpzFile,
    screen_origins: list[str],
    screen_targets: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    source = row["source"]
    if source == "screen_npz":
        forecasts = np.array(screen[str(row["key"])], dtype=float)
        refit = _protocol_refit(forecasts.shape[0])
        origins, targets = screen_origins, screen_targets
    elif source == "checkpoint":
        forecasts, origins, targets, refit = _load_checkpoint(Path(row["path"]))
    else:
        forecasts, origins, targets, refit = _ensemble(list(row["paths"]))
    if origins != screen_origins or targets != screen_targets:
        raise ValueError(f"{row['model_id']} dates do not match the SCREEN cube")
    if not np.array_equal(refit, _protocol_refit(len(targets))):
        raise ValueError(f"{row['model_id']} refit flags are not the 21-origin cadence")
    return forecasts, refit


def _mean(values: np.ndarray, failures: int) -> float | None:
    if failures or not np.isfinite(values).all():
        return None
    return float(np.mean(values))


def main() -> None:
    if SPEC_COMMIT == "PENDING":
        raise SystemExit("record the spec commit before the empirical run")
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    screen = np.load(SCREEN_NPZ)
    origins = _dates(screen["origin_dates"])
    targets = _dates(screen["target_dates"])
    proxies = np.array(screen["proxy_rcov"], dtype=float)
    segments = [segment_label(date.fromisoformat(day)) for day in targets]
    if any(label != "SCREEN" for label in segments):
        raise SystemExit("a SCREEN target is outside the SCREEN segment")
    rows = _catalog()
    raw_stack = []
    r1_stack = []
    r2_stack = []
    summaries = []
    path_rows = []
    model_ids = []
    for row in rows:
        model_id = str(row["model_id"])
        forecasts, refit = _load_row(row, screen, origins, targets)
        mapped = recalibrate_forecast_sequence(
            forecasts,
            proxies,
            origin_dates=origins,
            target_dates=targets,
            segments=segments,
            refit=refit,
        )
        scored = {}
        failures = {}
        for version, cube in (
            ("raw", forecasts),
            ("r1", mapped.r1),
            ("r2", mapped.r2),
        ):
            qlike, frobenius, qlike_fail, frobenius_fail = score_forecast_cube(
                cube, proxies
            )
            scored[version] = {
                "mean_qlike": _mean(qlike, qlike_fail),
                "mean_squared_frobenius": _mean(frobenius, frobenius_fail),
            }
            failures[version] = {
                "qlike": qlike_fail,
                "frobenius": frobenius_fail,
            }
        model_ids.append(model_id)
        raw_stack.append(forecasts)
        r1_stack.append(mapped.r1)
        r2_stack.append(mapped.r2)
        summaries.append(
            {
                "model_id": model_id,
                "role": row["role"],
                "appendix": row["appendix"],
                "n_targets": int(forecasts.shape[0]),
                "identity_count_r1": int(mapped.identity_r1.sum()),
                "identity_count_r2": int(mapped.identity_r2.sum()),
                "hbar_failure_count": len(mapped.hbar_failure_origins),
                "hbar_failure_origins": list(mapped.hbar_failure_origins),
                "scores": scored,
                "score_failures": failures,
            }
        )
        for index, origin in enumerate(origins):
            path_rows.append(
                {
                    "model_id": model_id,
                    "origin": origin,
                    "target": targets[index],
                    "scale_r1": mapped.scale_r1[index],
                    "scale_r2": mapped.scale_r2[index],
                    "lambda_r2": mapped.lambda_r2[index],
                    "identity_r1": int(mapped.identity_r1[index]),
                    "identity_r2": int(mapped.identity_r2[index]),
                }
            )
        print(model_id, scored["raw"]["mean_qlike"], scored["r2"]["mean_qlike"], flush=True)
    by_id = {item["model_id"]: item for item in summaries}
    reference = by_id["HARQDRD01"]
    for item in summaries:
        item["gap_vs_harqdrd01"] = {}
        for version in ("raw", "r1", "r2"):
            model_loss = item["scores"][version]["mean_qlike"]
            reference_loss = reference["scores"][version]["mean_qlike"]
            if model_loss is None or reference_loss is None:
                item["gap_vs_harqdrd01"][version] = None
            else:
                item["gap_vs_harqdrd01"][version] = float(model_loss - reference_loss)
    mz = {}
    for model_id in MZ_MODELS:
        position = model_ids.index(model_id)
        result = mincer_zarnowitz_pooled_vech(proxies, r2_stack[position])
        mz[model_id] = {
            "alpha": float(result.alpha),
            "beta": float(result.beta),
            "n_observations": int(result.n_observations),
        }
    payload = {
        "task": "C3",
        "spec_commit": SPEC_COMMIT,
        "headline": "raw",
        "status": "diagnostic",
        "no_forecasts": ["DCC01", "DCCNL01"],
        "validation_pairs_used": False,
        "models": summaries,
        "mz_slope_after_r2": mz,
        "runtime_seconds": time.perf_counter() - started,
    }
    summary_path = OUTPUT / "c3_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    with (OUTPUT / "c3_paths.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)
    np.savez(
        OUTPUT / "recalibrated_forecasts.npz",
        model_ids=np.array(model_ids),
        origin_dates=np.array(origins),
        target_dates=np.array(targets),
        proxy_rcov=proxies,
        raw=np.stack(raw_stack),
        r1=np.stack(r1_stack),
        r2=np.stack(r2_stack),
    )
    screen.close()
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
