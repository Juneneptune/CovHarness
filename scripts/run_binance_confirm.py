"""One CONFIRM run. The unlock is inside the orchestrator.

Forecast cubes stay local. This script does not edit the preregistration,
the production panel, or any model module.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from covharness.models.bekk import BEKKCovariance, LSTMBEKKNoDynamicC
from covharness.models.dcc import DCCCovariance, DCCNonlinearCovariance
from covharness.models.lstm_bekk_variants import realized_feature_matrix, variant_class
from covharness.models.shar_drd import SHARDRDRealizedCovariance
from covharness.protocol.binance import build_binance_segmented_protocol
from covharness.protocol.binance_confirm import (
    N_TARGETS,
    assert_confirm_schedule,
    confirm_schedule,
    load_confirm_view,
    run_confirm_forecasts,
)
from covharness.protocol.binance_validation import build_model, score_forecast_cube
from covharness.realized.binance_panel import (
    ANCHOR_DATE,
    _load_close_cube,
    measure_day_on_interval,
)
from covharness.realized.rcov import realized_semivariances

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL = REPO_ROOT / "data/processed/binance_five_asset_panel.npz"
CORE = REPO_ROOT / "configs/binance_open_data_core.yaml"
SCREEN_CONFIG = REPO_ROOT / "configs/binance_open_data_screen.yaml"
D1_CONFIG = REPO_ROOT / "configs/binance_post_screen_d1.yaml"
CACHE = REPO_ROOT / "data/cache/open_data/binance/spot/monthly/klines/1m"
C4_SEMI = REPO_ROOT / "results/post_screen/C4/semivariance_panel.npz"
OUTPUT = REPO_ROOT / "results/confirm"
INPUTS = OUTPUT / "inputs"
JOBS = OUTPUT / "jobs"
PYTHON = Path(sys.executable)
MANIFEST_HEAD = "965b229b349b5c229abc2e7dc39f67e07a179770"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _log(line: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _json(payload: dict) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _load_rows() -> dict[str, dict]:
    screen = yaml.safe_load(SCREEN_CONFIG.read_text(encoding="utf-8"))
    d1 = yaml.safe_load(D1_CONFIG.read_text(encoding="utf-8"))
    representatives = screen["representatives"]
    ewma21 = next(row for row in d1["ewma"] if row["id"] == "EWMA21")
    lstm22 = next(row for row in d1["lstm"] if row["id"] == "LSTM22")
    return {
        "screen": representatives,
        "ewma21": ewma21,
        "lstm22": lstm22,
    }


def build_inputs() -> dict[str, object]:
    """Build CONFIRM-dated 15-minute RCov and five-minute semivariances."""
    INPUTS.mkdir(parents=True, exist_ok=True)
    marker = INPUTS / "inputs_manifest.json"
    cube_path = INPUTS / "confirm_measurements.npz"
    if marker.is_file() and cube_path.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    schedule = confirm_schedule(build_binance_segmented_protocol())
    assert_confirm_schedule(schedule)
    view = load_confirm_view(PANEL, schedule)
    targets = [step.target.date() for step in schedule.steps]
    origins = [step.origin.date() for step in schedule.steps]
    _log("CONFIRM measurement load start")
    cube, _cube_days, _checksums = _load_close_cube(
        CACHE, start=ANCHOR_DATE, end=targets[-1]
    )
    rcov15 = np.empty((N_TARGETS, 5, 5), dtype=float)
    positive = np.empty((N_TARGETS, 5), dtype=float)
    negative = np.empty((N_TARGETS, 5), dtype=float)
    origin_returns = np.empty((N_TARGETS, 288, 5), dtype=float)
    # Measure each CONFIRM target on the frozen five-minute and 15-minute grids.
    for index, day in enumerate(targets):
        returns, _matrix, _daily = measure_day_on_interval(cube, ANCHOR_DATE, day, 5)
        rs_pos, rs_neg = realized_semivariances(returns)
        _returns15, matrix15, _daily15 = measure_day_on_interval(cube, ANCHOR_DATE, day, 15)
        positive[index] = rs_pos
        negative[index] = rs_neg
        rcov15[index] = matrix15
        if index % 50 == 0:
            _log(f"CONFIRM measurement {day.isoformat()}")
    # Keep origin-day five-minute returns for the frozen Giacomini-White instruments.
    for index, day in enumerate(origins):
        returns, _matrix, _daily = measure_day_on_interval(cube, ANCHOR_DATE, day, 5)
        origin_returns[index] = returns
    np.savez_compressed(
        cube_path,
        target_dates=np.array([day.isoformat() for day in targets]),
        origin_dates=np.array([day.isoformat() for day in origins]),
        rcov_15min=rcov15,
        rs_positive=positive,
        rs_negative=negative,
        origin_returns_5min=origin_returns,
    )
    payload = {
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_commit": MANIFEST_HEAD,
        "panel_sha256": _sha256_file(PANEL),
        "file_sha256": _sha256_file(cube_path),
        "rcov_15min_sha256": _sha256_array(rcov15),
        "rs_positive_sha256": _sha256_array(positive),
        "rs_negative_sha256": _sha256_array(negative),
        "origin_returns_5min_sha256": _sha256_array(origin_returns),
        "n_targets": N_TARGETS,
        "view_start": view.dates[0].isoformat(),
        "view_end": view.dates[-1].isoformat(),
        "note": (
            "15-minute RCov and five-minute semivariances are CONFIRM target dates. "
            "Origin-day five-minute returns are the frozen measurement-stress input."
        ),
    }
    marker.write_text(_json(payload), encoding="utf-8")
    _log("CONFIRM measurement done")
    return payload


def _semivariance_view(view_dates: tuple[dt.date, ...]) -> tuple[np.ndarray, np.ndarray]:
    stored = np.load(C4_SEMI)
    measured = np.load(INPUTS / "confirm_measurements.npz")
    c4_lookup = {str(day): index for index, day in enumerate(stored["dates"].tolist())}
    confirm_lookup = {
        str(day): index for index, day in enumerate(measured["target_dates"].tolist())
    }
    positive = np.empty((len(view_dates), 5), dtype=float)
    negative = np.empty((len(view_dates), 5), dtype=float)
    # Prefer the frozen pre-CONFIRM panel and fill only CONFIRM dates from this run.
    for index, day in enumerate(view_dates):
        key = day.isoformat()
        if key in c4_lookup:
            row = c4_lookup[key]
            positive[index] = stored["rs_positive"][row]
            negative[index] = stored["rs_negative"][row]
        elif key in confirm_lookup:
            row = confirm_lookup[key]
            positive[index] = measured["rs_positive"][row]
            negative[index] = measured["rs_negative"][row]
        else:
            raise RuntimeError(f"no semivariance for {key}")
    return positive, negative


def _construct(model_id: str, seed: int | None, rows: dict[str, dict]):
    screen = rows["screen"]
    if model_id == "RW01":
        return build_model("RW", {"id": "RW01"}, seed)
    if model_id == "EWMA01":
        return build_model("EWMA", screen["EWMA"], seed)
    if model_id == "EWMA21":
        return build_model("EWMA", rows["ewma21"], seed)
    if model_id == "HARDRD01":
        return build_model("HAR-DRD", {"id": "HARDRD01"}, seed)
    if model_id == "HARQDRD01":
        return build_model("HARQ-DRD", {"id": "HARQDRD01"}, seed)
    if model_id == "SHARDRD01":
        return SHARDRDRealizedCovariance()
    if model_id == "LWLIN01":
        return build_model("LW-linear", {"id": "LWLIN01"}, seed)
    if model_id == "LWNL01":
        return build_model("LW-NL", {"id": "LWNL01"}, seed)
    if model_id == "RIDGE01":
        return build_model("Ridge-DRD", screen["Ridge-DRD"], seed)
    if model_id == "XGB08":
        return build_model("XGBoost-DRD", screen["XGBoost-DRD"]["parameters"], seed)
    if model_id == "DCC02":
        return DCCCovariance(failure_policy="carry_forward")
    if model_id == "DCCNL02":
        return DCCNonlinearCovariance(failure_policy="carry_forward")
    if model_id == "DCC03":
        return DCCCovariance(failure_policy="interior_map")
    if model_id == "DCCNL03":
        return DCCNonlinearCovariance(failure_policy="interior_map")
    if model_id == "BEKKS01":
        return BEKKCovariance(
            form="scalar",
            targeting=False,
            failure_policy="carry_forward",
            specification_id="BEKKS01",
        )
    if model_id == "LSTM19":
        return build_model("LSTM-BEKK", screen["LSTM-BEKK"]["parameters"], seed)
    if model_id == "LSTM22":
        return build_model("LSTM-BEKK", rows["lstm22"], seed)
    if model_id == "LSTMBEKK-noC":
        row = rows["lstm22"]
        return LSTMBEKKNoDynamicC(
            seed=int(seed),
            num_layers=int(row["num_layers"]),
            dropout=float(row["dropout"]),
            learning_rate=float(row["learning_rate"]),
            gradient_clip_norm=float(row["gradient_clip_norm"]),
            max_epochs=int(row["max_epochs"]),
            architecture_range=str(row["architecture_range"]),
        )
    if model_id in ("V10", "V01", "V11"):
        row = rows["lstm22"]
        model_cls = variant_class(model_id)
        return model_cls(
            seed=int(seed),
            num_layers=int(row["num_layers"]),
            dropout=float(row["dropout"]),
            learning_rate=float(row["learning_rate"]),
            gradient_clip_norm=float(row["gradient_clip_norm"]),
            max_epochs=int(row["max_epochs"]),
            architecture_range=str(row["architecture_range"]),
        )
    raise ValueError(f"unknown CONFIRM model {model_id}")


def _job_dir(model_id: str, seed: int | None) -> Path:
    if seed is None:
        return JOBS / model_id
    return JOBS / model_id / f"seed_{seed}"


def _checkpoint_complete(path: Path, targets: list[str]) -> bool:
    if not path.is_file():
        return False
    blob = np.load(path)
    stored = [str(value) for value in blob["targets"].tolist()]
    return stored == targets and int(blob["forecasts"].shape[0]) == N_TARGETS


def run_job(model_id: str, seed: int | None) -> None:
    """Forecast one configuration. A complete checkpoint is not refit."""
    torch.set_num_threads(1)
    directory = _job_dir(model_id, seed)
    directory.mkdir(parents=True, exist_ok=True)
    schedule = confirm_schedule(build_binance_segmented_protocol())
    assert_confirm_schedule(schedule)
    targets = [step.target.date().isoformat() for step in schedule.steps]
    checkpoint = directory / "forecast.npz"
    label = model_id if seed is None else f"{model_id} seed {seed}"
    if _checkpoint_complete(checkpoint, targets):
        _log(f"CONFIRM {label} checkpoint already complete")
        return
    view = load_confirm_view(PANEL, schedule)
    positive, negative = _semivariance_view(view.dates)
    features = None
    proxy = None
    if model_id in ("V10", "V11"):
        features = realized_feature_matrix(positive, negative, view.rcov, view.rq)
        proxy = view.rcov
    elif model_id == "V01":
        # V01 does not read features. The shared fit signature still requires the array.
        features = np.zeros((len(view.dates), 1), dtype=float)
        proxy = view.rcov
    semi_pos = positive if model_id == "SHARDRD01" else None
    semi_neg = negative if model_id == "SHARDRD01" else None
    rows = _load_rows()
    model = _construct(model_id, seed, rows)
    events: list[dict] = []
    seen = {"n": 0}

    def on_origin(origin, refit, fitted) -> None:
        if refit:
            _log(f"CONFIRM {label} refit {origin.isoformat()}")
        policy_events = getattr(fitted, "policy_events", None)
        if policy_events is None:
            return
        while seen["n"] < len(policy_events):
            event = dict(policy_events[seen["n"]])
            event["origin"] = origin.isoformat()
            event["model_id"] = model_id
            events.append(event)
            seen["n"] += 1

    started = time.perf_counter()
    _log(f"CONFIRM {label} start")
    records, repaired, fit_failures, extras = run_confirm_forecasts(
        model=model,
        schedule=schedule,
        view=view,
        on_origin=on_origin,
        semivariance_positive=semi_pos,
        semivariance_negative=semi_neg,
        feature_panel=features,
        proxy_panel=proxy,
    )
    forecasts = np.stack([record.covariance for record in records], axis=0)
    target_rows = [view.dates.index(step.target.date()) for step in schedule.steps]
    qlike, frobenius, qlike_fail, frobenius_fail = score_forecast_cube(
        forecasts, view.rcov[target_rows]
    )
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        checkpoint,
        forecasts=forecasts,
        qlike=qlike,
        frobenius=frobenius,
        repaired=repaired,
        refit=np.array([record.refit for record in records]),
        origins=np.array([record.origin.date().isoformat() for record in records]),
        targets=np.array([record.target.date().isoformat() for record in records]),
        min_eigenvalues=extras["min_eigenvalues"],
        positive_definite=extras["positive_definite"],
        actions=np.array(extras["actions"]),
    )
    repaired_dates = [
        targets[index] for index, flag in enumerate(repaired.tolist()) if flag
    ]
    (directory / "summary.json").write_text(
        _json(
            {
                "model_id": model_id,
                "seed": seed,
                "n_targets": len(records),
                "fit_failures": fit_failures,
                "qlike_failures": qlike_fail,
                "frobenius_failures": frobenius_fail,
                "mean_qlike": None if not np.isfinite(qlike).all() else float(np.mean(qlike)),
                "mean_frobenius": None
                if not np.isfinite(frobenius).all()
                else float(np.mean(frobenius)),
                "repair_count": int(np.sum(repaired)),
                "repaired_targets": repaired_dates,
                "policy_events": events,
                "runtime_seconds": elapsed,
                "manifest_commit": MANIFEST_HEAD,
            }
        ),
        encoding="utf-8",
    )
    _log(f"CONFIRM {label} done {elapsed:.1f}s")


def _command(model_id: str, seed: int | None) -> list[str]:
    command = [str(PYTHON), str(Path(__file__).resolve()), "--job", model_id]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def launch_jobs() -> None:
    """Run every roster job once, with at most nproc-4 processes."""
    build_inputs()
    pending: list[tuple[str, int | None]] = []
    for model_id in ROSTER:
        if model_id in SEEDED:
            pending.extend((model_id, seed) for seed in range(5))
        else:
            pending.append((model_id, None))
    limit = max(1, os.cpu_count() - 4)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    running: list[tuple[subprocess.Popen, str, int | None, int]] = []
    failures: list[str] = []

    def _reap() -> None:
        still = []
        for process, model_id, seed, restarts in running:
            code = process.poll()
            if code is None:
                still.append((process, model_id, seed, restarts))
                continue
            label = model_id if seed is None else f"{model_id} seed {seed}"
            if code != 0 and restarts < 1 and code < 0:
                _log(f"CONFIRM {label} infrastructure exit {code}; restart once")
                restarted = subprocess.Popen(_command(model_id, seed), cwd=REPO_ROOT, env=env)
                still.append((restarted, model_id, seed, restarts + 1))
            elif code != 0:
                failures.append(f"{label} exit {code}")
                _log(f"CONFIRM {label} failed exit {code}")
        running[:] = still

    for model_id, seed in pending:
        while len(running) >= limit:
            _reap()
            if len(running) >= limit:
                time.sleep(5)
        label = model_id if seed is None else f"{model_id} seed {seed}"
        _log(f"CONFIRM launch {label}")
        process = subprocess.Popen(_command(model_id, seed), cwd=REPO_ROOT, env=env)
        running.append((process, model_id, seed, 0))
    while running:
        _reap()
        if running:
            time.sleep(5)
    if failures:
        (OUTPUT / "job_failures.json").write_text(_json({"failures": failures}), encoding="utf-8")
    else:
        failure_path = OUTPUT / "job_failures.json"
        if failure_path.is_file():
            failure_path.unlink()
    subprocess.run(
        [str(PYTHON), str(REPO_ROOT / "scripts/score_binance_confirm.py")],
        cwd=REPO_ROOT,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.job:
        try:
            run_job(args.job, args.seed)
        except Exception:
            _log(traceback.format_exc())
            raise
        return
    if args.build_only:
        print(json.dumps(build_inputs(), indent=2))
        return
    if args.launch:
        launch_jobs()
        return
    raise SystemExit("pass --job, --build-only, or --launch")


if __name__ == "__main__":
    main()
