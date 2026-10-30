"""VALIDATION and SCREEN for SHAR-DRD. CONFIRM is not read.

Task C4. The production panel is not rewritten. No SPA or MCS is run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from covharness.models.shar_drd import SHARDRDRealizedCovariance
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
from covharness.realized.binance_panel import build_preconfirm_semivariances

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "PENDING"
PANEL = REPO_ROOT / "data/processed/binance_five_asset_panel.npz"
CORE = REPO_ROOT / "configs/binance_open_data_core.yaml"
SCREEN_CONFIG = REPO_ROOT / "configs/binance_open_data_screen.yaml"
CACHE = REPO_ROOT / "data/cache/open_data/binance/spot/monthly/klines/1m"
OUTPUT = REPO_ROOT / "results/post_screen/C4"
SEMI_PATH = OUTPUT / "semivariance_panel.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log(period: str, origin: str, detail: str) -> None:
    path = OUTPUT / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"C4 {period} {origin} SHARDRD01 {detail}\n")


def _align(dates: tuple, panel_dates: np.ndarray, positive, negative) -> tuple[np.ndarray, np.ndarray]:
    lookup = {str(day): index for index, day in enumerate(panel_dates)}
    rows = [lookup[day.isoformat()] for day in dates]
    return positive[rows], negative[rows]


def _build_semivariances() -> dict[str, object]:
    if SEMI_PATH.is_file():
        blob = np.load(SEMI_PATH)
        positive = np.array(blob["rs_positive"], dtype=float)
        negative = np.array(blob["rs_negative"], dtype=float)
        variance = np.array(blob["rv_same_returns"], dtype=float)
        dates = [str(value) for value in blob["dates"].tolist()]
    else:
        date_objs, positive, negative, variance = build_preconfirm_semivariances(CACHE)
        dates = [day.isoformat() for day in date_objs]
        np.savez_compressed(
            SEMI_PATH,
            dates=np.array(dates),
            rs_positive=positive,
            rs_negative=negative,
            rv_same_returns=variance,
        )
    bitwise_same = bool(np.array_equal(positive + negative, variance))
    panel = np.load(PANEL)
    panel_dates = [str(value) for value in panel["dates"].tolist()]
    lookup = {day: index for index, day in enumerate(panel_dates)}
    stored = np.diagonal(panel["rcov_5min"][[lookup[day] for day in dates]], axis1=1, axis2=2)
    gap = positive + negative - stored
    return {
        "dates": np.array(dates),
        "rs_positive": positive,
        "rs_negative": negative,
        "rv_same_returns": variance,
        "bitwise_same_return_rv": bitwise_same,
        "max_abs_versus_production_rv": float(np.max(np.abs(gap))),
        "bitwise_production_rv": bool(np.array_equal(positive + negative, stored)),
        "sha256": _sha256(SEMI_PATH),
    }


def _run(period: str, view, schedule, runner, positive, negative) -> dict[str, object]:
    aligned_pos, aligned_neg = _align(view.dates, _build_cache["dates"], positive, negative)

    def factory(_family, _row, _seed):
        return SHARDRDRealizedCovariance()

    def on_origin(origin, refit, model) -> None:
        if refit:
            _log(period, origin.isoformat(), "refit")

    result = runner(
        family="SHAR-DRD",
        row={"id": "SHARDRD01"},
        seed=None,
        view=view,
        schedule=schedule,
        checkpoint_dir=OUTPUT / f"{period.lower()}_checkpoints",
        model_factory=factory,
        on_origin=on_origin,
        semivariance_positive=aligned_pos,
        semivariance_negative=aligned_neg,
    )
    return {
        "mean_qlike": result.mean_qlike,
        "mean_frobenius": result.mean_frobenius,
        "complete_support": result.complete_support,
        "repair_count": result.repair_count,
        "n_targets": result.n_targets,
        "runtime_seconds": result.runtime_seconds,
        "failure_reason": result.failure_reason,
    }


_build_cache: dict[str, object] = {}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    built = _build_semivariances()
    _build_cache["dates"] = built["dates"]
    protocol = build_binance_segmented_protocol()
    validation = load_validation_view(PANEL, CORE, protocol)
    screen = load_screen_view(PANEL, CORE, SCREEN_CONFIG, protocol)
    validation_row = _run(
        "VALIDATION",
        validation,
        request_validation_schedule(protocol),
        run_one_configuration,
        built["rs_positive"],
        built["rs_negative"],
    )
    screen_row = _run(
        "SCREEN",
        screen,
        request_screen_schedule(protocol),
        run_one_screen_configuration,
        built["rs_positive"],
        built["rs_negative"],
    )
    payload = {
        "task": "C4",
        "spec_commit": SPEC_COMMIT,
        "semivariance_sha256": built["sha256"],
        "bitwise_same_return_rv": built["bitwise_same_return_rv"],
        "bitwise_production_rv": built["bitwise_production_rv"],
        "max_abs_versus_production_rv": built["max_abs_versus_production_rv"],
        "validation": validation_row,
        "screen": screen_row,
    }
    (OUTPUT / "c4_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("C4", validation_row["mean_qlike"], screen_row["mean_qlike"], built["bitwise_production_rv"], flush=True)


if __name__ == "__main__":
    main()
