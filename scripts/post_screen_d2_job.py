"""One D2 seed. VALIDATION then SCREEN. This process does not run git.

The variant and seed come from the command line. Threads are pinned to one.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch

from covharness.models.lstm_bekk import ARCHITECTURE_RANGE_POST_SCREEN_D1
from covharness.models.lstm_bekk_variants import realized_feature_matrix, variant_class
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
SPEC_COMMIT = "60b240bae32577e0159df8cc494019b3db4935d4"
PANEL = REPO_ROOT / "data/processed/binance_five_asset_panel.npz"
CORE = REPO_ROOT / "configs/binance_open_data_core.yaml"
SCREEN_CONFIG = REPO_ROOT / "configs/binance_open_data_screen.yaml"
SEMI = REPO_ROOT / "results/post_screen/C4/semivariance_panel.npz"
LSTM22 = {
    "id": "LSTM22",
    "num_layers": 7,
    "dropout": 0.2,
    "max_epochs": 75,
    "learning_rate": 0.001,
    "gradient_clip_norm": 1.0,
    "architecture_range": ARCHITECTURE_RANGE_POST_SCREEN_D1,
}


def _log(directory: Path, line: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _align(dates: tuple, panel_dates: np.ndarray, values: np.ndarray) -> np.ndarray:
    lookup = {str(day): index for index, day in enumerate(panel_dates)}
    rows = [lookup[day.isoformat()] for day in dates]
    return values[rows]


def _features_for_view(view, semi_dates: np.ndarray, positive: np.ndarray, negative: np.ndarray):
    # Align day t features to this view. The target row is not read by the runner.
    return realized_feature_matrix(
        _align(view.dates, semi_dates, positive),
        _align(view.dates, semi_dates, negative),
        view.rcov,
        view.rq,
    )


def _run_period(period, view, schedule, runner, model_cls, seed, directory, features) -> dict[str, object]:
    def factory(_family, row, seed_value):
        return model_cls(
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
            _log(directory, f"D2 {period} {origin.isoformat()} seed {seed} refit")

    result = runner(
        family=model_cls.variant_id,
        row=LSTM22,
        seed=seed,
        view=view,
        schedule=schedule,
        checkpoint_dir=directory / f"{period.lower()}_checkpoints",
        model_factory=factory,
        on_origin=on_origin,
        feature_panel=features,
        proxy_panel=view.rcov,
    )
    _log(directory, f"D2 {period} seed {seed} done")
    return {
        "mean_qlike": result.mean_qlike,
        "mean_frobenius": result.mean_frobenius,
        "complete_support": result.complete_support,
        "repair_count": result.repair_count,
        "n_targets": result.n_targets,
        "runtime_seconds": result.runtime_seconds,
        "failure_reason": result.failure_reason,
    }


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("V10", "V01", "V11"))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    directory = REPO_ROOT / "results/post_screen/D2" / args.variant / f"seed_{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    _log(directory, f"D2 {args.variant} seed {args.seed} start")
    try:
        semi = np.load(SEMI)
        protocol = build_binance_segmented_protocol()
        validation = load_validation_view(PANEL, CORE, protocol)
        screen = load_screen_view(PANEL, CORE, SCREEN_CONFIG, protocol)
        model_cls = variant_class(args.variant)
        validation_features = _features_for_view(
            validation, semi["dates"], semi["rs_positive"], semi["rs_negative"]
        )
        screen_features = _features_for_view(
            screen, semi["dates"], semi["rs_positive"], semi["rs_negative"]
        )
        validation_row = _run_period(
            "VALIDATION",
            validation,
            request_validation_schedule(protocol),
            run_one_configuration,
            model_cls,
            args.seed,
            directory,
            validation_features,
        )
        screen_row = _run_period(
            "SCREEN",
            screen,
            request_screen_schedule(protocol),
            run_one_screen_configuration,
            model_cls,
            args.seed,
            directory,
            screen_features,
        )
        payload = {
            "task": "D2",
            "variant": args.variant,
            "seed": args.seed,
            "spec_commit": SPEC_COMMIT,
            "validation": validation_row,
            "screen": screen_row,
        }
        (directory / "summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        _log(directory, f"D2 {args.variant} seed {args.seed} finished")
    except Exception:
        _log(directory, f"D2 {args.variant} seed {args.seed} failed")
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
