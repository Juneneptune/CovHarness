"""SCREEN SPA and MCS at block lengths 7, 14, and 30.

Task B2. Length 7 must match the frozen 2026-09-21 p-values. Those files
are not rewritten. The CONFIRM block length is not chosen here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from covharness.protocol.binance_screen import screen_mcs, screen_spa

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "8d0cbfb7184290f7db819e56c45e0da3548c831c"
BLOCK_LENGTHS = (7, 14, 30)
N_BOOT = 5000
SEED = 20260913
SCREEN_START = "2023-11-30"
SCREEN_END = "2025-04-12"
FROZEN = {
    "spa": REPO_ROOT / "results" / "binance_screen_spa.json",
    "mcs_qlike": REPO_ROOT / "results" / "binance_screen_mcs_qlike.json",
    "mcs_frobenius": REPO_ROOT / "results" / "binance_screen_mcs_frobenius.json",
    "subblock": REPO_ROOT / "results" / "binance_screen_subblock_ranks.csv",
}
LOSS_PATHS = {
    "qlike": REPO_ROOT / "results" / "binance_screen_losses_qlike.npz",
    "frobenius": REPO_ROOT / "results" / "binance_screen_losses_frobenius.npz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_loss_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    blob = np.load(path)
    labels = [str(value) for value in blob["labels"].tolist()]
    complete = [str(value) for value in blob["complete_labels"].tolist()]
    dates = [str(value) for value in blob["dates"].tolist()]
    if dates[0] != SCREEN_START or dates[-1] != SCREEN_END or len(dates) != 500:
        raise RuntimeError(f"{path.name} is not the 500 SCREEN dates")
    if any(date > SCREEN_END for date in dates):
        raise RuntimeError(f"{path.name} contains a date after 2025-04-12")
    index = [labels.index(name) for name in complete]
    losses = np.asarray(blob["losses"][:, index], dtype=float)
    if losses.shape != (500, 9):
        raise RuntimeError(f"{path.name} complete block is not (500, 9)")
    if not np.isfinite(losses).all():
        raise RuntimeError(f"{path.name} complete columns are not finite")
    return losses, complete


def spa_matches(saved: dict[str, object], fresh: dict[str, object]) -> list[str]:
    mismatches = []
    for key in (
        "p_value_lower",
        "p_value_consistent",
        "p_value_upper",
        "statistic",
        "block_length",
        "n_boot",
        "seed",
        "benchmark",
        "exact_benchmark_ties",
        "alternative_labels",
    ):
        if saved[key] != fresh[key]:
            mismatches.append(key)
    return mismatches


def mcs_matches(saved: dict[str, object], fresh: dict[str, object]) -> list[str]:
    mismatches = []
    for arm in ("primary", "companion"):
        left = saved[arm]
        right = fresh[arm]
        assert isinstance(left, dict) and isinstance(right, dict)
        for key in ("procedure", "block_length", "mcs_pvalues", "in_mcs_primary_alpha", "labels"):
            if left[key] != right[key]:
                mismatches.append(f"{arm}.{key}")
    if saved["primary"]["procedure"] == saved["companion"]["procedure"]:
        mismatches.append("procedures_crossed")
    return mismatches


def rows_from_spa(loss_name: str, result: dict[str, object]) -> dict[str, object]:
    return {
        "loss": loss_name,
        "block_length": result["block_length"],
        "p_value_lower": result["p_value_lower"],
        "p_value_consistent": result["p_value_consistent"],
        "p_value_upper": result["p_value_upper"],
        "statistic": result["statistic"],
        "exact_benchmark_ties": " ".join(result["exact_benchmark_ties"]),
    }


def rows_from_mcs(loss_name: str, result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for arm in ("primary", "companion"):
        payload = result[arm]
        assert isinstance(payload, dict)
        for label in payload["labels"]:
            rows.append(
                {
                    "loss": loss_name,
                    "block_length": payload["block_length"],
                    "arm": arm,
                    "procedure": payload["procedure"],
                    "model": label,
                    "mcs_pvalue": payload["mcs_pvalues"][label],
                    "in_mcs_0.10": payload["in_mcs_primary_alpha"][label],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 SCREEN block-length sensitivity")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "post_screen" / "B2",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_before = {name: sha256_file(path) for name, path in FROZEN.items()}
    saved_spa = json.loads(FROZEN["spa"].read_text(encoding="utf-8"))
    saved_mcs = {
        "qlike": json.loads(FROZEN["mcs_qlike"].read_text(encoding="utf-8")),
        "frobenius": json.loads(FROZEN["mcs_frobenius"].read_text(encoding="utf-8")),
    }
    spa_rows = []
    mcs_rows = []
    fresh: dict[str, dict[int, dict[str, object]]] = {}
    for loss_name, path in LOSS_PATHS.items():
        losses, labels = complete_loss_matrix(path)
        fresh[loss_name] = {}
        for length in BLOCK_LENGTHS:
            print(f"B2 start {loss_name} block_length={length}", flush=True)
            spa = screen_spa(
                losses, labels, n_boot=N_BOOT, seed=SEED, block_length=length
            )
            mcs = screen_mcs(
                losses, labels, n_boot=N_BOOT, seed=SEED, block_length=length
            )
            print(f"B2 done {loss_name} block_length={length}", flush=True)
            if length == 7:
                spa_bad = spa_matches(saved_spa[loss_name], spa)
                mcs_bad = mcs_matches(saved_mcs[loss_name], mcs)
                if spa_bad or mcs_bad:
                    failure = {
                        "task": "B2",
                        "status": "failed",
                        "loss": loss_name,
                        "spa_mismatches": spa_bad,
                        "mcs_mismatches": mcs_bad,
                    }
                    output_dir.joinpath("b2_failure.json").write_text(
                        json.dumps(failure, indent=2) + "\n", encoding="utf-8"
                    )
                    raise RuntimeError(
                        f"block length 7 did not reproduce saved {loss_name} p-values"
                    )
            fresh[loss_name][length] = {"spa": spa, "mcs": mcs}
            spa_rows.append(rows_from_spa(loss_name, spa))
            mcs_rows.extend(rows_from_mcs(loss_name, mcs))
    frozen_after = {name: sha256_file(path) for name, path in FROZEN.items()}
    if frozen_before != frozen_after:
        raise RuntimeError("a frozen 2026-09-21 artifact changed during B2")
    payload = {
        "task": "B2",
        "label": "SCREEN SPA and MCS block-length sensitivity",
        "spec_commit": SPEC_COMMIT,
        "block_lengths": list(BLOCK_LENGTHS),
        "n_boot": N_BOOT,
        "seed": SEED,
        "benchmark": "HAR-DRD",
        "primary_recentering": "consistent",
        "mcs_primary": "range",
        "mcs_companion": "max",
        "length_7_matches_frozen": True,
        "confirm_block_length": "not chosen; decided at G1 and tied to B1",
        "finalist_rule_changed": False,
        "frozen_sha256_before": frozen_before,
        "frozen_sha256_after": frozen_after,
        "spa": {
            loss: {str(length): result["spa"] for length, result in by_length.items()}
            for loss, by_length in fresh.items()
        },
        "mcs": {
            loss: {str(length): result["mcs"] for length, result in by_length.items()}
            for loss, by_length in fresh.items()
        },
    }
    output_dir.joinpath("b2_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "b2_spa.csv", spa_rows)
    write_csv(output_dir / "b2_mcs.csv", mcs_rows)


if __name__ == "__main__":
    main()
