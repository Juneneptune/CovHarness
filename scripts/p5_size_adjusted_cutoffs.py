"""Freeze the size-adjusted P5 critical values on the B1 resampled null.

Task B1b. The null and the P5 statistic are the existing B1 objects.
CONFIRM is not read, and the cutoffs are not applied to a model comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from covharness.inference.dm import diebold_mariano
from covharness.inference.hac import ewc_nu
from covharness.inference.size import CALIBRATION_SEED

# Reuse the B1 path builder. The script directory is on sys.path when this file is the entry point.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dm_procedure_calibration import _cell_seed, _d3_paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "e4f01c0301320d5f92e0041e09069abc85ab2ba4"
OUTPUT = REPO_ROOT / "results" / "post_screen" / "B1b"
N_NULL = 20000
N_CHECK = 5000
N_BOOT_SE = 1000
BLOCKS = (7, 29)
QUANTILE = 0.95
T = 500


def _log(message: str) -> None:
    path = OUTPUT / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _absolute_statistics(paths: np.ndarray) -> np.ndarray:
    # Score each resampled gap with the existing equal-weighted cosine test.
    statistics = np.empty(paths.shape[0], dtype=float)
    for index, series in enumerate(paths):
        statistics[index] = abs(diebold_mariano(series, reference="ewc").statistic)
    return statistics


def _quantile_se(absolute: np.ndarray, seed: int) -> float:
    # Resample the null statistics and record the spread of the 0.95 quantile.
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT_SE, dtype=float)
    n_obs = absolute.shape[0]
    for index in range(N_BOOT_SE):
        sample = rng.choice(absolute, size=n_obs, replace=True)
        draws[index] = float(np.quantile(sample, QUANTILE))
    return float(np.std(draws, ddof=1))


def _one_block(block_length: int, ordinary: float) -> dict[str, object]:
    null_key = f"B1b-null-{block_length}"
    se_key = f"B1b-se-{block_length}"
    check_key = f"B1b-check-{block_length}"
    _log(f"B1b null block={block_length}")
    null_paths = _d3_paths(block_length, n_reps=N_NULL, seed_key=null_key)
    absolute = _absolute_statistics(null_paths)
    cutoff = float(np.quantile(absolute, QUANTILE))
    se_seed = _cell_seed(se_key)
    quantile_se = _quantile_se(absolute, se_seed)
    _log(f"B1b check block={block_length}")
    check_paths = _d3_paths(block_length, n_reps=N_CHECK, seed_key=check_key)
    check_abs = _absolute_statistics(check_paths)
    rate = float(np.mean(check_abs > cutoff))
    mcse = float(np.sqrt(0.05 * 0.95 / N_CHECK))
    return {
        "block_length": block_length,
        "n_null": N_NULL,
        "n_check": N_CHECK,
        "null_seed_key": null_key,
        "null_seed": _cell_seed(null_key),
        "se_seed_key": se_key,
        "se_seed": se_seed,
        "check_seed_key": check_key,
        "check_seed": _cell_seed(check_key),
        "calibration_seed": CALIBRATION_SEED,
        "quantile": QUANTILE,
        "quantile_method": "linear",
        "size_adjusted_critical_value": cutoff,
        "quantile_bootstrap_se": quantile_se,
        "n_bootstrap_se": N_BOOT_SE,
        "ordinary_critical_value": ordinary,
        "check_rejection_rate": rate,
        "check_mcse_at_nominal": mcse,
        "check_within_two_mcse": bool(abs(rate - 0.05) <= 2.0 * mcse),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nu = ewc_nu(T)
    ordinary = float(student_t.ppf(0.975, nu))
    rows = [_one_block(block, ordinary) for block in BLOCKS]
    payload = {
        "task": "B1b",
        "spec_commit": SPEC_COMMIT,
        "T": T,
        "ewc_nu": nu,
        "ordinary_critical_value": ordinary,
        "blocks": rows,
    }
    (OUTPUT / "b1b_cutoffs.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "B1b",
        rows[0]["size_adjusted_critical_value"],
        rows[1]["size_adjusted_critical_value"],
        rows[0]["check_within_two_mcse"],
        rows[1]["check_within_two_mcse"],
        flush=True,
    )


if __name__ == "__main__":
    main()
