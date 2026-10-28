"""Descriptive A1-addendum checks of the SCREEN refit-phase Wald.

The script does not rewrite A1 committed files, does not unlock CONFIRM,
and does not run Diebold-Mariano.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from covharness.inference import (
    equal_refit_phase_means_wald,
    intercept_slope_wald,
    largest_absolute_indices,
    loss_differential,
    refit_phase_from_flags,
    refit_phase_index,
    roll_refit_phase,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "0187bb1a5921bba53152f2610343cb0c8b4aaeb0"
A1_RESULT_COMMIT = "084e3915656be4d326ac242fc8bc709d86e335b3"
A1_SUMMARY_SHA256 = "58e4484211573c071435d38e79c77c9d35056f8999de667e57321f0bc5805cb9"
SCREEN_END = "2025-04-12"
SCREEN_START = "2023-11-30"
N_SCREEN = 500
CADENCE = 21
WALD_LAG = 29
CONTRAST_LAGS = (5, 29)
N_INFLUENCE = 10
QLIKE_LOSS_SHA256 = "3fa68921871277b52e60412bae9fb23f31e7b26e39bec446d1051f7f6e8af61d"
FROBENIUS_LOSS_SHA256 = "63a4de10db499b5527e5f1c276c17460481d85a7c3699a7a6226a97819729713"
HARQ_CHECKPOINT_SHA256 = "d2232e6e3e6aa013ff705ff4519312c3dd72cdfcd0e34eed9d50b714eaa0cc14"

PAIRS = (
    {
        "pair_id": "headline",
        "config_a": "HARQDRD01",
        "config_b": "LSTM19",
        "label_a": "HARQ-DRD",
        "label_b": "LSTM-BEKK",
    },
    {
        "pair_id": "quarticity",
        "config_a": "HARQDRD01",
        "config_b": "HARDRD01",
        "label_a": "HARQ-DRD",
        "label_b": "HAR-DRD",
    },
    {
        "pair_id": "nonlinear",
        "config_a": "HARDRD01",
        "config_b": "XGB08",
        "label_a": "HAR-DRD",
        "label_b": "XGBoost-DRD",
    },
    {
        "pair_id": "qlike_mcs",
        "config_a": "HARQDRD01",
        "config_b": "EWMA01",
        "label_a": "HARQ-DRD",
        "label_b": "EWMA",
    },
)

LOSSES = (
    ("reduced_qlike", "results/binance_screen_losses_qlike.npz", QLIKE_LOSS_SHA256),
    (
        "squared_frobenius",
        "results/binance_screen_losses_frobenius.npz",
        FROBENIUS_LOSS_SHA256,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise RuntimeError(f"hash mismatch for {path}: {digest} != {expected}")


def load_loss_panel(path: Path) -> tuple[np.ndarray, dict[str, int], tuple[str, ...]]:
    blob = np.load(path)
    dates = tuple(str(value) for value in blob["dates"].tolist())
    if dates[0] != SCREEN_START or dates[-1] != SCREEN_END:
        raise RuntimeError("SCREEN loss dates drifted")
    if any(date > SCREEN_END for date in dates):
        raise RuntimeError("a SCREEN loss date is after 2025-04-12")
    labels = [str(value) for value in blob["labels"].tolist()]
    lookup = {label: index for index, label in enumerate(labels)}
    return np.asarray(blob["losses"], dtype=float), lookup, dates


def wald_payload(result) -> dict[str, object]:
    return {
        "statistic": result.statistic,
        "df": result.df,
        "p_value": result.p_value,
        "hac_maxlags": int(result.hac_maxlags),
        "notes": result.notes,
    }


def slope_payload(result) -> dict[str, object]:
    return {
        "intercept": result.intercept,
        "slope": result.slope,
        "statistic": result.statistic,
        "df": result.df,
        "p_value": result.p_value,
        "hac_maxlags": int(result.hac_maxlags),
        "notes": result.notes,
    }


def placebo_summary(statistics: list[float | None], p_values: list[float | None]) -> dict[str, object]:
    true_stat = statistics[0]
    comparable = [float(value) for value in statistics if value is not None]
    n_reject = sum(p is not None and p < 0.05 for p in p_values)
    if true_stat is None or not comparable:
        rank = None
    else:
        rank = 1 + sum(value > true_stat for value in comparable)
    return {
        "true_statistic": true_stat,
        "true_p_value": p_values[0],
        "rank_from_largest": rank,
        "n_shifts": len(statistics),
        "n_p_below_0_05": int(n_reject),
        "max_statistic": max(comparable) if comparable else None,
        "min_statistic": min(comparable) if comparable else None,
    }


def pair_check(
    pair: dict[str, str],
    loss_name: str,
    series: np.ndarray,
    phase: np.ndarray,
    dates: tuple[str, ...],
    committed_wald: dict[str, object] | None,
) -> dict[str, object]:
    true = equal_refit_phase_means_wald(series, phase, maxlags=WALD_LAG)
    if committed_wald is not None and true.statistic is not None:
        committed_stat = committed_wald["statistic"]
        if abs(float(true.statistic) - float(committed_stat)) > 1e-10:
            raise RuntimeError(
                f"A1 Wald statistic drifted for {pair['pair_id']} {loss_name}"
            )

    # Time-roll the phase calendar. d_t is left unchanged.
    shifts: list[dict[str, object]] = []
    stats: list[float | None] = []
    pvals: list[float | None] = []
    for shift in range(CADENCE):
        rolled = equal_refit_phase_means_wald(
            series, roll_refit_phase(phase, shift), maxlags=WALD_LAG
        )
        payload = {"shift": shift, **wald_payload(rolled)}
        shifts.append(payload)
        stats.append(rolled.statistic)
        pvals.append(rolled.p_value)

    dummy = (phase == 0).astype(float)
    k_linear = phase.astype(float)
    contrasts = {}
    for lag in CONTRAST_LAGS:
        contrasts[str(lag)] = {
            "fresh_refit": slope_payload(
                intercept_slope_wald(series, dummy, maxlags=lag)
            ),
            "linear_staleness": slope_payload(
                intercept_slope_wald(series, k_linear, maxlags=lag)
            ),
        }

    drop = largest_absolute_indices(series, N_INFLUENCE)
    keep = np.ones(series.size, dtype=bool)
    keep[drop] = False
    influence_days = []
    for index in drop:
        influence_days.append(
            {
                "position": int(index),
                "date": dates[int(index)],
                "k": int(phase[int(index)]),
                "d_t": float(series[int(index)]),
            }
        )
    dropped = equal_refit_phase_means_wald(series[keep], phase[keep], maxlags=WALD_LAG)

    return {
        "pair_id": pair["pair_id"],
        "config_a": pair["config_a"],
        "config_b": pair["config_b"],
        "loss": loss_name,
        "true_wald": wald_payload(true),
        "placebo": {
            **placebo_summary(stats, pvals),
            "shifts": shifts,
        },
        "contrasts": contrasts,
        "influence": {
            "n_dropped": N_INFLUENCE,
            "days": influence_days,
            "wald_without_ten": wald_payload(dropped),
            "notes": (
                "HAC on the remaining dates treats them as contiguous. "
                "This is a sensitivity, not a cleaning rule."
            ),
        },
    }


def committed_wald_lookup(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = {}
    for row in payload["results"]:
        table[(row["pair_id"], row["loss"])] = row["refit_wald"]
    return table


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    header = [
        "pair_id",
        "loss",
        "true_wald",
        "true_p",
        "placebo_rank_from_largest",
        "n_p_below_0_05",
        "fresh_refit_b5_slope",
        "fresh_refit_b5_p",
        "fresh_refit_b29_slope",
        "fresh_refit_b29_p",
        "linear_k_b5_slope",
        "linear_k_b5_p",
        "linear_k_b29_slope",
        "linear_k_b29_p",
        "wald_drop10",
        "wald_drop10_p",
    ]
    lines = [",".join(header)]
    for row in rows:
        placebo = row["placebo"]
        c5 = row["contrasts"]["5"]
        c29 = row["contrasts"]["29"]
        dropped = row["influence"]["wald_without_ten"]
        values = [
            row["pair_id"],
            row["loss"],
            _fmt(row["true_wald"]["statistic"]),
            _fmt(row["true_wald"]["p_value"]),
            "" if placebo["rank_from_largest"] is None else str(placebo["rank_from_largest"]),
            str(placebo["n_p_below_0_05"]),
            _fmt(c5["fresh_refit"]["slope"]),
            _fmt(c5["fresh_refit"]["p_value"]),
            _fmt(c29["fresh_refit"]["slope"]),
            _fmt(c29["fresh_refit"]["p_value"]),
            _fmt(c5["linear_staleness"]["slope"]),
            _fmt(c5["linear_staleness"]["p_value"]),
            _fmt(c29["linear_staleness"]["slope"]),
            _fmt(c29["linear_staleness"]["p_value"]),
            _fmt(dropped["statistic"]),
            _fmt(dropped["p_value"]),
        ]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.12g}"


def write_placebo_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(9.5, 11.0), sharex=True)
    by_pair = {pair["pair_id"]: pair_idx for pair_idx, pair in enumerate(PAIRS)}
    loss_col = {"reduced_qlike": 0, "squared_frobenius": 1}
    for row in rows:
        axis = axes[by_pair[row["pair_id"]], loss_col[row["loss"]]]
        shifts = [item["shift"] for item in row["placebo"]["shifts"]]
        stats = [item["statistic"] for item in row["placebo"]["shifts"]]
        axis.plot(shifts, stats, "o-", color="C0", markersize=3.5)
        axis.plot(0, stats[0], "o", color="C3", markersize=7)
        axis.set_title(f"{row['pair_id']}, {row['loss']}")
        axis.set_ylabel("Wald")
    axes[-1, 0].set_xlabel("time-roll r")
    axes[-1, 1].set_xlabel("time-roll r")
    fig.suptitle("SCREEN refit-phase Wald under time-rolled phase labels")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="A1-addendum SCREEN refit Wald checks")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "post_screen" / "A1",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    a1_summary = output_dir / "a1_summary.json"
    # Leave committed A1 files untouched and refuse to proceed if they changed.
    require_hash(a1_summary, A1_SUMMARY_SHA256)
    committed = committed_wald_lookup(a1_summary)

    qlike_path = REPO_ROOT / LOSSES[0][1]
    frob_path = REPO_ROOT / LOSSES[1][1]
    checkpoint_path = (
        REPO_ROOT / "results" / "binance_screen_checkpoints" / "HARQ-DRD_HARQDRD01.npz"
    )
    require_hash(qlike_path, QLIKE_LOSS_SHA256)
    require_hash(frob_path, FROBENIUS_LOSS_SHA256)
    require_hash(checkpoint_path, HARQ_CHECKPOINT_SHA256)

    qlike_losses, qlike_lookup, dates = load_loss_panel(qlike_path)
    frob_losses, frob_lookup, frob_dates = load_loss_panel(frob_path)
    if dates != frob_dates:
        raise RuntimeError("QLIKE and Frobenius SCREEN dates differ")

    checkpoint = np.load(checkpoint_path)
    phase = refit_phase_index(N_SCREEN, cadence=CADENCE, first_is_refit=True)
    if not np.array_equal(
        phase, refit_phase_from_flags(np.asarray(checkpoint["refit"], dtype=bool))
    ):
        raise RuntimeError("SCREEN refit flags do not match the 21-origin cadence")

    panels = {
        "reduced_qlike": (qlike_losses, qlike_lookup),
        "squared_frobenius": (frob_losses, frob_lookup),
    }
    rows: list[dict[str, object]] = []
    for loss_name, (panel, lookup) in panels.items():
        for pair in PAIRS:
            series = loss_differential(
                panel[:, lookup[pair["label_a"]]],
                panel[:, lookup[pair["label_b"]]],
            )
            rows.append(
                pair_check(
                    pair,
                    loss_name,
                    series,
                    phase,
                    dates,
                    committed.get((pair["pair_id"], loss_name)),
                )
            )

    payload = {
        "task": "A1-addendum",
        "spec_commit": SPEC_COMMIT,
        "a1_result_commit": A1_RESULT_COMMIT,
        "descriptive_only": True,
        "confirm_locked": True,
        "a1_files_rewritten": False,
        "results": rows,
    }
    (output_dir / "a1_addendum_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "a1_addendum_summary.csv", rows)
    write_placebo_plot(output_dir / "a1_addendum_placebo_wald.png", rows)
    if sha256_file(a1_summary) != A1_SUMMARY_SHA256:
        raise RuntimeError("A1 summary hash changed during the addendum run")


if __name__ == "__main__":
    main()
