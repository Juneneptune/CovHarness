"""Descriptive SCREEN loss-differential persistence and refit-cycle diagnostics.

Task A1. Spec commit must precede this run. The script does not unlock
CONFIRM, does not fit models, and does not run Diebold-Mariano, SPA, or MCS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from covharness.inference import (
    ar1_effective_sample_size,
    equal_refit_phase_means_wald,
    hac_long_run_variance,
    loss_differential,
    loss_differential_diagnostics,
    newey_west_1994_lags,
    refit_cycle_means,
    refit_phase_from_flags,
    refit_phase_index,
    sample_acf,
)
from covharness.inference.exceptions import DegenerateLossDifferentialError

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "80612248ae6f772262c4eadc8b48c6a12a9887c0"
SCREEN_END = "2025-04-12"
SCREEN_START = "2023-11-30"
N_SCREEN = 500
CADENCE = 21
ACF_LAGS = 30
WALD_LAG = 29
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
        "role": "SCREEN headline comparison",
    },
    {
        "pair_id": "quarticity",
        "config_a": "HARQDRD01",
        "config_b": "HARDRD01",
        "label_a": "HARQ-DRD",
        "label_b": "HAR-DRD",
        "role": "realized-quarticity ladder",
    },
    {
        "pair_id": "nonlinear",
        "config_a": "HARDRD01",
        "config_b": "XGB08",
        "label_a": "HAR-DRD",
        "label_b": "XGBoost-DRD",
        "role": "nonlinear HAR-DRD ladder",
    },
    {
        "pair_id": "qlike_mcs",
        "config_a": "HARQDRD01",
        "config_b": "EWMA01",
        "label_a": "HARQ-DRD",
        "label_b": "EWMA",
        "role": "primary reduced-QLIKE MCS pair",
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
    if len(dates) != N_SCREEN:
        raise RuntimeError("SCREEN loss length is not 500")
    labels = [str(value) for value in blob["labels"].tolist()]
    lookup = {label: index for index, label in enumerate(labels)}
    losses = np.asarray(blob["losses"], dtype=float)
    return losses, lookup, dates


def bartlett_lags(n_obs: int) -> dict[str, int]:
    automatic = newey_west_1994_lags(n_obs)
    sqrt_rule = int(np.floor(1.3 * np.sqrt(n_obs)))
    return {
        "nw94_auto": automatic,
        "nw94_auto_x2": 2 * automatic,
        "nw94_auto_x4": 4 * automatic,
        "sqrt_1_3": sqrt_rule,
    }


def lrv_or_none(series: np.ndarray, maxlags: int) -> float | None:
    try:
        return float(hac_long_run_variance(series, maxlags=maxlags).long_run_variance)
    except DegenerateLossDifferentialError:
        return None


def pair_row(
    pair: dict[str, str],
    loss_name: str,
    series: np.ndarray,
    phase: np.ndarray,
    lag_map: dict[str, int],
) -> dict[str, object]:
    acf = sample_acf(series, ACF_LAGS)
    rho_hat = float(acf[0])
    diagnostic = loss_differential_diagnostics(series, acf_lags=ACF_LAGS, maxlags=None)
    cycle = refit_cycle_means(series, phase, cadence=CADENCE)
    wald = equal_refit_phase_means_wald(series, phase, maxlags=WALD_LAG)
    lrvs = {name: lrv_or_none(series, lag) for name, lag in lag_map.items()}
    return {
        "pair_id": pair["pair_id"],
        "config_a": pair["config_a"],
        "config_b": pair["config_b"],
        "label_a": pair["label_a"],
        "label_b": pair["label_b"],
        "role": pair["role"],
        "loss": loss_name,
        "n_observations": int(series.size),
        "mean_differential": float(series.mean()),
        "rho_hat": rho_hat,
        "t_eff_ar1": ar1_effective_sample_size(int(series.size), rho_hat),
        "kappa": diagnostic.kappa,
        "t_eff_kappa": diagnostic.t_eff,
        "kappa_hac_maxlags": int(diagnostic.hac_maxlags),
        "acf": [float(value) for value in acf],
        "lrv": {
            name: {"lag": int(lag_map[name]), "long_run_variance": lrvs[name]}
            for name in lag_map
        },
        "refit_cycle": {
            "cadence": CADENCE,
            "n_complete_cycles": int(cycle.n_complete_cycles),
            "final_cycle_length": int(cycle.final_cycle_length),
            "occupied_phases": [int(value) for value in cycle.occupied_phases],
            "counts": [int(value) for value in cycle.counts],
            "means": [float(value) for value in cycle.means],
        },
        "refit_wald": {
            "statistic": wald.statistic,
            "df": wald.df,
            "p_value": wald.p_value,
            "hac_maxlags": int(wald.hac_maxlags),
            "n_phases": int(wald.n_phases),
            "notes": wald.notes,
        },
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    header = [
        "pair_id",
        "loss",
        "config_a",
        "config_b",
        "rho_hat",
        "t_eff_ar1",
        "t_eff_kappa",
        "kappa",
        "lrv_nw94_auto",
        "lrv_nw94_auto_x2",
        "lrv_nw94_auto_x4",
        "lrv_sqrt_1_3",
        "wald_p_value",
        "wald_statistic",
        "wald_df",
    ]
    lines = [",".join(header)]
    for row in rows:
        lrv = row["lrv"]
        wald = row["refit_wald"]
        values = [
            row["pair_id"],
            row["loss"],
            row["config_a"],
            row["config_b"],
            f"{row['rho_hat']:.12g}",
            "" if row["t_eff_ar1"] is None else f"{row['t_eff_ar1']:.12g}",
            "" if row["t_eff_kappa"] is None else f"{row['t_eff_kappa']:.12g}",
            "" if row["kappa"] is None else f"{row['kappa']:.12g}",
            "" if lrv["nw94_auto"]["long_run_variance"] is None else f"{lrv['nw94_auto']['long_run_variance']:.12g}",
            "" if lrv["nw94_auto_x2"]["long_run_variance"] is None else f"{lrv['nw94_auto_x2']['long_run_variance']:.12g}",
            "" if lrv["nw94_auto_x4"]["long_run_variance"] is None else f"{lrv['nw94_auto_x4']['long_run_variance']:.12g}",
            "" if lrv["sqrt_1_3"]["long_run_variance"] is None else f"{lrv['sqrt_1_3']['long_run_variance']:.12g}",
            "" if wald["p_value"] is None else f"{wald['p_value']:.12g}",
            "" if wald["statistic"] is None else f"{wald['statistic']:.12g}",
            "" if wald["df"] is None else str(wald["df"]),
        ]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cycle_csv(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["pair_id,loss,k,count,mean_d"]
    for row in rows:
        cycle = row["refit_cycle"]
        for k, count, mean in zip(
            cycle["occupied_phases"], cycle["counts"], cycle["means"]
        ):
            lines.append(
                f"{row['pair_id']},{row['loss']},{k},{count},{mean:.12g}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(output_dir: Path, rows: list[dict[str, object]], n_obs: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iid_band = 1.96 / np.sqrt(n_obs)
    loss_titles = {
        "reduced_qlike": "reduced QLIKE",
        "squared_frobenius": "squared Frobenius",
    }
    for loss_name, title in loss_titles.items():
        subset = [row for row in rows if row["loss"] == loss_name]
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True, sharey=False)
        for axis, row in zip(axes.ravel(), subset):
            lags = np.arange(1, ACF_LAGS + 1)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.axhline(iid_band, color="0.6", linewidth=0.8, linestyle="--")
            axis.axhline(-iid_band, color="0.6", linewidth=0.8, linestyle="--")
            axis.vlines(lags, 0.0, row["acf"], color="C0", linewidth=1.2)
            axis.plot(lags, row["acf"], "o", color="C0", markersize=3.5)
            axis.set_title(f"{row['config_a']} minus {row['config_b']}")
            axis.set_xlabel("lag")
            axis.set_ylabel("ACF")
        fig.suptitle(f"SCREEN loss-differential ACF, {title}")
        fig.tight_layout()
        fig.savefig(output_dir / f"acf_{loss_name}.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True)
        for axis, row in zip(axes.ravel(), subset):
            cycle = row["refit_cycle"]
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.plot(cycle["occupied_phases"], cycle["means"], "o-", color="C1")
            axis.set_title(f"{row['config_a']} minus {row['config_b']}")
            axis.set_xlabel("days since last refit, k")
            axis.set_ylabel("mean d_t")
        fig.suptitle(f"SCREEN mean loss differential by refit phase, {title}")
        fig.tight_layout()
        fig.savefig(output_dir / f"refit_means_{loss_name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="A1 SCREEN loss-differential persistence")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "post_screen" / "A1",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    qlike_path = REPO_ROOT / LOSSES[0][1]
    frob_path = REPO_ROOT / LOSSES[1][1]
    checkpoint_path = (
        REPO_ROOT / "results" / "binance_screen_checkpoints" / "HARQ-DRD_HARQDRD01.npz"
    )
    # Refuse to proceed if the frozen 2026-09-21 SCREEN artifacts have changed.
    require_hash(qlike_path, QLIKE_LOSS_SHA256)
    require_hash(frob_path, FROBENIUS_LOSS_SHA256)
    require_hash(checkpoint_path, HARQ_CHECKPOINT_SHA256)

    qlike_losses, qlike_lookup, dates = load_loss_panel(qlike_path)
    frob_losses, frob_lookup, frob_dates = load_loss_panel(frob_path)
    if dates != frob_dates:
        raise RuntimeError("QLIKE and Frobenius SCREEN dates differ")

    checkpoint = np.load(checkpoint_path)
    refit_flags = np.asarray(checkpoint["refit"], dtype=bool)
    # Rebuild k from stored SCREEN refit flags and from the 21-origin rule.
    phase_from_flags = refit_phase_from_flags(refit_flags)
    phase = refit_phase_index(N_SCREEN, cadence=CADENCE, first_is_refit=True)
    if not np.array_equal(phase, phase_from_flags):
        raise RuntimeError("SCREEN refit flags do not match the 21-origin cadence")

    lag_map = bartlett_lags(N_SCREEN)
    if lag_map["nw94_auto"] != 5 or lag_map["sqrt_1_3"] != WALD_LAG:
        raise RuntimeError("frozen A1 lag map drifted")

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
            if not np.isfinite(series).all():
                raise RuntimeError(
                    f"non-finite differential for {pair['pair_id']} {loss_name}"
                )
            rows.append(pair_row(pair, loss_name, series, phase, lag_map))

    payload = {
        "task": "A1",
        "spec_commit": SPEC_COMMIT,
        "descriptive_only": True,
        "confirm_locked": True,
        "n_observations": N_SCREEN,
        "date_start": SCREEN_START,
        "date_end": SCREEN_END,
        "acf_lags": ACF_LAGS,
        "refit_cadence": CADENCE,
        "wald_hac_lag": WALD_LAG,
        "lags": lag_map,
        "loss_hashes": {
            "qlike": QLIKE_LOSS_SHA256,
            "frobenius": FROBENIUS_LOSS_SHA256,
        },
        "results": rows,
    }
    (output_dir / "a1_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "a1_summary.csv", rows)
    write_cycle_csv(output_dir / "a1_refit_cycle_means.csv", rows)
    write_plots(output_dir, rows, N_SCREEN)


if __name__ == "__main__":
    main()
