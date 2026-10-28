"""Descriptive SCREEN pooled Mincer-Zarnowitz calibration.

Task A2. Spec commit must precede this run. The script does not unlock
CONFIRM, does not fit models, does not recalibrate forecasts, and does
not rank models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from covharness.inference import (
    WEIGHTING_APPROXIMATE_PS21,
    WEIGHTING_NONE,
    hac_long_run_variance,
    mincer_zarnowitz_pooled_vech,
    newey_west_1994_lags,
)
from covharness.inference.exceptions import DegenerateLossDifferentialError
from covharness.inference.mz import MZResult, MZSubsetResult
from covharness.losses import whitened_trace_ratio_panel

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "75240e13873322df8d93f4dfb3bb4a76626a3bb4"
SCREEN_END = "2025-04-12"
SCREEN_START = "2023-11-30"
N_SCREEN = 500
N_ASSETS = 5
N_UNIQUE = 15
SENSITIVITY_LAG = 29
Z_CRITICAL = 1.96
HARQ_FALLBACK_DATES = ("2024-12-10", "2025-03-03")
PANEL_SHA256 = "2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a"
FORECASTS_SHA256 = "118f9eb4f5cfc8f2ffa1b1e73128358725aa179d91ebea8149c1f997a89e6c94"
HARQ_CHECKPOINT_SHA256 = "d2232e6e3e6aa013ff705ff4519312c3dd72cdfcd0e34eed9d50b714eaa0cc14"

COMPLETE_MODELS = (
    {"family": "RW", "config_id": "RW01", "forecast_key": "RW_forecasts"},
    {"family": "EWMA", "config_id": "EWMA01", "forecast_key": "EWMA_forecasts"},
    {"family": "HAR-DRD", "config_id": "HARDRD01", "forecast_key": "HAR-DRD_forecasts"},
    {
        "family": "HARQ-DRD",
        "config_id": "HARQDRD01",
        "forecast_key": "HARQ-DRD_forecasts",
    },
    {
        "family": "LW-linear",
        "config_id": "LWLIN01",
        "forecast_key": "LW-linear_forecasts",
    },
    {"family": "LW-NL", "config_id": "LWNL01", "forecast_key": "LW-NL_forecasts"},
    {
        "family": "Ridge-DRD",
        "config_id": "RIDGE01",
        "forecast_key": "Ridge-DRD_forecasts",
    },
    {
        "family": "XGBoost-DRD",
        "config_id": "XGB08",
        "forecast_key": "XGBoost-DRD_forecasts",
    },
    {
        "family": "LSTM-BEKK",
        "config_id": "LSTM19",
        "forecast_key": "LSTM-BEKK_forecasts",
    },
)

LSTM_SEEDS = tuple(
    {
        "family": "LSTM-BEKK",
        "config_id": f"LSTM19_seed{seed}",
        "forecast_key": f"LSTM-BEKK_LSTM19_seed{seed}_forecasts",
        "seed": seed,
    }
    for seed in range(5)
)

MISSING_MODELS = (
    {"family": "DCC", "config_id": "DCC01", "status": "no forecasts"},
    {"family": "DCC-NL", "config_id": "DCCNL01", "status": "no forecasts"},
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


def json_number(value: float | None) -> float | str | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return str(number)
    return number


def load_screen_panel_rcov(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Copy realized covariance for SCREEN targets only. Stop at SCREEN_END."""
    blob = np.load(path, mmap_mode="r")
    dates = np.asarray(blob["dates"])
    keep: list[int] = []
    # Walk the calendar until the first post-SCREEN date, then stop.
    for index, label in enumerate(dates):
        text = str(label)
        if text > SCREEN_END:
            break
        if SCREEN_START <= text <= SCREEN_END:
            keep.append(index)
    if len(keep) != N_SCREEN:
        raise RuntimeError("SCREEN panel length is not 500")
    index = np.asarray(keep, dtype=int)
    rcov = np.asarray(blob["rcov_5min"][index], dtype=float)
    selected = tuple(str(dates[i]) for i in index)
    if selected[0] != SCREEN_START or selected[-1] != SCREEN_END:
        raise RuntimeError("SCREEN panel date bounds drifted")
    if any(date > SCREEN_END for date in selected):
        raise RuntimeError("a copied panel date is after 2025-04-12")
    return rcov, selected


def subset_payload(result: MZSubsetResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "subset": result.subset,
        "n_entries": int(result.n_entries),
        "alpha": json_number(result.alpha),
        "beta": json_number(result.beta),
        "wald_statistic": json_number(result.wald_statistic),
        "wald_df": result.wald_df,
        "p_value": json_number(result.p_value),
    }


def mz_payload(result: MZResult) -> dict[str, object]:
    return {
        "n_observations": int(result.n_observations),
        "n_assets": int(result.n_assets),
        "n_unique_entries": int(result.n_unique_entries),
        "alpha": json_number(result.alpha),
        "beta": json_number(result.beta),
        "wald_statistic": json_number(result.wald_statistic),
        "wald_df": int(result.wald_df),
        "p_value": json_number(result.p_value),
        "hac_lags": int(result.hac_lags),
        "weighting_rule": result.weighting_rule,
        "inference_case": result.inference_case,
        "covariance_degenerate": bool(result.covariance_degenerate),
        "diagonal": subset_payload(result.diagonal),
        "off_diagonal": subset_payload(result.off_diagonal),
    }


def mean_interval(series: np.ndarray, maxlags: int | None) -> dict[str, object]:
    mean = float(np.mean(series))
    try:
        hac = hac_long_run_variance(series, maxlags=maxlags)
        half_width = Z_CRITICAL * float(hac.standard_error_of_mean)
        return {
            "mean": mean,
            "standard_error": float(hac.standard_error_of_mean),
            "ci_low": mean - half_width,
            "ci_high": mean + half_width,
            "maxlags": int(hac.maxlags),
            "lag_rule": hac.lag_rule,
            "degenerate": False,
        }
    except DegenerateLossDifferentialError:
        lag, rule = (
            (newey_west_1994_lags(int(series.size)), "newey_west_1994")
            if maxlags is None
            else (int(maxlags), "user")
        )
        return {
            "mean": mean,
            "standard_error": None,
            "ci_low": None,
            "ci_high": None,
            "maxlags": lag,
            "lag_rule": rule,
            "degenerate": True,
        }


def ratio_payload(
    ratios: np.ndarray,
    dates: tuple[str, ...],
    fallback_dates: tuple[str, ...] = (),
) -> dict[str, object]:
    auto = mean_interval(ratios, maxlags=None)
    lag29 = mean_interval(ratios, maxlags=SENSITIVITY_LAG)
    lookup = {date: index for index, date in enumerate(dates)}
    flagged = []
    for date in fallback_dates:
        if date not in lookup:
            raise RuntimeError(f"HARQ fallback date {date} is not a SCREEN target")
        flagged.append(
            {
                "date": date,
                "index": int(lookup[date]),
                "ratio": float(ratios[lookup[date]]),
            }
        )
    return {
        "mean": auto["mean"],
        "share_above_one": float(np.mean(ratios > 1.0)),
        "n_above_one": int(np.sum(ratios > 1.0)),
        "automatic": auto,
        "lag_29": lag29,
        "fallback_dates": flagged,
    }


def fit_model(
    family: str,
    config_id: str,
    forecast: np.ndarray,
    proxy: np.ndarray,
    dates: tuple[str, ...],
    fallback_dates: tuple[str, ...] = (),
    seed: int | None = None,
) -> dict[str, object]:
    # Reuse the existing pooled-vech estimator. Do not write a new MZ fit.
    ols_auto = mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_NONE, maxlags=None
    )
    ols_lag29 = mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_NONE, maxlags=SENSITIVITY_LAG
    )
    ps21_auto = mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_APPROXIMATE_PS21, maxlags=None
    )
    ps21_lag29 = mincer_zarnowitz_pooled_vech(
        proxy, forecast, weighting=WEIGHTING_APPROXIMATE_PS21, maxlags=SENSITIVITY_LAG
    )
    ratios = whitened_trace_ratio_panel(proxy, forecast)
    row: dict[str, object] = {
        "family": family,
        "config_id": config_id,
        "status": "complete",
        "ols_automatic": mz_payload(ols_auto),
        "ols_lag_29": mz_payload(ols_lag29),
        "ps21_automatic": mz_payload(ps21_auto),
        "ps21_lag_29": mz_payload(ps21_lag29),
        "whitened_ratio": ratio_payload(ratios, dates, fallback_dates=fallback_dates),
    }
    if seed is not None:
        row["seed"] = int(seed)
    return row


def headline_csv_row(row: dict[str, object]) -> dict[str, object]:
    ols = row["ols_automatic"]
    ps21 = row["ps21_automatic"]
    ratio = row["whitened_ratio"]
    auto = ratio["automatic"]
    return {
        "family": row["family"],
        "config_id": row["config_id"],
        "alpha_ols": ols["alpha"],
        "beta_ols": ols["beta"],
        "wald_p_ols": ols["p_value"],
        "alpha_ps21": ps21["alpha"],
        "beta_ps21": ps21["beta"],
        "wald_p_ps21": ps21["p_value"],
        "mean_ratio": ratio["mean"],
        "mean_ratio_ci_low": auto["ci_low"],
        "mean_ratio_ci_high": auto["ci_high"],
        "share_above_one": ratio["share_above_one"],
    }


def write_csv(path: Path, rows: list[dict[str, object]], header: list[str]) -> None:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join("" if row[key] is None else str(row[key]) for key in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ratio_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["config_id"]) for row in rows]
    means = np.asarray([row["whitened_ratio"]["mean"] for row in rows], dtype=float)
    lows = np.asarray(
        [row["whitened_ratio"]["automatic"]["ci_low"] for row in rows], dtype=float
    )
    highs = np.asarray(
        [row["whitened_ratio"]["automatic"]["ci_high"] for row in rows], dtype=float
    )
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.axvline(1.0, color="black", linewidth=0.9)
    axis.errorbar(
        means,
        positions,
        xerr=[means - lows, highs - means],
        fmt="o",
        color="C0",
        capsize=3.0,
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel("mean whitened ratio")
    axis.set_title("SCREEN mean tr(H^{-1} S)/N, HAC 95 percent interval")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 SCREEN pooled Mincer-Zarnowitz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "post_screen" / "A2",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_path = REPO_ROOT / "data" / "processed" / "binance_five_asset_panel.npz"
    forecasts_path = REPO_ROOT / "results" / "binance_screen_forecasts.npz"
    checkpoint_path = (
        REPO_ROOT / "results" / "binance_screen_checkpoints" / "HARQ-DRD_HARQDRD01.npz"
    )
    # Refuse to proceed if the frozen panel or SCREEN forecast cube has changed.
    require_hash(panel_path, PANEL_SHA256)
    require_hash(forecasts_path, FORECASTS_SHA256)
    require_hash(checkpoint_path, HARQ_CHECKPOINT_SHA256)

    proxy, dates = load_screen_panel_rcov(panel_path)
    forecasts = np.load(forecasts_path)
    target_dates = tuple(str(value) for value in forecasts["target_dates"].tolist())
    if target_dates != dates:
        raise RuntimeError("SCREEN panel dates do not match forecast target dates")
    if any(date > SCREEN_END for date in target_dates):
        raise RuntimeError("a SCREEN forecast date is after 2025-04-12")
    np.testing.assert_allclose(
        proxy, np.asarray(forecasts["proxy_rcov"], dtype=float), rtol=0.0, atol=0.0
    )

    checkpoint = np.load(checkpoint_path)
    repaired = np.asarray(checkpoint["repaired"], dtype=bool)
    fallback_indices = [index for index, date in enumerate(dates) if date in HARQ_FALLBACK_DATES]
    if int(repaired.sum()) != 2:
        raise RuntimeError("HARQ SCREEN repair_count is not 2")
    if not np.array_equal(np.flatnonzero(repaired), np.asarray(fallback_indices)):
        raise RuntimeError("HARQ fallback dates do not match checkpoint repaired flags")

    if "DCC_forecasts" in forecasts.files or "DCC-NL_forecasts" in forecasts.files:
        raise RuntimeError("DCC forecasts appeared in the SCREEN NPZ")

    complete_rows = []
    for model in COMPLETE_MODELS:
        fallback = HARQ_FALLBACK_DATES if model["config_id"] == "HARQDRD01" else ()
        complete_rows.append(
            fit_model(
                model["family"],
                model["config_id"],
                np.asarray(forecasts[model["forecast_key"]], dtype=float),
                proxy,
                dates,
                fallback_dates=fallback,
            )
        )

    seed_rows = [
        fit_model(
            item["family"],
            item["config_id"],
            np.asarray(forecasts[item["forecast_key"]], dtype=float),
            proxy,
            dates,
            seed=item["seed"],
        )
        for item in LSTM_SEEDS
    ]

    summary = {
        "task": "A2",
        "label": "descriptive SCREEN pooled Mincer-Zarnowitz calibration",
        "spec_commit": SPEC_COMMIT,
        "screen_start": SCREEN_START,
        "screen_end": SCREEN_END,
        "n_observations": N_SCREEN,
        "n_assets": N_ASSETS,
        "n_unique_entries": N_UNIQUE,
        "null": {"alpha": 0.0, "beta": 1.0},
        "weighting": [WEIGHTING_NONE, WEIGHTING_APPROXIMATE_PS21],
        "hac_lags": {
            "automatic": int(newey_west_1994_lags(N_SCREEN)),
            "sensitivity": SENSITIVITY_LAG,
        },
        "recalibration": (
            "SCREEN ratios from this task do not set any recalibration constant. "
            "Task C3, if later run, fits such a constant on past data only."
        ),
        "log_or_sqrt_transforms": False,
        "panel_sha256": PANEL_SHA256,
        "forecasts_sha256": FORECASTS_SHA256,
        "harq_checkpoint_sha256": HARQ_CHECKPOINT_SHA256,
        "harq_fallback_dates": list(HARQ_FALLBACK_DATES),
        "missing_models": list(MISSING_MODELS),
        "complete_models": complete_rows,
        "lstm_seeds": seed_rows,
    }
    (output_dir / "a2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    headline_header = [
        "family",
        "config_id",
        "alpha_ols",
        "beta_ols",
        "wald_p_ols",
        "alpha_ps21",
        "beta_ps21",
        "wald_p_ps21",
        "mean_ratio",
        "mean_ratio_ci_low",
        "mean_ratio_ci_high",
        "share_above_one",
    ]
    write_csv(
        output_dir / "a2_summary.csv",
        [headline_csv_row(row) for row in complete_rows],
        headline_header,
    )
    write_csv(
        output_dir / "a2_appendix_lstm_seeds.csv",
        [headline_csv_row(row) for row in seed_rows],
        headline_header,
    )

    sensitivity_header = [
        "family",
        "config_id",
        "weighting",
        "lag",
        "alpha",
        "beta",
        "wald_p",
        "diag_alpha",
        "diag_beta",
        "diag_p",
        "off_alpha",
        "off_beta",
        "off_p",
    ]
    sensitivity_rows: list[dict[str, object]] = []
    for row in complete_rows:
        for weighting, key in (
            (WEIGHTING_NONE, "ols_automatic"),
            (WEIGHTING_NONE, "ols_lag_29"),
            (WEIGHTING_APPROXIMATE_PS21, "ps21_automatic"),
            (WEIGHTING_APPROXIMATE_PS21, "ps21_lag_29"),
        ):
            fit = row[key]
            diagonal = fit["diagonal"]
            off_diagonal = fit["off_diagonal"]
            sensitivity_rows.append(
                {
                    "family": row["family"],
                    "config_id": row["config_id"],
                    "weighting": weighting,
                    "lag": fit["hac_lags"],
                    "alpha": fit["alpha"],
                    "beta": fit["beta"],
                    "wald_p": fit["p_value"],
                    "diag_alpha": None if diagonal is None else diagonal["alpha"],
                    "diag_beta": None if diagonal is None else diagonal["beta"],
                    "diag_p": None if diagonal is None else diagonal["p_value"],
                    "off_alpha": None if off_diagonal is None else off_diagonal["alpha"],
                    "off_beta": None if off_diagonal is None else off_diagonal["beta"],
                    "off_p": None if off_diagonal is None else off_diagonal["p_value"],
                }
            )
    write_csv(output_dir / "a2_sensitivity.csv", sensitivity_rows, sensitivity_header)
    write_ratio_plot(output_dir / "mean_whitened_ratio.png", complete_rows)

    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
