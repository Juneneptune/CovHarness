"""Descriptive SCREEN leverage regression for realized variance and forecasts.

Task A4. The spec commit must precede this run. The script does not refit
LSTM-BEKK or HARQ-DRD, does not change a model, and does not read CONFIRM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from covharness.inference.leverage import (
    COEF_NAMES,
    LeverageFit,
    fit_leverage_ols,
    fit_pooled_leverage_ols,
    har_leverage_design,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "d0d3f70547038bee845b2984c1ffdb2349f61650"
PANEL_SHA256 = "2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a"
FORECASTS_SHA256 = "118f9eb4f5cfc8f2ffa1b1e73128358725aa179d91ebea8149c1f997a89e6c94"
HISTORY_B_START = "2023-03-25"
SCREEN_START = "2023-11-30"
SCREEN_END = "2025-04-12"
HARQ_FALLBACK_DATES = ("2024-12-10", "2025-03-03")
ASSETS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "ADAUSDT")


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


def load_history_b_through_screen(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Copy variance diagonals and daily returns from HISTORY_B through SCREEN."""
    blob = np.load(path, mmap_mode="r")
    labels = np.asarray(blob["dates"])
    keep: list[int] = []
    # Stop at the first post-SCREEN date. Do not copy CONFIRM rows.
    for index, label in enumerate(labels):
        text = str(label)
        if text > SCREEN_END:
            break
        if HISTORY_B_START <= text <= SCREEN_END:
            keep.append(index)
    if not keep:
        raise RuntimeError("no HISTORY_B through SCREEN rows were found")
    selected = tuple(str(labels[index]) for index in keep)
    if selected[0] != HISTORY_B_START or selected[-1] != SCREEN_END:
        raise RuntimeError("lag sample does not run from HISTORY_B through SCREEN")
    if any(date > SCREEN_END for date in selected):
        raise RuntimeError("a copied panel date is after 2025-04-12")
    if "2023-03-24" in selected:
        raise RuntimeError("the halt date entered the lag sample")
    index = np.asarray(keep, dtype=int)
    rcov = np.asarray(blob["rcov_5min"][index], dtype=float)
    daily = np.asarray(blob["daily_returns"][index], dtype=float)
    variance = np.diagonal(rcov, axis1=1, axis2=2).copy()
    return variance, daily, selected


def fit_payload(name: str, asset: str, fit: LeverageFit, scope: str) -> dict[str, object]:
    coefficients = {
        key: float(value) for key, value in zip(COEF_NAMES, fit.coefficients)
    }
    return {
        "dependent": name,
        "scope": scope,
        "asset": asset,
        "coefficients": coefficients,
        "g_pos": coefficients["g_pos"],
        "g_neg": coefficients["g_neg"],
        "gap": fit.gap,
        "gap_se": fit.gap_se,
        "gap_ci_low": fit.gap_ci_low,
        "gap_ci_high": fit.gap_ci_high,
        "hac_lag": fit.hac_lag,
        "n_observations": fit.n_observations,
        "g_neg_negative": fit.g_neg_negative,
        "abs_g_neg_exceeds_g_pos": fit.abs_g_neg_exceeds_g_pos,
        "leverage_label": fit.leverage_label,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    flat = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "coefficients"}
        coefficients = row["coefficients"]
        assert isinstance(coefficients, dict)
        item.update(coefficients)
        flat.append(item)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="A4 SCREEN leverage regression")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "post_screen" / "A4",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = REPO_ROOT / "data" / "processed" / "binance_five_asset_panel.npz"
    forecasts_path = REPO_ROOT / "results" / "binance_screen_forecasts.npz"
    require_hash(panel_path, PANEL_SHA256)
    require_hash(forecasts_path, FORECASTS_SHA256)

    variance, daily, dates = load_history_b_through_screen(panel_path)
    screen_index = np.asarray(
        [index for index, date in enumerate(dates) if date >= SCREEN_START],
        dtype=np.intp,
    )
    if dates[int(screen_index[0])] != SCREEN_START or len(screen_index) != 500:
        raise RuntimeError("SCREEN response block is not the 500 stored targets")
    if int(screen_index[0]) < 22:
        raise RuntimeError("SCREEN responses do not have a full HAR lag inside the copied sample")

    forecasts = np.load(forecasts_path)
    target_dates = tuple(str(value) for value in forecasts["target_dates"].tolist())
    if target_dates != tuple(dates[index] for index in screen_index):
        raise RuntimeError("forecast dates do not match SCREEN targets")
    if any(date > SCREEN_END for date in target_dates):
        raise RuntimeError("a forecast date is after 2025-04-12")
    lstm = np.asarray(forecasts["LSTM-BEKK_forecasts"], dtype=float)
    harq = np.asarray(forecasts["HARQ-DRD_forecasts"], dtype=float)
    if lstm.shape != (500, 5, 5) or harq.shape != lstm.shape:
        raise RuntimeError("forecast cubes are not (500, 5, 5)")
    if np.any(np.diagonal(lstm, axis1=1, axis2=2) <= 0.0):
        raise RuntimeError("LSTM19 has a nonpositive variance forecast")
    if np.any(np.diagonal(harq, axis1=1, axis2=2) <= 0.0):
        raise RuntimeError("HARQ-DRD has a nonpositive variance forecast")

    rows: list[dict[str, object]] = []
    pooled_designs: dict[str, list[np.ndarray]] = {"data": [], "LSTM19": [], "HARQDRD01": []}
    pooled_responses: dict[str, list[np.ndarray]] = {"data": [], "LSTM19": [], "HARQDRD01": []}
    for asset_index, asset in enumerate(ASSETS):
        design, data_response = har_leverage_design(
            variance[:, asset_index], daily[:, asset_index], screen_index
        )
        # Forecast diagonals replace the response. The design stays on realized lags.
        lstm_response = np.log(np.diagonal(lstm, axis1=1, axis2=2)[:, asset_index])
        harq_response = np.log(np.diagonal(harq, axis1=1, axis2=2)[:, asset_index])
        if design.shape[0] != lstm_response.shape[0]:
            raise RuntimeError("forecast length does not match the SCREEN design")
        fits = {
            "data": fit_leverage_ols(design, data_response),
            "LSTM19": fit_leverage_ols(design, lstm_response),
            "HARQDRD01": fit_leverage_ols(design, harq_response),
        }
        for name, fit in fits.items():
            rows.append(fit_payload(name, asset, fit, "asset"))
            pooled_designs[name].append(design)
            pooled_responses[name].append(
                {"data": data_response, "LSTM19": lstm_response, "HARQDRD01": harq_response}[name]
            )
    for name in ("data", "LSTM19", "HARQDRD01"):
        pooled = fit_pooled_leverage_ols(
            np.stack(pooled_designs[name], axis=0),
            np.stack(pooled_responses[name], axis=0),
        )
        rows.append(fit_payload(name, "pooled", pooled, "pooled"))

    payload = {
        "task": "A4",
        "label": "descriptive SCREEN leverage regression",
        "spec_commit": SPEC_COMMIT,
        "screen_start": SCREEN_START,
        "screen_end": SCREEN_END,
        "lag_sample_start": dates[0],
        "n_screen": 500,
        "halt_in_sample": False,
        "confirm_rows": 0,
        "harq_fallback_dates": list(HARQ_FALLBACK_DATES),
        "fallback_dates_dropped": False,
        "hac": "newey_west_1994",
        "pooled_is_secondary": True,
        "models_changed": False,
        "panel_sha256": PANEL_SHA256,
        "forecasts_sha256": FORECASTS_SHA256,
        "fits": rows,
    }
    output_dir.joinpath("a4_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "a4_summary.csv", rows)


if __name__ == "__main__":
    main()
