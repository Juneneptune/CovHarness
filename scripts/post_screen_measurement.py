"""Descriptive pre-CONFIRM signature plot and return-sign asymmetry.

Task A3. The spec commit must precede this run. The script does not
rewrite the production panel, does not change the frozen five-minute
proxy, and does not read CONFIRM dates into an output row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from covharness.data.binance_calendar import HALT_DATE
from covharness.inference.hac import hac_long_run_variance
from covharness.realized.binance_panel import (
    SIGNATURE_INTERVALS,
    build_preconfirm_interval_measurements,
    production_rcov_through_screen,
)
from covharness.realized.sign_asymmetry import bias_corrected_skewness, tail_sign_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "c30b27a5739d803458006853c2d8bbc80c80d32b"
PANEL_SHA256 = "2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a"
SCREEN_END = "2025-04-12"
Z_CRITICAL = 1.96
TAIL_QUANTILES = (0.95, 0.99)
CACHE_ROOT = (
    REPO_ROOT
    / "data"
    / "cache"
    / "open_data"
    / "binance"
    / "spot"
    / "monthly"
    / "klines"
    / "1m"
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


def json_number(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise RuntimeError("a summary number is not finite")
    return number


def mean_pairwise_correlation(rcov: np.ndarray) -> float:
    """Mean over days of the average strict-upper-triangle correlation."""
    variance = np.diagonal(rcov, axis1=1, axis2=2)
    if np.any(variance <= 0.0):
        raise RuntimeError("realized correlation requires a positive variance")
    scale = np.sqrt(variance)
    corr = rcov / scale[:, :, None] / scale[:, None, :]
    row, col = np.triu_indices(rcov.shape[-1], k=1)
    # Average pairs within the day, then average days.
    return float(corr[:, row, col].mean())


def signature_rows(measurement) -> list[dict[str, object]]:
    n_assets = measurement.rcov.shape[-1]
    m_over_n = measurement.n_intervals / n_assets
    pair = mean_pairwise_correlation(measurement.rcov)
    mean_variance = measurement.rcov.diagonal(axis1=1, axis2=2).mean(axis=0)
    rows = []
    for asset_index, asset in enumerate(measurement.assets):
        rows.append(
            {
                "interval_minutes": measurement.interval_minutes,
                "M": measurement.n_intervals,
                "N": n_assets,
                "M_over_N": m_over_n,
                "asset": asset,
                "mean_realized_variance": float(mean_variance[asset_index]),
                "mean_pairwise_correlation": pair,
            }
        )
    return rows


def asymmetry_rows(name: str, panel: np.ndarray, assets: tuple[str, ...]) -> list[dict[str, object]]:
    """Per-asset rows, then one pooled row, for a ``(T, N)`` or ``(T, M, N)`` array."""
    rows: list[dict[str, object]] = []
    if panel.ndim == 2:
        per_asset = [panel[:, index] for index in range(panel.shape[1])]
        pooled = panel.reshape(-1)
    elif panel.ndim == 3:
        per_asset = [panel[:, :, index].reshape(-1) for index in range(panel.shape[2])]
        pooled = panel.reshape(-1)
    else:
        raise RuntimeError(f"unexpected return panel ndim {panel.ndim}")
    series = [(asset, values, "asset") for asset, values in zip(assets, per_asset)]
    series.append(("pooled", pooled, "pooled"))
    for asset, values, scope in series:
        skewness = bias_corrected_skewness(values)
        for quantile in TAIL_QUANTILES:
            summary = tail_sign_summary(values, quantile)
            rows.append(
                {
                    "series": name,
                    "scope": scope,
                    "asset": asset,
                    "skewness": skewness,
                    **summary,
                }
            )
    return rows


def semivariance_rows(returns, assets: tuple[str, ...]) -> list[dict[str, object]]:
    """Means and HAC intervals for RS+ - RS- on a ``(T, M, N)`` return cube."""
    squares = returns * returns
    # Zeros stay zero, so the sign split reconstructs every square.
    partition = np.where(returns > 0.0, squares, 0.0) + np.where(returns < 0.0, squares, 0.0)
    if not np.array_equal(partition, squares):
        raise RuntimeError("semivariances do not partition realized variance")
    positive = np.sum(np.where(returns > 0.0, squares, 0.0), axis=1)
    negative = np.sum(np.where(returns < 0.0, squares, 0.0), axis=1)
    rows = []
    for asset_index, asset in enumerate(assets):
        gap = positive[:, asset_index] - negative[:, asset_index]
        hac = hac_long_run_variance(gap)
        half_width = Z_CRITICAL * hac.standard_error_of_mean
        rows.append(
            {
                "asset": asset,
                "mean_rs_pos": float(positive[:, asset_index].mean()),
                "mean_rs_neg": float(negative[:, asset_index].mean()),
                "mean_gap": float(hac.mean),
                "hac_lag": int(hac.maxlags),
                "hac_lag_rule": hac.lag_rule,
                "ci_low": float(hac.mean - half_width),
                "ci_high": float(hac.mean + half_width),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intervals = sorted({int(row["interval_minutes"]) for row in rows})
    assets = []
    for row in rows:
        asset = str(row["asset"])
        if asset not in assets:
            assets.append(asset)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    for asset in assets:
        subset = [row for row in rows if row["asset"] == asset]
        subset = sorted(subset, key=lambda row: int(row["interval_minutes"]))
        axes[0].plot(
            [int(row["interval_minutes"]) for row in subset],
            [float(row["mean_realized_variance"]) for row in subset],
            marker="o",
            label=asset,
        )
    axes[0].set_ylabel("mean realized variance")
    axes[0].legend(frameon=False, fontsize=8)
    corr_x = []
    corr_y = []
    for interval in intervals:
        match = next(row for row in rows if int(row["interval_minutes"]) == interval)
        corr_x.append(interval)
        corr_y.append(float(match["mean_pairwise_correlation"]))
    axes[1].plot(corr_x, corr_y, marker="o", color="black")
    axes[1].set_ylabel("mean pairwise correlation")
    axes[1].set_xlabel("sampling interval, minutes")
    axes[1].set_xticks(intervals)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 pre-CONFIRM signature and sign asymmetry")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "post_screen" / "A3",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = REPO_ROOT / "data" / "processed" / "binance_five_asset_panel.npz"
    require_hash(panel_path, PANEL_SHA256)

    measurements = build_preconfirm_interval_measurements(CACHE_ROOT, SIGNATURE_INTERVALS)
    by_interval = {item.interval_minutes: item for item in measurements}
    five = by_interval[5]
    if five.dates[-1].isoformat() != SCREEN_END:
        raise RuntimeError("measurement does not end on 2025-04-12")
    if HALT_DATE in five.dates:
        raise RuntimeError("halt date is a measured row")
    if any(day.isoformat() > SCREEN_END for day in five.dates):
        raise RuntimeError("a measured date is after 2025-04-12")
    stored = production_rcov_through_screen(panel_path, five.dates)
    if not np.array_equal(five.rcov, stored):
        raise RuntimeError("five-minute RCov is not bitwise equal to the production panel")

    signature = []
    for interval in SIGNATURE_INTERVALS:
        signature.extend(signature_rows(by_interval[interval]))
    asymmetry = []
    asymmetry.extend(asymmetry_rows("daily_utc", five.daily_returns, five.assets))
    asymmetry.extend(asymmetry_rows("five_minute", five.returns, five.assets))
    semivariance = semivariance_rows(five.returns, five.assets)

    payload = {
        "task": "A3",
        "label": "descriptive pre-CONFIRM signature plot and return-sign asymmetry",
        "spec_commit": SPEC_COMMIT,
        "date_start": five.dates[0].isoformat(),
        "date_end": five.dates[-1].isoformat(),
        "n_dates": len(five.dates),
        "halt_excluded": True,
        "confirm_rows": 0,
        "intervals_minutes": list(SIGNATURE_INTERVALS),
        "anchor": "prior calendar day 23:59 one-minute close",
        "forward_fill": False,
        "five_minute_bitwise_parity": True,
        "panel_sha256": PANEL_SHA256,
        "proxy_changed": False,
        "production_panel_rewritten": False,
        "tail_quantiles": list(TAIL_QUANTILES),
        "quantile_method": "linear",
        "binomial_null": 0.5,
        "multiplicity_adjustment": False,
        "semivariance_hac": "newey_west_1994",
        "signature": signature,
        "asymmetry": [
            {key: (json_number(value) if isinstance(value, float) else value) for key, value in row.items()}
            for row in asymmetry
        ],
        "semivariance": semivariance,
    }
    summary_path = output_dir / "a3_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "a3_signature.csv", signature)
    write_csv(
        output_dir / "a3_asymmetry.csv",
        payload["asymmetry"],
    )
    write_csv(output_dir / "a3_semivariance.csv", semivariance)
    write_plot(output_dir / "signature_plot.png", signature)


if __name__ == "__main__":
    main()
