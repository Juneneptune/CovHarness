"""Crash-day influence on the saved CONFIRM reduced-QLIKE ranking.

Exploratory, computed after CONFIRM. No model is refit. No day is removed
from one model only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from covharness.inference.hac import ewc_long_run_variance
from covharness.protocol.binance_validation import ensemble_covariances, score_forecast_cube

LABEL = "exploratory, computed after CONFIRM"
REPO = Path(__file__).resolve().parents[2]
JOBS = REPO / "results/confirm/jobs"
MEASURED = REPO / "results/confirm/inputs/confirm_measurements.npz"
PANEL = REPO / "data/processed/binance_five_asset_panel.npz"
OUT = REPO / "results/exploratory/X3"
CRASH = "2025-10-10"
NU = 25
ORDINARY = 2.0595385527532972
C29 = 2.4226501420943998
C7 = 2.4728093597276533
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
PAIRS = (
    ("HARQDRD01", "V01"),
    ("V01", "LSTMBEKK-noC"),
    ("LSTM22", "LSTMBEKK-noC"),
    ("V01", "LSTM22"),
    ("BEKKS01", "V01"),
)


def p5_statistic(differential: np.ndarray, nu: int = NU) -> float:
    """Equal-weighted cosine statistic of a one-dimensional loss gap.

    ``differential`` has shape ``(T,)``. Nu is fixed. The statistic is the
    mean divided by the square root of the long-run variance over ``T``.
    """
    series = np.asarray(differential, dtype=float)
    if series.ndim != 1 or series.shape[0] < 2 or not np.isfinite(series).all():
        raise ValueError("P5 requires a finite one-dimensional series of length at least 2")
    omega, _terms = ewc_long_run_variance(series, nu=nu)
    return float(series.mean() / np.sqrt(omega / series.shape[0]))


def leave_one_out_statistics(differential: np.ndarray, nu: int = NU) -> np.ndarray:
    """P5 statistic after dropping each observation once.

    The result has shape ``(T,)``. Index ``t`` drops observation ``t`` from
    both series that formed the gap. The remaining order is unchanged.
    """
    series = np.asarray(differential, dtype=float)
    if series.ndim != 1 or series.shape[0] < 3:
        raise ValueError("leave-one-out requires at least three observations")
    statistics = np.empty(series.shape[0], dtype=float)
    # Drop one date for the whole gap. Do not reorder the remaining dates.
    for index in range(series.shape[0]):
        kept = np.concatenate([series[:index], series[index + 1 :]])
        statistics[index] = p5_statistic(kept, nu=nu)
    return statistics


def _load_forecasts(model_id: str) -> np.ndarray:
    """Equal-weight seed ensemble, or the single saved cube."""
    if model_id in SEEDED:
        cubes = [
            np.array(np.load(JOBS / model_id / f"seed_{seed}" / "forecast.npz")["forecasts"], dtype=float)
            for seed in range(5)
        ]
        return ensemble_covariances(cubes)
    return np.array(np.load(JOBS / model_id / "forecast.npz")["forecasts"], dtype=float)


def _average_rank(values: np.ndarray) -> np.ndarray:
    """Rank 1 is the lowest loss. Ties receive the average rank."""
    return rankdata(np.asarray(values, dtype=float), method="average")


def _ranking(losses: dict[str, np.ndarray], keep: np.ndarray) -> pd.DataFrame:
    """Three views of one proxy. ``keep`` drops the crash day inside this view."""
    rows = []
    means = []
    without = []
    medians = []
    for model_id in ROSTER:
        series = losses[model_id]
        means.append(float(series.mean()))
        without.append(float(series[keep].mean()))
        medians.append(float(np.median(series)))
        rows.append({"model": model_id, "n": int(series.shape[0])})
    frame = pd.DataFrame(rows)
    frame["mean_all"] = means
    frame["mean_without_2025_10_10"] = without
    frame["median"] = medians
    frame["rank_mean_all"] = _average_rank(frame["mean_all"].to_numpy())
    frame["rank_mean_without_2025_10_10"] = _average_rank(frame["mean_without_2025_10_10"].to_numpy())
    frame["rank_median"] = _average_rank(frame["median"].to_numpy())
    frame["label"] = LABEL
    return frame


def _known_truth(
    dates: np.ndarray,
    qlike5: dict[str, np.ndarray],
    qlike15: dict[str, np.ndarray],
    stored_harq: np.ndarray,
) -> dict[str, float]:
    """Stop unless the saved CONFIRM series are reproduced."""
    harq = qlike5["HARQDRD01"]
    if harq.shape != stored_harq.shape or float(np.max(np.abs(harq - stored_harq))) > 1e-10:
        raise SystemExit("HARQDRD01 five-minute QLIKE does not match the stored CONFIRM loss")
    gap15 = qlike15["HARQDRD01"] - qlike15["V01"]
    stat15 = p5_statistic(gap15)
    if abs(float(gap15.mean()) - 0.70451) > 5e-6 or abs(stat15 - 2.388403) > 5e-7:
        raise SystemExit(
            f"15-minute headline check failed. gap {float(gap15.mean())} statistic {stat15}"
        )
    crash = dates == CRASH
    if int(crash.sum()) != 1:
        raise SystemExit("2025-10-10 is not a single CONFIRM target")
    gap5 = qlike5["HARQDRD01"] - qlike5["V01"]
    crash_gap = float(gap5[crash][0])
    if abs(crash_gap - 59.83) > 0.005:
        raise SystemExit(f"crash-day gap {crash_gap} is not 59.83")
    without = ~crash
    checks = {
        "EWMA21": qlike5["EWMA21"][without].mean(),
        "V01": qlike5["V01"][without].mean(),
        "HARQDRD01": qlike5["HARQDRD01"][without].mean(),
    }
    expected = {"EWMA21": -34.895, "V01": -34.437, "HARQDRD01": -33.738}
    for name, target in expected.items():
        if abs(float(checks[name]) - target) >= 0.005:
            raise SystemExit(f"{name} mean without the crash day is {float(checks[name])}")
    full = p5_statistic(gap5)
    held_out = p5_statistic(gap5[without])
    if abs(full - 2.6519476756730893) > 1e-9:
        raise SystemExit(f"full-sample P5 {full} does not match the stored headline")
    if abs(held_out - 2.400) >= 0.01:
        raise SystemExit(f"without-crash P5 {held_out} is not 2.400 at the stated tolerance")
    return {
        "n_five_minute": float(gap5.shape[0]),
        "p5_all": full,
        "p5_without": held_out,
        "crash_gap": crash_gap,
        "p5_15": stat15,
        "gap_15": float(gap15.mean()),
    }


def main() -> None:
    measured = np.load(MEASURED)
    panel = np.load(PANEL)
    dates = np.array([str(value) for value in measured["target_dates"].tolist()])
    stored_targets = np.load(JOBS / "HARQDRD01" / "forecast.npz")["targets"].astype(str)
    if not np.array_equal(dates, stored_targets):
        raise SystemExit("measurement dates and HARQDRD01 targets differ")
    panel_dates = [str(value) for value in panel["dates"].tolist()]
    lookup = {day: index for index, day in enumerate(panel_dates)}
    try:
        rows = [lookup[day] for day in dates.tolist()]
    except KeyError as exc:
        raise SystemExit(f"CONFIRM target {exc} is not in the production panel") from exc
    proxy5 = np.array(panel["rcov_5min"][rows], dtype=float)
    proxy15 = np.array(measured["rcov_15min"], dtype=float)
    if proxy5.shape[0] != dates.shape[0] or proxy15.shape[0] != dates.shape[0]:
        raise SystemExit("proxy length does not match the CONFIRM targets")
    # Score the saved cubes. A non-finite loss drops that target from the view.
    forecasts = {model_id: _load_forecasts(model_id) for model_id in ROSTER}
    qlike5: dict[str, np.ndarray] = {}
    qlike15: dict[str, np.ndarray] = {}
    finite5 = np.ones(dates.shape[0], dtype=bool)
    finite15 = np.ones(dates.shape[0], dtype=bool)
    for model_id, cube in forecasts.items():
        loss5, _frobenius, _fail5, _fail_f = score_forecast_cube(cube, proxy5)
        loss15, _frobenius15, _fail15, _fail_f15 = score_forecast_cube(cube, proxy15)
        qlike5[model_id] = loss5
        qlike15[model_id] = loss15
        finite5 &= np.isfinite(loss5)
        finite15 &= np.isfinite(loss15)
    if not finite5.all() or not finite15.all():
        qlike5 = {key: value[finite5] for key, value in qlike5.items()}
        qlike15 = {key: value[finite15] for key, value in qlike15.items()}
        dates5 = dates[finite5]
        dates15 = dates[finite15]
    else:
        dates5 = dates
        dates15 = dates
    stored = np.load(JOBS / "HARQDRD01" / "forecast.npz")["qlike"]
    summary = _known_truth(dates5, qlike5, qlike15, np.asarray(stored, dtype=float))
    if int(summary["n_five_minute"]) != 499:
        print(
            "NOTE task text stated 499 five-minute targets and P5 2.647. "
            f"The saved series has {int(summary['n_five_minute'])} finite targets "
            f"and P5 {summary['p5_all']:.6f}.",
            file=sys.stderr,
        )
    crash5 = dates5 == CRASH
    keep5 = ~crash5
    crash15 = dates15 == CRASH
    keep15 = ~crash15
    rank5 = _ranking(qlike5, keep5)
    rank15 = _ranking(qlike15, keep15)
    gap = qlike5["HARQDRD01"] - qlike5["V01"]
    loo = leave_one_out_statistics(gap)
    influence = summary["p5_all"] - loo
    loo_frame = pd.DataFrame(
        {
            "date": dates5,
            "statistic_without": loo,
            "influence": influence,
            "label": LABEL,
        }
    )
    pair_rows = []
    for left, right in PAIRS:
        diff = qlike5[left] - qlike5[right]
        row = {
            "left": left,
            "right": right,
            "mean_all": float(diff.mean()),
            "mean_without_2025_10_10": float(diff[keep5].mean()),
            "median": float(np.median(diff)),
            "label": LABEL,
        }
        if (left, right) == ("HARQDRD01", "V01"):
            crash_gap = float(diff[crash5][0])
            row.update(
                {
                    "crash_gap": crash_gap,
                    "crash_share_of_sum": crash_gap / float(diff.sum()),
                    "p5_all": summary["p5_all"],
                    "p5_without_2025_10_10": summary["p5_without"],
                    "nu": NU,
                    "cutoff_ordinary": ORDINARY,
                    "cutoff_c29": C29,
                    "cutoff_c7": C7,
                    "n": int(diff.shape[0]),
                }
            )
        pair_rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    rank5.to_csv(OUT / "ranking_5min.csv", index=False)
    rank15.to_csv(OUT / "ranking_15min.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "pairs.csv", index=False)
    loo_frame.to_csv(OUT / "loo_influence.csv", index=False)
    order = np.argsort(-np.abs(influence))
    minimum = int(np.argmin(loo))
    print(f"label {LABEL}")
    print(f"n5 {int(summary['n_five_minute'])} p5 {summary['p5_all']} without {summary['p5_without']}")
    print(f"crash_gap {summary['crash_gap']} share {pair_rows[0]['crash_share_of_sum']}")
    print(f"loo_min {dates5[minimum]} {loo[minimum]}")
    print("loo_top", [(dates5[int(index)], float(influence[int(index)])) for index in order[:5]])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
