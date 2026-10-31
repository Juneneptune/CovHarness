"""SCREEN global-minimum-variance evaluation from saved forecasts.

Task E1. No model is refit. CONFIRM dates are not scored.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from covharness.inference.dm import diebold_mariano
from covharness.portfolio.gmv import (
    LongOnlySolverError,
    drift_weights,
    long_only_gmv_weights,
    subtract_turnover_cost,
    turnover,
    unconstrained_gmv_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_COMMIT = "PENDING"
PANEL = REPO_ROOT / "data/processed/binance_five_asset_panel.npz"
FORECASTS = REPO_ROOT / "results/post_screen/C3/recalibrated_forecasts.npz"
C3_SUMMARY = REPO_ROOT / "results/post_screen/C3/c3_summary.json"
B1B = REPO_ROOT / "results/post_screen/B1b/b1b_cutoffs.json"
OUTPUT = REPO_ROOT / "results/post_screen/E1"
CONFIRM_START = "2025-04-13"
ANNUALIZATION = 365.0
PAIRS = ("LSTM22", "V11", "EWMA21", "HARDRD01")
REFERENCE = "HARQDRD01"
FROZEN_ORDINARY = 2.0595385527532972
FROZEN_C7 = 2.4728093597276533
FROZEN_C29 = 2.4226501420943998


def _dates(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def _book(
    forecasts: np.ndarray,
    simple_returns: np.ndarray,
    proxies: np.ndarray,
    kind: str,
) -> dict[str, object]:
    n_rows, n_assets = simple_returns.shape
    weights = np.full((n_rows, n_assets), np.nan)
    gross = np.full(n_rows, np.nan)
    traded = np.full(n_rows, np.nan)
    proxy_variance = np.full(n_rows, np.nan)
    failures: list[str] = []
    for index in range(n_rows):
        try:
            if kind == "unconstrained":
                current = unconstrained_gmv_weights(forecasts[index])
            elif kind == "long_only":
                current = long_only_gmv_weights(forecasts[index])
            else:
                current = np.full(n_assets, 1.0 / n_assets)
        except (np.linalg.LinAlgError, LongOnlySolverError, ValueError):
            failures.append(str(index))
            continue
        weights[index] = current
        gross[index] = float(current @ simple_returns[index])
        proxy_variance[index] = float(current @ proxies[index] @ current)
        if index == 0:
            traded[index] = 0.0
        else:
            previous = weights[index - 1]
            if not np.isfinite(previous).all():
                traded[index] = np.nan
                continue
            drifted = drift_weights(previous, simple_returns[index - 1])
            traded[index] = turnover(current, drifted)
    finite = np.isfinite(gross) & np.isfinite(traded)
    return {
        "gross": gross,
        "turnover": traded,
        "proxy_variance": proxy_variance,
        "failures": failures,
        "n_used": int(finite.sum()),
    }


def _moments(gross: np.ndarray, traded: np.ndarray, proxy_variance: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(gross) & np.isfinite(traded)
    if int(finite.sum()) < 2:
        return {
            "annualized_sd_gross": None,
            "annualized_sd_net_5bps": None,
            "annualized_sd_net_10bps": None,
            "mean_squared_return": None,
            "mean_turnover": None,
            "sharpe_net_5bps": None,
            "sharpe_net_10bps": None,
            "mean_proxy_variance": None,
        }
    used = gross[finite]
    costs = traded[finite]
    net5 = subtract_turnover_cost(used, costs, 5.0)
    net10 = subtract_turnover_cost(used, costs, 10.0)

    def sd(values: np.ndarray) -> float:
        return float(np.std(values, ddof=1) * np.sqrt(ANNUALIZATION))

    def sharpe(values: np.ndarray) -> float | None:
        scale = float(np.std(values, ddof=1))
        if scale == 0.0:
            return None
        return float(np.mean(values) / scale * np.sqrt(ANNUALIZATION))

    proxy = proxy_variance[np.isfinite(proxy_variance)]
    return {
        "annualized_sd_gross": sd(used),
        "annualized_sd_net_5bps": sd(net5),
        "annualized_sd_net_10bps": sd(net10),
        "mean_squared_return": float(np.mean(used**2)),
        "mean_turnover": float(np.mean(costs)),
        "sharpe_net_5bps": sharpe(net5),
        "sharpe_net_10bps": sharpe(net10),
        "mean_proxy_variance": None if proxy.size == 0 else float(np.mean(proxy)),
    }


def _cutoffs() -> dict[str, float]:
    payload = json.loads(B1B.read_text())
    by_length = {
        int(block["block_length"]): float(block["size_adjusted_critical_value"])
        for block in payload["blocks"]
    }
    ordinary = float(payload["ordinary_critical_value"])
    if (
        ordinary != FROZEN_ORDINARY
        or by_length[7] != FROZEN_C7
        or by_length[29] != FROZEN_C29
    ):
        raise SystemExit("B1b cutoffs do not match the frozen register values")
    return {"ordinary": ordinary, "c_7": by_length[7], "c_29": by_length[29]}


def _p5(reference: np.ndarray, other: np.ndarray, cutoffs: dict[str, float]) -> dict[str, object]:
    paired = np.isfinite(reference) & np.isfinite(other)
    differential = reference[paired] ** 2 - other[paired] ** 2
    result = diebold_mariano(differential, reference="ewc")
    statistic = float(result.statistic)
    return {
        "n_observations": int(paired.sum()),
        "mean_differential": float(result.mean_differential),
        "statistic": statistic,
        "ewc_critical_value": float(result.critical_value),
        "win_rate": float(np.mean(differential < 0.0)),
        "reject_ordinary": abs(statistic) > cutoffs["ordinary"],
        "reject_c7": abs(statistic) > cutoffs["c_7"],
        "reject_c29": abs(statistic) > cutoffs["c_29"],
        "cumulative": np.cumsum(differential).tolist(),
    }


def main() -> None:
    if SPEC_COMMIT == "PENDING":
        raise SystemExit("record the spec commit before the empirical run")
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = np.load(PANEL)
    forecasts = np.load(FORECASTS)
    summary = json.loads(C3_SUMMARY.read_text())
    targets = _dates(forecasts["target_dates"])
    if any(day >= CONFIRM_START for day in targets):
        raise SystemExit("a CONFIRM target entered the GMV sample")
    date_index = {day: index for index, day in enumerate(_dates(panel["dates"]))}
    rows = np.array([date_index[day] for day in targets])
    if set(_dates(panel["segments"][rows])) != {"SCREEN"}:
        raise SystemExit("a GMV return date is outside SCREEN")
    proxy = np.array(forecasts["proxy_rcov"], dtype=float)
    panel_gap = float(np.max(np.abs(panel["rcov_5min"][rows] - proxy)))
    simple = np.expm1(np.array(panel["daily_returns"][rows], dtype=float))
    roles = {row["model_id"]: row for row in summary["models"]}
    model_ids = _dates(forecasts["model_ids"])
    cutoffs = _cutoffs()
    books: dict[tuple[str, str, str], dict[str, object]] = {}
    rows_out = []
    for position, model_id in enumerate(model_ids):
        role = roles[model_id]
        for version, key in (("raw", "raw"), ("r1", "r1"), ("r2", "r2")):
            cube = np.array(forecasts[key][position], dtype=float)
            for kind in ("unconstrained", "long_only"):
                book = _book(cube, simple, proxy, kind)
                books[(model_id, version, kind)] = book
                moments = _moments(
                    book["gross"], book["turnover"], book["proxy_variance"]
                )
                rows_out.append(
                    {
                        "model_id": model_id,
                        "role": role["role"],
                        "appendix": role["appendix"],
                        "version": version,
                        "book": kind,
                        "n_used": book["n_used"],
                        "solver_failures": len(book["failures"]),
                        **moments,
                    }
                )
        print(model_id, flush=True)
    equal = _book(np.zeros((simple.shape[0], simple.shape[1], simple.shape[1])), simple, proxy, "equal")
    equal_moments = _moments(equal["gross"], equal["turnover"], equal["proxy_variance"])
    rows_out.append(
        {
            "model_id": "EQUAL",
            "role": "anchor",
            "appendix": False,
            "version": "none",
            "book": "equal_weight",
            "n_used": equal["n_used"],
            "solver_failures": 0,
            **equal_moments,
        }
    )
    inference = {}
    for version, kind in (
        ("raw", "unconstrained"),
        ("raw", "long_only"),
        ("r1", "unconstrained"),
        ("r2", "unconstrained"),
    ):
        reference_book = books[(REFERENCE, version, kind)]["gross"]
        for other in PAIRS:
            label = f"{REFERENCE}-{other}:{version}:{kind}"
            inference[label] = _p5(
                reference_book, books[(other, version, kind)]["gross"], cutoffs
            )
            inference[label]["cumulative"] = inference[label]["cumulative"]
    primary_keys = [f"{REFERENCE}-{other}:raw:unconstrained" for other in PAIRS]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for other, key in zip(PAIRS, primary_keys):
        axis.plot(inference[key]["cumulative"], label=f"{REFERENCE} minus {other}")
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.set_xlabel("SCREEN target")
    axis.set_ylabel("Cumulative squared-return gap")
    axis.legend()
    figure.tight_layout()
    figure_path = OUTPUT / "e1_cumulative_squared_return_gap.png"
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)
    for item in inference.values():
        item.pop("cumulative")
    payload = {
        "task": "E1",
        "spec_commit": SPEC_COMMIT,
        "status": "descriptive",
        "risk_target_not_return_strategy": True,
        "annualization_days": ANNUALIZATION,
        "sample_sd": "ddof_1",
        "solver": "scipy.optimize.minimize SLSQP",
        "confirm_dates_scored": False,
        "proxy_panel_max_abs": panel_gap,
        "no_forecasts": summary["no_forecasts"],
        "cutoffs": cutoffs,
        "rows": rows_out,
        "p5": inference,
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "e1_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    panel.close()
    forecasts.close()
    print(f"wrote {OUTPUT / 'e1_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
