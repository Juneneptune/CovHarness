"""Score one completed CONFIRM forecast directory. No model is refit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from covharness.evaluation.recalibration import recalibrate_forecast_sequence
from covharness.features.origin_state import measurement_stress_instruments
from covharness.inference.dm import diebold_mariano
from covharness.inference.fluctuation import giacomini_rossi_fluctuation
from covharness.inference.gw import giacomini_white_two_specifications, market_state_instruments
from covharness.inference.mz import (
    WEIGHTING_APPROXIMATE_PS21,
    WEIGHTING_NONE,
    mincer_zarnowitz_pooled_vech,
)
from covharness.portfolio.gmv import (
    LongOnlySolverError,
    drift_weights,
    fleming_kirby_ostdiek_fee,
    long_only_gmv_weights,
    subtract_turnover_cost,
    turnover,
    unconstrained_gmv_weights,
)
from covharness.inference.exceptions import DegenerateLossDifferentialError
from covharness.protocol.binance_screen import screen_mcs, screen_spa
from covharness.protocol.binance_validation import ensemble_covariances, score_forecast_cube

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL = REPO_ROOT / "data/processed/binance_five_asset_panel.npz"
OUTPUT = REPO_ROOT / "results/confirm"
JOBS = OUTPUT / "jobs"
INPUTS = OUTPUT / "inputs" / "confirm_measurements.npz"
B1B = REPO_ROOT / "results/post_screen/B1b/b1b_cutoffs.json"
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
ORDINARY = 2.0595385527532972
C7 = 2.4728093597276533
C29 = 2.4226501420943998
HEADLINE = ("HARQDRD01", "V01")
SECONDARIES = (("V01", "LSTM22"), ("V11", "LSTM22"))


def _json(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=_default) + "\n"


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value))


def _write(name: str, payload: dict) -> None:
    (OUTPUT / name).write_text(_json(payload), encoding="utf-8")


def _cutoffs() -> dict[str, float]:
    payload = json.loads(B1B.read_text(encoding="utf-8"))
    by_length = {
        int(block["block_length"]): float(block["size_adjusted_critical_value"])
        for block in payload["blocks"]
    }
    ordinary = float(payload["ordinary_critical_value"])
    if ordinary != ORDINARY or by_length[7] != C7 or by_length[29] != C29:
        raise SystemExit("B1b cutoffs do not match the preregistered values")
    return {"ordinary": ordinary, "c_7": by_length[7], "c_29": by_length[29]}


def _load_cube(model_id: str) -> tuple[np.ndarray | None, list[dict], dict]:
    seed_rows: list[dict] = []
    if model_id in SEEDED:
        cubes = []
        for seed in range(5):
            directory = JOBS / model_id / f"seed_{seed}"
            path = directory / "forecast.npz"
            summary_path = directory / "summary.json"
            if not path.is_file():
                return None, seed_rows, {"missing_seed": seed}
            blob = np.load(path)
            cubes.append(np.array(blob["forecasts"], dtype=float))
            if summary_path.is_file():
                seed_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
        return ensemble_covariances(cubes), seed_rows, {"seeds": 5}
    directory = JOBS / model_id
    path = directory / "forecast.npz"
    if not path.is_file():
        return None, [], {"missing": True}
    blob = np.load(path)
    summary = {}
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return np.array(blob["forecasts"], dtype=float), [], summary


def _p5(loss_a: np.ndarray, loss_b: np.ndarray, cutoffs: dict[str, float]) -> dict[str, object]:
    differential = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    if not np.isfinite(differential).all():
        return {"complete": False, "n_observations": int(np.isfinite(differential).sum())}
    result = diebold_mariano(differential, reference="ewc")
    statistic = float(result.statistic)
    blocks = [
        float(np.mean(differential[start : start + 125])) for start in (0, 125, 250, 375)
    ]
    return {
        "complete": True,
        "n_observations": int(differential.shape[0]),
        "mean_a": float(np.mean(loss_a)),
        "mean_b": float(np.mean(loss_b)),
        "mean_gap": float(result.mean_differential),
        "statistic": statistic,
        "p_value": float(result.p_value),
        "ordinary_critical_value": float(result.critical_value),
        "nu": int(result.hac_maxlags),
        "win_rate": float(np.mean(differential < 0.0)),
        "block_mean_gaps": blocks,
        "block_mean_signs": [int(np.sign(value)) for value in blocks],
        "reject_ordinary": abs(statistic) > cutoffs["ordinary"],
        "reject_c7": abs(statistic) > cutoffs["c_7"],
        "reject_c29": abs(statistic) > cutoffs["c_29"],
        "reject_all_three": abs(statistic) > cutoffs["ordinary"]
        and abs(statistic) > cutoffs["c_7"]
        and abs(statistic) > cutoffs["c_29"],
        "differential": differential,
    }


def _book(forecasts: np.ndarray, simple: np.ndarray, kind: str) -> dict[str, np.ndarray]:
    n_rows, n_assets = simple.shape
    weights = np.full((n_rows, n_assets), np.nan)
    gross = np.full(n_rows, np.nan)
    traded = np.full(n_rows, np.nan)
    for index in range(n_rows):
        try:
            if kind == "unconstrained":
                current = unconstrained_gmv_weights(forecasts[index])
            elif kind == "long_only":
                current = long_only_gmv_weights(forecasts[index])
            else:
                current = np.full(n_assets, 1.0 / n_assets)
        except (np.linalg.LinAlgError, LongOnlySolverError, ValueError):
            continue
        weights[index] = current
        gross[index] = float(current @ simple[index])
        if index == 0:
            traded[index] = 0.0
        elif np.isfinite(weights[index - 1]).all():
            drifted = drift_weights(weights[index - 1], simple[index - 1])
            traded[index] = turnover(current, drifted)
    return {"gross": gross, "turnover": traded}


def _gap_series(book_a: dict, book_b: dict, bps: float | None) -> np.ndarray:
    if bps is None:
        left, right = book_a["gross"], book_b["gross"]
    else:
        left = subtract_turnover_cost(book_a["gross"], book_a["turnover"], bps)
        right = subtract_turnover_cost(book_b["gross"], book_b["turnover"], bps)
    return left**2 - right**2


def _holm(first: dict, second: dict) -> dict[str, object]:
    rows = [("first", first), ("second", second)]
    ordered = sorted(rows, key=lambda item: float(item[1]["p_value"]))
    decisions = {}
    for rank, (name, row) in enumerate(ordered):
        threshold = 0.025 if rank == 0 else 0.05
        size_ok = bool(row["reject_c7"] and row["reject_c29"])
        decisions[name] = {
            "holm_threshold": threshold,
            "holm_reject_ordinary": float(row["p_value"]) <= threshold,
            "size_adjusted_both": size_ok,
            "secondary_claim": float(row["p_value"]) <= threshold and size_ok,
        }
    return {
        "smaller_p": ordered[0][0],
        "larger_p": ordered[1][0],
        "decisions_in_pair_order": {
            "V01_minus_LSTM22": decisions["first"],
            "V11_minus_LSTM22": decisions["second"],
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _robust_sentence(label: str, primary: dict, other: dict) -> str:
    if not primary.get("reject_all_three"):
        return f"There is no headline claim to describe as {label}."
    same_sign = np.sign(primary.get("mean_gap", 0.0)) == np.sign(other.get("mean_gap", 0.0))
    if other.get("reject_all_three") and same_sign:
        return f"The headline claim is {label}."
    return f"The headline claim is not {label}."


def _sentence_pair(title: str, row: dict) -> str:
    if not row.get("complete"):
        return f"{title} is incomplete."
    decision = "rejects" if row["reject_all_three"] else "does not reject"
    direction = "favours the first named model" if row["mean_gap"] < 0 else "favours the second named model"
    if row["mean_gap"] == 0:
        direction = "is zero"
    return (
        f"{title} has mean losses {_fmt(row['mean_a'])} and {_fmt(row['mean_b'])}. "
        f"The mean gap is {_fmt(row['mean_gap'])} and {direction}. "
        f"The P5 statistic is {_fmt(row['statistic'])} with ordinary p-value {_fmt(row['p_value'])}. "
        f"The absolute statistic {decision} at all three cutoffs. "
        f"The win rate is {_fmt(row['win_rate'])}. "
        f"The four block mean signs are {row['block_mean_signs']}."
    )


def _num(value) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.6f}"


def _report(headline: dict, tables: dict, extras: dict, deviations: str) -> str:
    primary = headline["primary_five_minute_qlike"]
    lines = [
        "# Binance CONFIRM report",
        "",
        "## 1 Headline",
        "",
        _sentence_pair("HARQDRD01 minus V01 on raw reduced QLIKE under the five-minute proxy", primary),
        "",
        "## 2 Loss and proxy robustness",
        "",
        _sentence_pair("The same pair on squared Frobenius under the five-minute proxy", headline["frobenius_five_minute"]),
        _sentence_pair("The same pair on reduced QLIKE under the 15-minute proxy", headline["qlike_fifteen_minute"]),
        _sentence_pair("The same pair on squared Frobenius under the 15-minute proxy", headline["frobenius_fifteen_minute"]),
        _robust_sentence("loss-robust", primary, headline["frobenius_five_minute"]),
        _robust_sentence("proxy-robust", primary, headline["qlike_fifteen_minute"]),
        "",
        "## 3 Secondary contrasts",
        "",
        _sentence_pair("V01 minus LSTM22 on five-minute reduced QLIKE", headline["secondary_qlike"][0]),
        _sentence_pair("V11 minus LSTM22 on five-minute reduced QLIKE", headline["secondary_qlike"][1]),
        "Holm uses the two ordinary p-values. A secondary claim also requires both size-adjusted cutoffs.",
    ]
    holm = headline.get("holm", {})
    for name, decision in holm.get("decisions_in_pair_order", {}).items():
        claim = "is made" if decision["secondary_claim"] else "is not made"
        lines.append(
            f"The secondary claim for {name} {claim}. "
            f"The Holm threshold is {decision['holm_threshold']} and the ordinary comparison is "
            f"{'a rejection' if decision['holm_reject_ordinary'] else 'not a rejection'}."
        )
    lines.extend(["", "## 4 SPA and MCS", ""])
    for loss_name, block in extras["spa_mcs"].items():
        if not isinstance(block, dict):
            continue
        for length in (29, 7):
            spa = block.get(f"spa_{length}")
            mcs = block.get(f"mcs_{length}")
            if isinstance(spa, dict) and "p_value_consistent" in spa:
                lines.append(
                    f"SPA on {loss_name} at block length {length} has consistent p-value "
                    f"{_num(spa['p_value_consistent'])}. The benchmark is {spa['benchmark']}. "
                    f"Exact ties are {spa['exact_benchmark_ties'] or 'none'}."
                )
            elif isinstance(spa, dict) and "error" in spa:
                lines.append(f"SPA on {loss_name} at block length {length} is undefined. {spa['error']}")
            if isinstance(mcs, dict) and "primary" in mcs:
                members = [
                    name for name, keep in mcs["primary"]["in_mcs_primary_alpha"].items() if keep
                ]
                lines.append(
                    f"The range MCS on {loss_name} at block length {length} and alpha 0.10 contains {', '.join(members)}."
                )
    lines.extend(
        [
            "",
            "## 5 Conditional and calibration diagnostics",
            "",
            "Giacomini-White, pooled Mincer-Zarnowitz, and Giacomini-Rossi are headline-pair diagnostics. They do not change the headline decision.",
        ]
    )
    gw = extras["diagnostics"].get("gw")
    if gw:
        lines.append(
            f"The market-state Giacomini-White statistic is {_num(gw['market_statistic'])} "
            f"with Bonferroni p-value {_num(gw['market_bonferroni'])}. "
            f"The measurement-stress statistic is {_num(gw['measurement_statistic'])} "
            f"with Bonferroni p-value {_num(gw['measurement_bonferroni'])}. "
            f"The per-test cutoff is {gw['per_test_cutoff']}."
        )
    for key, row in extras["diagnostics"].items():
        if not key.startswith("mz_"):
            continue
        lines.append(
            f"Pooled Mincer-Zarnowitz {key} has intercept {_num(row['alpha'])}, slope {_num(row['beta'])}, "
            f"and p-value {_num(row['p_value'])}."
        )
    rossi = extras["diagnostics"].get("giacomini_rossi")
    if rossi:
        decision = "rejects" if rossi["reject"] else "does not reject"
        lines.append(
            f"The Giacomini-Rossi maximum absolute statistic is {_num(rossi['max_abs_statistic'])}. "
            f"The critical value is {_num(rossi['critical_value'])}. The test {decision}."
        )
    lines.extend(
        [
            "",
            "## 6 Economic evaluation",
            "",
            "The primary gap is the no-cost squared simple return of the unconstrained global-minimum-variance book. Costed gaps and the six fee cells are variants. No gamma or cost is selected.",
        ]
    )
    for pair in extras["economic"].get("pairs", []):
        if not pair.get("complete"):
            lines.append(f"The economic pair {pair.get('left')} minus {pair.get('right')} is incomplete.")
            continue
        for gap_name in ("gross", "cost_5bps", "cost_10bps", "long_only_gross"):
            gap = pair["gaps"].get(gap_name, {})
            if not gap.get("complete"):
                lines.append(f"The {gap_name} gap {pair['left']} minus {pair['right']} is incomplete.")
                continue
            decision = "rejects" if gap["reject_all_three"] else "does not reject"
            lines.append(
                f"The {gap_name} squared-return gap {pair['left']} minus {pair['right']} has statistic "
                f"{_num(gap['statistic'])} and {decision} at all three cutoffs."
            )
        for row in pair.get("equal_weight", []):
            lines.append(
                f"The unconstrained book {row['model_id']} minus equal weight has statistic "
                f"{_num(row['statistic'])} and mean gap {_num(row['mean_gap'])}."
            )
        for fee in pair.get("fees", []):
            if "daily_fee" not in fee:
                continue
            lines.append(
                f"The fee for {pair['left']} minus {pair['right']} at gamma {fee['gamma']} and "
                f"{fee['bps']} basis points is {_num(fee['daily_fee'])} per day "
                f"and {_num(fee['fee_times_365'])} times 365."
            )
    lines.extend(["", "## 7 HARQ fallback sensitivity", "", headline["harq_fallback_sentence"], ""])
    reduced = extras["fallback"].get("reduced_headline")
    if isinstance(reduced, dict) and reduced.get("complete"):
        lines.append(_sentence_pair("The headline pair with fallback targets removed", reduced))
    lines.extend(
        [
            "",
            "## 8 Recalibration diagnostics",
            "",
            "R1 and R2 are diagnostics. The headline uses the raw forecast.",
        ]
    )
    for row in extras["recalibration"]:
        if row["model_id"] not in ("HARQDRD01", "V01", "LSTM22", "V11"):
            continue
        lines.append(
            f"{row['model_id']} {row['version']} mean reduced QLIKE is {_num(row['mean_qlike'])} "
            f"and mean squared Frobenius is {_num(row['mean_frobenius'])}."
        )
    lines.extend(
        [
            "",
            "## 9 Model table",
            "",
            "Mean reduced QLIKE and squared Frobenius are reported under both proxies. Stochastic rows are equal-weight covariance ensembles.",
            "",
        ]
    )
    for row in tables["models"]:
        q5 = "incomplete" if row["mean_qlike_5min"] is None else _fmt(row["mean_qlike_5min"])
        f5 = "incomplete" if row["mean_frobenius_5min"] is None else _fmt(row["mean_frobenius_5min"])
        q15 = "incomplete" if row["mean_qlike_15min"] is None else _fmt(row["mean_qlike_15min"])
        f15 = "incomplete" if row["mean_frobenius_15min"] is None else _fmt(row["mean_frobenius_15min"])
        lines.append(
            f"{row['model_id']} five-minute QLIKE {q5}, five-minute Frobenius {f5}, "
            f"15-minute QLIKE {q15}, 15-minute Frobenius {f15}."
        )
    lines.extend(["", "## Deviations", "", deviations, ""])
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cutoffs = _cutoffs()
    panel = np.load(PANEL)
    measured = np.load(INPUTS)
    dates = [str(value) for value in panel["dates"].tolist()]
    lookup = {day: index for index, day in enumerate(dates)}
    targets = [str(value) for value in measured["target_dates"].tolist()]
    origins = [str(value) for value in measured["origin_dates"].tolist()]
    target_rows = [lookup[day] for day in targets]
    origin_rows = [lookup[day] for day in origins]
    proxy5 = np.array(panel["rcov_5min"][target_rows], dtype=float)
    proxy15 = np.array(measured["rcov_15min"], dtype=float)
    origin_rcov = np.array(panel["rcov_5min"][origin_rows], dtype=float)
    simple = np.expm1(np.array(panel["daily_returns"][target_rows], dtype=float))
    losses5: dict[str, np.ndarray] = {}
    losses5_f: dict[str, np.ndarray] = {}
    losses15: dict[str, np.ndarray] = {}
    losses15_f: dict[str, np.ndarray] = {}
    cubes: dict[str, np.ndarray] = {}
    seed_tables: dict[str, list] = {}
    summaries: dict[str, dict] = {}
    model_rows = []
    # Score every available raw column. Missing models stay incomplete.
    for model_id in ROSTER:
        cube, seed_rows, summary = _load_cube(model_id)
        seed_tables[model_id] = seed_rows
        summaries[model_id] = summary
        if cube is None:
            model_rows.append(
                {
                    "model_id": model_id,
                    "mean_qlike_5min": None,
                    "mean_frobenius_5min": None,
                    "mean_qlike_15min": None,
                    "mean_frobenius_15min": None,
                    "complete": False,
                }
            )
            continue
        q5, f5, q5_fail, f5_fail = score_forecast_cube(cube, proxy5)
        q15, f15, q15_fail, f15_fail = score_forecast_cube(cube, proxy15)
        complete = bool(np.isfinite(q5).all() and np.isfinite(f5).all())
        if complete:
            losses5[model_id] = q5
            losses5_f[model_id] = f5
            cubes[model_id] = cube
        if np.isfinite(q15).all():
            losses15[model_id] = q15
        if np.isfinite(f15).all():
            losses15_f[model_id] = f15
        model_rows.append(
            {
                "model_id": model_id,
                "mean_qlike_5min": None if not np.isfinite(q5).all() else float(np.mean(q5)),
                "mean_frobenius_5min": None if not np.isfinite(f5).all() else float(np.mean(f5)),
                "mean_qlike_15min": None if not np.isfinite(q15).all() else float(np.mean(q15)),
                "mean_frobenius_15min": None if not np.isfinite(f15).all() else float(np.mean(f15)),
                "qlike_failures_5min": q5_fail,
                "frobenius_failures_5min": f5_fail,
                "qlike_failures_15min": q15_fail,
                "frobenius_failures_15min": f15_fail,
                "complete": complete,
                "repair_count": summary.get("repair_count"),
                "repaired_targets": summary.get("repaired_targets", []),
            }
        )
    def pair(store, left, right):
        if left not in store or right not in store:
            return {"complete": False, "left": left, "right": right}
        row = _p5(store[left], store[right], cutoffs)
        row["left"] = left
        row["right"] = right
        return row

    primary = pair(losses5, *HEADLINE)
    frobenius = pair(losses5_f, *HEADLINE)
    qlike15 = pair(losses15, *HEADLINE)
    frobenius15 = pair(losses15_f, *HEADLINE)
    secondary = [pair(losses5, left, right) for left, right in SECONDARIES]
    if primary.get("complete") and secondary[0].get("complete") and secondary[1].get("complete"):
        holm = _holm(secondary[0], secondary[1])
    else:
        holm = {"complete": False}
    if primary.get("complete"):
        differential = primary.pop("differential")
        figure = OUTPUT / "headline_cumulative_qlike.png"
        plt.figure(figsize=(8, 4))
        plt.plot(np.cumsum(differential))
        plt.xlabel("CONFIRM target index")
        plt.ylabel("Cumulative reduced QLIKE gap")
        plt.tight_layout()
        plt.savefig(figure)
        plt.close()
    else:
        figure = None
    for row in (primary, frobenius, qlike15, frobenius15, *secondary):
        row.pop("differential", None)
    # SPA and MCS on complete raw columns only.
    def panel_from(store):
        labels = [model_id for model_id in ROSTER if model_id in store]
        if len(labels) < 2 or "HARDRD01" not in labels:
            return None, labels
        return np.column_stack([store[model_id] for model_id in labels]), labels

    spa_mcs = {}
    for loss_name, store in (
        ("qlike_5min", losses5),
        ("frobenius_5min", losses5_f),
        ("qlike_15min", losses15),
        ("frobenius_15min", losses15_f),
    ):
        matrix, labels = panel_from(store)
        spa_mcs[loss_name] = {"labels": labels}
        if matrix is None:
            continue
        for block in (29, 7):
            try:
                spa_mcs[loss_name][f"spa_{block}"] = screen_spa(
                    matrix, labels, benchmark="HARDRD01", block_length=block
                )
            except DegenerateLossDifferentialError as exc:
                spa_mcs[loss_name][f"spa_{block}"] = {"error": str(exc)}
            spa_mcs[loss_name][f"mcs_{block}"] = screen_mcs(
                matrix, labels, block_length=block
            )
    diagnostics = {}
    if primary.get("complete"):
        market = market_state_instruments(origin_rcov)
        stress = measurement_stress_instruments(measured["origin_returns_5min"])
        gw = giacomini_white_two_specifications(
            losses5["HARQDRD01"], losses5["V01"], market, stress
        )
        diagnostics["gw"] = {
            "market_statistic": gw.market.statistic,
            "market_p_value": gw.market.p_value,
            "market_bonferroni": gw.p_value_market_bonferroni,
            "measurement_statistic": gw.measurement.statistic,
            "measurement_p_value": gw.measurement.p_value,
            "measurement_bonferroni": gw.p_value_measurement_bonferroni,
            "per_test_cutoff": gw.per_test_cutoff,
        }
        origin_dates = origins
        target_dates = targets
        segments = ["CONFIRM"] * len(targets)
        refit = np.array([(index % 21) == 0 for index in range(len(targets))])
        for label, weighting in (
            ("ols", WEIGHTING_NONE),
            ("approximate_ps21", WEIGHTING_APPROXIMATE_PS21),
        ):
            for model_id in ("HARQDRD01", "V01"):
                fitted = mincer_zarnowitz_pooled_vech(
                    proxy5, cubes[model_id], weighting=weighting
                )
                diagnostics[f"mz_{model_id}_{label}"] = {
                    "alpha": fitted.alpha,
                    "beta": fitted.beta,
                    "wald": fitted.wald_statistic,
                    "p_value": fitted.p_value,
                }
        fluctuation = giacomini_rossi_fluctuation(losses5["HARQDRD01"], losses5["V01"], mu=0.30)
        diagnostics["giacomini_rossi"] = {
            "max_abs_statistic": fluctuation.max_abs_statistic,
            "critical_value": fluctuation.critical_value,
            "reject": fluctuation.reject,
            "window_length": fluctuation.window_length,
        }
        del origin_dates, target_dates, segments, refit
    # Economic channel on raw unconstrained books.
    economic = {"pairs": []}
    books = {}
    for model_id, cube in cubes.items():
        books[(model_id, "unconstrained")] = _book(cube, simple, "unconstrained")
        books[(model_id, "long_only")] = _book(cube, simple, "long_only")
    equal = _book(np.zeros((len(simple), 5, 5)), simple, "equal")
    for left, right in (HEADLINE, *SECONDARIES):
        if (left, "unconstrained") not in books or (right, "unconstrained") not in books:
            economic["pairs"].append({"left": left, "right": right, "complete": False})
            continue
        entry = {"left": left, "right": right, "complete": True, "gaps": {}}
        for name, bps, kind in (
            ("gross", None, "unconstrained"),
            ("cost_5bps", 5.0, "unconstrained"),
            ("cost_10bps", 10.0, "unconstrained"),
            ("long_only_gross", None, "long_only"),
        ):
            series = _gap_series(books[(left, kind)], books[(right, kind)], bps)
            finite = np.isfinite(series)
            if not finite.all():
                entry["gaps"][name] = {"complete": False, "n_finite": int(finite.sum())}
            else:
                result = diebold_mariano(series, reference="ewc")
                statistic = float(result.statistic)
                entry["gaps"][name] = {
                    "complete": True,
                    "mean_gap": float(result.mean_differential),
                    "statistic": statistic,
                    "p_value": float(result.p_value),
                    "reject_all_three": abs(statistic) > ORDINARY
                    and abs(statistic) > C7
                    and abs(statistic) > C29,
                }
        fees = []
        for gamma in (1.0, 5.0, 10.0):
            for bps in (5.0, 10.0):
                net_a = subtract_turnover_cost(
                    books[(left, "unconstrained")]["gross"],
                    books[(left, "unconstrained")]["turnover"],
                    bps,
                )
                net_b = subtract_turnover_cost(
                    books[(right, "unconstrained")]["gross"],
                    books[(right, "unconstrained")]["turnover"],
                    bps,
                )
                if not np.isfinite(net_a).all() or not np.isfinite(net_b).all():
                    fees.append({"gamma": gamma, "bps": bps, "complete": False})
                else:
                    fees.append(
                        {
                            "gamma": gamma,
                            "bps": bps,
                            "daily_fee": fleming_kirby_ostdiek_fee(net_a, net_b, gamma),
                            "fee_times_365": 365.0 * fleming_kirby_ostdiek_fee(net_a, net_b, gamma),
                        }
                    )
        entry["fees"] = fees
        if np.isfinite(equal["gross"]).all():
            versus_equal = []
            for model_id in (left, right):
                series = books[(model_id, "unconstrained")]["gross"] ** 2 - equal["gross"] ** 2
                result = diebold_mariano(series, reference="ewc")
                versus_equal.append(
                    {
                        "model_id": model_id,
                        "statistic": float(result.statistic),
                        "p_value": float(result.p_value),
                        "mean_gap": float(result.mean_differential),
                    }
                )
            entry["equal_weight"] = versus_equal
        economic["pairs"].append(entry)
    fallback_dates = summaries.get("HARQDRD01", {}).get("repaired_targets", [])
    if not fallback_dates:
        fallback_sentence = "HARQDRD01 recorded no estimation-window-mean fallback on a CONFIRM target."
        fallback = {"count": 0, "targets": []}
    elif primary.get("complete"):
        keep = np.array([day not in set(fallback_dates) for day in targets])
        reduced = _p5(losses5["HARQDRD01"][keep], losses5["V01"][keep], cutoffs)
        reduced.pop("differential", None)
        fallback = {"count": len(fallback_dates), "targets": fallback_dates, "reduced_headline": reduced}
        fallback_sentence = (
            f"HARQDRD01 used a fallback on {len(fallback_dates)} CONFIRM targets. "
            "The primary table keeps those targets. The reduced pair is a sensitivity."
        )
    else:
        fallback = {"count": len(fallback_dates), "targets": fallback_dates}
        fallback_sentence = "HARQDRD01 recorded fallback targets and the headline pair is incomplete."
    recalibration = []
    if "HARQDRD01" in cubes:
        refit = np.array([(index % 21) == 0 for index in range(len(targets))])
        for model_id, cube in cubes.items():
            mapped = recalibrate_forecast_sequence(
                cube,
                proxy5,
                origin_dates=origins,
                target_dates=targets,
                segments=["CONFIRM"] * len(targets),
                refit=refit,
            )
            for version, transformed in (("r1", mapped.r1), ("r2", mapped.r2)):
                qlike_diag, frobenius_diag, q_fail, f_fail = score_forecast_cube(transformed, proxy5)
                recalibration.append(
                    {
                        "model_id": model_id,
                        "version": version,
                        "mean_qlike": None if not np.isfinite(qlike_diag).all() else float(np.mean(qlike_diag)),
                        "mean_frobenius": None
                        if not np.isfinite(frobenius_diag).all()
                        else float(np.mean(frobenius_diag)),
                        "qlike_failures": q_fail,
                        "frobenius_failures": f_fail,
                        "role": "diagnostic",
                    }
                )
    headline = {
        "primary_five_minute_qlike": primary,
        "frobenius_five_minute": frobenius,
        "qlike_fifteen_minute": qlike15,
        "frobenius_fifteen_minute": frobenius15,
        "secondary_qlike": secondary,
        "holm": holm,
        "harq_fallback_sentence": fallback_sentence,
        "cumulative_plot": None if figure is None else figure.name,
    }
    _write("headline.json", headline)
    _write("spa_mcs.json", spa_mcs)
    _write("diagnostics.json", diagnostics)
    _write("economic.json", economic)
    _write("harq_fallback.json", fallback)
    _write("recalibration.json", {"rows": recalibration})
    _write("model_table.json", {"models": model_rows, "seeds": seed_tables})
    deviations = (
        "V01 uses the return-only input. The shared fit signature still requires a feature array, "
        "so the runner passed a column of zeros that V01 does not read. The forecast is unchanged by that array."
    )
    (OUTPUT / "DEVIATIONS.md").write_text(
        "# Deviations\n\n"
        "V01 uses the return-only input. The shared fit signature still requires a feature array, "
        "so the runner passed a column of zeros that V01 does not read. "
        "The forecast is unchanged by that array. The headline definition is unchanged.\n",
        encoding="utf-8",
    )
    report = _report(
        headline,
        {"models": model_rows},
        {
            "spa_mcs": spa_mcs,
            "diagnostics": diagnostics,
            "economic": economic,
            "fallback": fallback,
            "recalibration": recalibration,
        },
        deviations,
    )
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("scored", sum(row["complete"] for row in model_rows), "complete columns", flush=True)


if __name__ == "__main__":
    main()
