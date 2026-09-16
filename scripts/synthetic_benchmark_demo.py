"""Demonstration of synthetic rolling forecasts scored by existing losses.

DEMONSTRATION SETTINGS. Hyperparameters below are chosen only to keep the
example short. They are not tuned and are not research configurations.

These numbers are integration diagnostics, not empirical benchmark results.
"""

from __future__ import annotations

import time

import pandas as pd

from covharness.evaluation import (
    LOSS_REDUCED_QLIKE,
    LOSS_SQUARED_FROBENIUS,
    build_loss_panel,
    score_forecast_records,
    summarize_loss_panel,
    target_covariance_panel,
)
from covharness.models import (
    DCCCovariance,
    DCCNonlinearCovariance,
    EWMARealizedCovariance,
    HARDRDRealizedCovariance,
    HARQDRDRealizedCovariance,
    LSTMBEKKCovariance,
    LedoitWolfLinearCovariance,
    LedoitWolfNonlinearCovariance,
    RandomWalkRealizedCovariance,
    RidgeDRDRealizedCovariance,
    XGBoostDRDRealizedCovariance,
)
from covharness.protocol import (
    DEFAULT_REFIT_CADENCE,
    build_schedule,
    run_rolling_forecasts,
)
from covharness.simulation import synthetic_benchmark_panel

DEMO_SEED = 20260916
DEMO_N_ASSETS = 3
DEMO_WINDOW = 40
DEMO_N_TARGETS = 8
DEMO_N_TIMES = DEMO_WINDOW + DEMO_N_TARGETS


def demonstration_models() -> list[object]:
    """Return a representative roster with explicit demonstration hyperparameters."""
    return [
        RandomWalkRealizedCovariance(),
        EWMARealizedCovariance(decay=0.94),
        HARDRDRealizedCovariance(),
        HARQDRDRealizedCovariance(),
        RidgeDRDRealizedCovariance(lambda_=1.0),
        XGBoostDRDRealizedCovariance(
            n_estimators=2,
            max_depth=1,
            learning_rate=0.3,
            min_child_weight=0.0,
            reg_lambda=0.0,
            reg_alpha=0.0,
            gamma=0.0,
        ),
        LedoitWolfLinearCovariance(),
        LedoitWolfNonlinearCovariance(),
        DCCCovariance(),
        DCCNonlinearCovariance(),
        LSTMBEKKCovariance(
            seed=0,
            num_layers=3,
            dropout=0.1,
            learning_rate=0.05,
            gradient_clip_norm=1.0,
            max_epochs=1,
        ),
    ]


def _print_summary_table(title: str, summaries) -> None:
    """Print roster-ordered descriptive statistics. This is not a ranking."""
    frame = pd.DataFrame(
        [
            {
                "model": row.model_name,
                "n": row.n_forecasts,
                "mean": row.mean_loss,
                "median": row.median_loss,
                "std": row.std_loss,
            }
            for row in summaries
        ]
    )
    print(title)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()


def main() -> None:
    print("These numbers are integration diagnostics, not empirical benchmark results.")
    print("DEMONSTRATION SETTINGS. These hyperparameters are not tuned.")
    print("CONFIRM remains locked. SCREEN is unused. No empirical data are used.")
    print()
    started = time.perf_counter()
    panel = synthetic_benchmark_panel(
        n_times=DEMO_N_TIMES,
        n_assets=DEMO_N_ASSETS,
        seed=DEMO_SEED,
    )
    schedule = build_schedule(
        panel.calendar,
        panel.calendar[DEMO_WINDOW:],
        m=DEMO_WINDOW,
        refit_cadence=DEFAULT_REFIT_CADENCE,
    )
    targets = target_covariance_panel(panel.calendar, panel.realized_covariances)
    models = demonstration_models()
    all_records = []
    for model in models:
        records = run_rolling_forecasts(
            model=model,
            schedule=schedule,
            calendar=panel.calendar,
            daily_returns=panel.daily_returns,
            realized_covariances=panel.realized_covariances,
            realized_quarticity=panel.realized_quarticity,
        )
        all_records.extend(records)
        print(
            f"forecasts {records[0].model_name}: {len(records)} "
            f"targets from {records[0].target.date()} to {records[-1].target.date()}"
        )
    scored = score_forecast_records(all_records, targets)
    qlike = build_loss_panel(
        scored,
        loss_name=LOSS_REDUCED_QLIKE,
        model_order=[model.identity.name for model in models],
    )
    frobenius = build_loss_panel(
        scored,
        loss_name=LOSS_SQUARED_FROBENIUS,
        model_order=qlike.model_names,
    )
    print()
    _print_summary_table(
        "Reduced multivariate QLIKE (descriptive only)",
        summarize_loss_panel(qlike),
    )
    _print_summary_table(
        "Squared Frobenius (descriptive only)",
        summarize_loss_panel(frobenius),
    )
    elapsed = time.perf_counter() - started
    print(
        f"demo roster {list(qlike.model_names)}; "
        f"forecast count {len(all_records)}; wall time {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
