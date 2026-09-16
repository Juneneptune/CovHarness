"""Synthetic data used to validate estimators, losses, and the evaluation adapter."""

from covharness.simulation.benchmark import (
    DEFAULT_BENCHMARK_SEED,
    SyntheticBenchmarkPanel,
    synthetic_benchmark_panel,
)
from covharness.simulation.intraday import simulate_synchronized_gaussian_returns

__all__ = [
    "DEFAULT_BENCHMARK_SEED",
    "SyntheticBenchmarkPanel",
    "simulate_synchronized_gaussian_returns",
    "synthetic_benchmark_panel",
]
