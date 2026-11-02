"""Leave-one-out P5 drops the planted outlier and no other index."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load():
    path = Path(__file__).resolve().parents[2] / "scripts" / "exploratory" / "x3_crash_day.py"
    spec = importlib.util.spec_from_file_location("x3_crash_day", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_planted_outlier_has_the_minimum_and_the_largest_influence():
    module = _load()
    series = np.linspace(-0.25, 0.25, 80)
    planted = 17
    series = series.copy()
    series[planted] = 40.0
    full = module.p5_statistic(series)
    held_out = module.leave_one_out_statistics(series)
    assert held_out.shape == (80,)
    assert int(np.argmin(held_out)) == planted
    influence = np.abs(full - held_out)
    assert int(np.argmax(influence)) == planted
    assert held_out[planted] < full


def test_leave_one_out_preserves_order_of_the_remaining_points():
    module = _load()
    series = np.array([0.2, -0.1, 0.4, -0.3, 0.5, 1.5], dtype=float)
    direct = module.p5_statistic(np.concatenate([series[:2], series[3:]]))
    held_out = module.leave_one_out_statistics(series)
    assert held_out[2] == pytest.approx(direct)


def test_short_series_is_rejected():
    module = _load()
    with pytest.raises(ValueError):
        module.leave_one_out_statistics(np.array([0.1, -0.2]))
