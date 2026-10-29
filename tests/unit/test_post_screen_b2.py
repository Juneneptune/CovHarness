"""Block-length pass-through for the existing SCREEN SPA and MCS wrappers."""

from __future__ import annotations

import numpy as np

from covharness.inference.bootstrap import default_block_length
from covharness.protocol.binance_screen import screen_mcs, screen_spa


def test_explicit_block_length_is_recorded_and_default_matches() -> None:
    rng = np.random.default_rng(0)
    losses = rng.random((500, 3))
    labels = ["HAR-DRD", "EWMA", "RW"]
    explicit = screen_spa(losses, labels, n_boot=20, seed=1, block_length=14)
    assert explicit["block_length"] == 14
    default = screen_spa(losses, labels, n_boot=20, seed=1)
    forced = screen_spa(
        losses,
        labels,
        n_boot=20,
        seed=1,
        block_length=default_block_length(500),
    )
    assert default["p_value_consistent"] == forced["p_value_consistent"]
    assert default["p_value_lower"] == forced["p_value_lower"]
    assert default["p_value_upper"] == forced["p_value_upper"]
    mcs = screen_mcs(losses, labels, n_boot=20, seed=1, block_length=14)
    assert mcs["primary"]["block_length"] == 14
    assert mcs["companion"]["block_length"] == 14
    assert mcs["primary"]["procedure"] != mcs["companion"]["procedure"]
