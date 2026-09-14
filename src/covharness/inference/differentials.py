"""Pairwise loss-differential convention.

For forecasts A and B with losses ``L_{A,t}`` and ``L_{B,t}``,

    d_t = L_{A,t} - L_{B,t}.

Then ``d_t < 0`` means A has lower loss on date t, and ``d_t > 0`` means B
has lower loss. The equal-accuracy null is ``H0: E[d_t] = 0``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Named alternatives for tests of E[d_t].
ALTERNATIVE_TWO_SIDED = "two_sided"
ALTERNATIVE_A_BETTER = "a_better"
ALTERNATIVE_B_BETTER = "b_better"
VALID_ALTERNATIVES = (
    ALTERNATIVE_TWO_SIDED,
    ALTERNATIVE_A_BETTER,
    ALTERNATIVE_B_BETTER,
)


def as_1d_finite(values: ArrayLike, name: str) -> NDArray[np.floating]:
    """Copy ``values`` to a finite 1-d array. The input is not mutated."""
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-d series; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite (NaN and inf are rejected)")
    return array


def loss_differential(loss_a: ArrayLike, loss_b: ArrayLike) -> NDArray[np.floating]:
    """Return ``d_t = L_{A,t} - L_{B,t}``.

    Negative entries are dates on which A wins. Positive entries are dates on
    which B wins.
    """
    a = as_1d_finite(loss_a, "loss_a")
    b = as_1d_finite(loss_b, "loss_b")
    if a.shape != b.shape:
        raise ValueError(
            f"loss_a and loss_b must have the same length; got {a.shape} and {b.shape}"
        )
    return a - b


def parse_alternative(alternative: str) -> str:
    """Accept the documented alternative names only."""
    key = str(alternative).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "two_sided": ALTERNATIVE_TWO_SIDED,
        "twosided": ALTERNATIVE_TWO_SIDED,
        "a_better": ALTERNATIVE_A_BETTER,
        "a_wins": ALTERNATIVE_A_BETTER,
        "less": ALTERNATIVE_A_BETTER,
        "b_better": ALTERNATIVE_B_BETTER,
        "b_wins": ALTERNATIVE_B_BETTER,
        "greater": ALTERNATIVE_B_BETTER,
    }
    if key not in aliases:
        raise ValueError(
            f"alternative must be one of {VALID_ALTERNATIVES}; got {alternative!r}"
        )
    return aliases[key]
