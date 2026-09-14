"""Predictive inference.

Diebold-Mariano tests with Bartlett / Newey-West HAC standard errors.
Hansen (2005) SPA and Hansen-Lunde-Nason (2011) MCS compare a universe of
models on one loss/proxy channel. Clark-West is a separate nested scalar
squared-error procedure. Giacomini-White and Mincer-Zarnowitz are not
implemented here.
"""

from covharness.inference.bootstrap import (
    BOOTSTRAP_METHOD,
    BOOTSTRAP_SEED,
    DEFAULT_N_BOOT,
    default_block_length,
    stationary_bootstrap_indices,
)
from covharness.inference.clark_west import ClarkWestResult, clark_west_squared_error
from covharness.inference.diagnostics import (
    LossDifferentialDiagnostic,
    loss_differential_diagnostics,
)
from covharness.inference.differentials import (
    ALTERNATIVE_A_BETTER,
    ALTERNATIVE_B_BETTER,
    ALTERNATIVE_TWO_SIDED,
    loss_differential,
)
from covharness.inference.dm import (
    DieboldMarianoResult,
    diebold_mariano,
    diebold_mariano_from_losses,
    naive_iid_t_statistic,
)
from covharness.inference.exceptions import (
    ClarkWestScopeError,
    DegenerateLossDifferentialError,
    DegenerateMCSDifferentialError,
)
from covharness.inference.hac import (
    HACResult,
    bartlett_weights,
    hac_long_run_variance,
    newey_west_1994_lags,
)
from covharness.inference.mcs import (
    MCS_ALPHA,
    MCSResult,
    PROCEDURE_MAX,
    PROCEDURE_RANGE,
    model_confidence_set,
)
from covharness.inference.size import (
    HACSizePowerResult,
    plot_hac_size_power,
    simulate_hac_size_power,
)
from covharness.inference.spa import (
    SPA_ALPHA,
    SPAResult,
    hansen_lil_threshold,
    superior_predictive_ability,
)

__all__ = [
    "ALTERNATIVE_A_BETTER",
    "ALTERNATIVE_B_BETTER",
    "ALTERNATIVE_TWO_SIDED",
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_SEED",
    "ClarkWestResult",
    "ClarkWestScopeError",
    "DEFAULT_N_BOOT",
    "DegenerateLossDifferentialError",
    "DegenerateMCSDifferentialError",
    "DieboldMarianoResult",
    "HACResult",
    "HACSizePowerResult",
    "LossDifferentialDiagnostic",
    "MCSResult",
    "MCS_ALPHA",
    "PROCEDURE_MAX",
    "PROCEDURE_RANGE",
    "SPAResult",
    "SPA_ALPHA",
    "bartlett_weights",
    "clark_west_squared_error",
    "default_block_length",
    "diebold_mariano",
    "diebold_mariano_from_losses",
    "hac_long_run_variance",
    "hansen_lil_threshold",
    "loss_differential",
    "loss_differential_diagnostics",
    "model_confidence_set",
    "naive_iid_t_statistic",
    "newey_west_1994_lags",
    "plot_hac_size_power",
    "simulate_hac_size_power",
    "stationary_bootstrap_indices",
    "superior_predictive_ability",
]
