"""Predictive inference.

Diebold-Mariano tests with Bartlett / Newey-West HAC standard errors.
Hansen (2005) SPA and Hansen-Lunde-Nason (2011) MCS compare a universe of
models on one loss/proxy channel. Clark-West is a separate nested scalar
squared-error procedure. One-step Giacomini-White tests, pooled-vech
Mincer-Zarnowitz calibration, and the Giacomini-Rossi fluctuation test
are implemented as Block 3C inference. Origin-day aggregate realized quarticity
and the BNS equal-weight market jump indicator supply the second GW
specification. Block 3C is not declared closed here.
"""

from covharness.features.origin_state import MEASUREMENT_STATE_SPECIFICATION
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
    DegenerateFluctuationVarianceError,
    DegenerateGWCovarianceError,
    DegenerateLossDifferentialError,
    DegenerateMCSDifferentialError,
    RankDeficientGWInstrumentsError,
    RankDeficientMZDesignError,
)
from covharness.inference.fluctuation import (
    FLUCTUATION_ALPHA,
    FLUCTUATION_CRITICAL_VALUE,
    FLUCTUATION_MU,
    FluctuationResult,
    giacomini_rossi_fluctuation,
)
from covharness.inference.gw import (
    GW_ALPHA,
    GW_BONFERRONI_CUTOFF,
    GW_FAMILY_ALPHA,
    GW_FAMILY_SIZE,
    BonferroniFamilyResult,
    GWFamilyResult,
    GWResult,
    giacomini_white,
    giacomini_white_two_specifications,
    market_state_instruments,
    measurement_stress_instruments,
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
from covharness.inference.mz import (
    INFERENCE_DETERMINISTIC_ALTERNATIVE,
    INFERENCE_DETERMINISTIC_NULL,
    INFERENCE_REGULAR,
    MZ_ALPHA,
    WEIGHTING_APPROXIMATE_PS21,
    WEIGHTING_NONE,
    MZResult,
    mincer_zarnowitz_pooled_vech,
    mincer_zarnowitz_two_finalists,
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
    "BonferroniFamilyResult",
    "ClarkWestResult",
    "ClarkWestScopeError",
    "DEFAULT_N_BOOT",
    "DegenerateFluctuationVarianceError",
    "DegenerateGWCovarianceError",
    "DegenerateLossDifferentialError",
    "DegenerateMCSDifferentialError",
    "DieboldMarianoResult",
    "FLUCTUATION_ALPHA",
    "FLUCTUATION_CRITICAL_VALUE",
    "FLUCTUATION_MU",
    "FluctuationResult",
    "GWFamilyResult",
    "GWResult",
    "GW_ALPHA",
    "GW_BONFERRONI_CUTOFF",
    "GW_FAMILY_ALPHA",
    "GW_FAMILY_SIZE",
    "HACResult",
    "HACSizePowerResult",
    "INFERENCE_DETERMINISTIC_ALTERNATIVE",
    "INFERENCE_DETERMINISTIC_NULL",
    "INFERENCE_REGULAR",
    "LossDifferentialDiagnostic",
    "MCSResult",
    "MCS_ALPHA",
    "MEASUREMENT_STATE_SPECIFICATION",
    "MZResult",
    "MZ_ALPHA",
    "PROCEDURE_MAX",
    "PROCEDURE_RANGE",
    "RankDeficientGWInstrumentsError",
    "RankDeficientMZDesignError",
    "SPAResult",
    "SPA_ALPHA",
    "WEIGHTING_APPROXIMATE_PS21",
    "WEIGHTING_NONE",
    "bartlett_weights",
    "clark_west_squared_error",
    "default_block_length",
    "diebold_mariano",
    "diebold_mariano_from_losses",
    "giacomini_rossi_fluctuation",
    "giacomini_white",
    "giacomini_white_two_specifications",
    "hac_long_run_variance",
    "hansen_lil_threshold",
    "loss_differential",
    "loss_differential_diagnostics",
    "market_state_instruments",
    "measurement_stress_instruments",
    "mincer_zarnowitz_pooled_vech",
    "mincer_zarnowitz_two_finalists",
    "model_confidence_set",
    "naive_iid_t_statistic",
    "newey_west_1994_lags",
    "plot_hac_size_power",
    "simulate_hac_size_power",
    "stationary_bootstrap_indices",
    "superior_predictive_ability",
]
