# Preregistration draft

This document records protocol decisions that are frozen for implementation. It is a draft. It is not immutable and it is not the final preregistration.

Final `PREREGISTRATION.md` is created once, after the exact graph-neural model specification and the empirical dataset are frozen, and before any Block-4 empirical model fitting begins. Once that file is created it is never edited. Later departures must be recorded separately as deviations.

The graph-neural deep-learning slot is still unresolved. The long historical empirical panel is still unresolved. The exact covariance-calibration representation and MZ-GLS weighting rule, the Giacomini–Rossi fluctuation window / critical-value specification, and any optional state-dependent threshold design are also not yet frozen. Therefore a final preregistration would be false.

---

## Primary forecast horizon

The forecast is the one-day-ahead conditional covariance matrix $\Sigma_{t+1\mid t}$. Multi-step horizons are out of scope for the committed comparison.

## Measurement proxies

Statistical evaluation uses the open-to-close realized-covariance proxy already defined in the harness. The primary sampling grid is five-minute previous-tick realized covariance. Five-minute subsampled realized covariance is the implemented complementary proxy. Realized kernels remain planned and are not yet implemented.

No proxy is chosen after seeing which one favours a model.

## Primary losses

The primary ranking loss is reduced multivariate QLIKE. Squared Frobenius is the complementary proxy-robust ranking loss. Full Stein is used when the proxy is strictly positive definite. Ordinary unsquared Frobenius is a labeled non-robust contrast only. Localization diagnostics are descriptive and are not ranking losses.

Training objectives, statistical evaluation losses, and later economic criteria remain separate.

## VALIDATION, SCREEN, and CONFIRM

The calendar is split into four chronological regions. HISTORY (rolling estimation burn-in), VALIDATION, SCREEN, and CONFIRM. Ordering is HISTORY < VALIDATION < SCREEN < CONFIRM. There is no shuffling. Evaluation target dates do not overlap across VALIDATION, SCREEN, and CONFIRM.

Boundary convention. Intervals are half-open on the ordered trading-date index, written $[start, end)$. Evaluation blocks are sets of forecast targets (day $t+1$). For a target at calendar position $i$, the origin is position $i-1$ and the estimation window is the half-open span $[i-m, i)$.

VALIDATION may be used for hyperparameter tuning, early stopping, architecture or configuration selection within the preregistered search space, and training-objective selection if that choice is preregistered.

SCREEN may be used to compare frozen candidate configurations, to apply MCS / SPA screening once Block 3 exists, and to select finalists under a frozen rule. SCREEN must not be used to retune hyperparameters after its results are seen.

CONFIRM is used for final locked confirmatory comparisons only. No tuning, no architecture changes, no proxy or loss changes, and no seed selection.

## Block-size rules

VALIDATION has target length 250 trading days. SCREEN has a committed minimum of 500 trading days. CONFIRM has a committed minimum of 500 trading days. SCREEN and CONFIRM are never shortened. If a calendar cannot support the preferred 250/500/500 evaluation allocation, shorten VALIDATION first and report that the realized VALIDATION length is below target. If VALIDATION has length zero, data-driven tuning is unavailable and must not be described as completed. If $T-m < 1000$, the confirmatory design is infeasible and allocation fails.

Surplus evaluation days beyond 1250 are assigned to VALIDATION.

## Rolling window and refit cadence

The common forecasting method uses a rolling estimation window of $m=250$ trading days and a monthly refit cadence of 21 trading days. At origin $t$, inputs may use observations through $t$. The target is $t+1$. No $t+1$ information enters fitting, transformation, feature creation, graph construction, scaling, or hyperparameter selection. The same origin and refit schedule will be reused by conventional, machine-learning, and deep-learning models.

Preprocessing estimators are fit on the estimation window through $t$ only. VALIDATION may be transformed but must not be used to fit a scaler. SCREEN and CONFIRM must not affect preprocessing estimates. Full-sample scaling is forbidden.

## Tuning parity

Each model family receives an auditable maximum of 20 candidate configurations. The protocol records the budget. It does not yet run a search engine. A later model must not silently receive a larger search.

## Seed rules

Stochastic methods use the pre-specified seed list $(0,1,2,3,4)$. All seeds are reported. Best-seed selection is forbidden. The confirmatory aggregation rule is the equal-weight mean ensemble across seeds, frozen before CONFIRM.

## Confirm lock

CONFIRM is locked by default (`LOCKED = True`). Retrieving or evaluating CONFIRM observations raises `ConfirmLockedError` unless `unlock_confirm=True` is passed at the protocol boundary. The later command-line runner will expose the same action as `--unlock-confirm`. The default remains locked.

## Statistical and economic overnight convention

Statistical covariance evaluation uses the open-to-close realized-covariance proxy as currently defined.

Economic global-minimum-variance evaluation, when later implemented, measures realized risk including the overnight return outer product, so that the proxy corresponds to a portfolio held across the overnight period. An open-to-close-only economic version is retained as a robustness channel.

GMV portfolios and overnight realized-covariance construction are not implemented in this draft. The convention is frozen before results exist.

## Current conventional and shallow roster

The committed first-stage roster is random-walk realized covariance, EWMA, HAR-DRD, HARQ-DRD, Ledoit–Wolf linear shrinkage, Ledoit–Wolf nonlinear shrinkage, DCC, DCC-NL, and Ridge-DRD.

XGBoost-DRD remains a planned shallow-learning control. It is not implemented here.

## Deep and structured extensions recorded as draft commitments

LSTM-BEKK is a committed deep-learning competitor that uses daily returns.

LSTM-BEKK-RC is a committed information-parity counterpart.

GHAR is a Block-4 structured graph / econometric covariance baseline. It is not a deep-learning model.

One graph-neural deep-learning slot remains explicitly unresolved. That specification must be frozen before final `PREREGISTRATION.md`.

iTransformer is optional and is not committed.

## Primary inference procedures

Diebold-Mariano tests with Bartlett / Newey-West HAC standard errors are implemented for pairwise comparisons. The Harvey-Leybourne-Newbold correction is not applied. Clark-West is available only for explicitly nested scalar squared-error comparisons. It is not a default adjustment for QLIKE or squared Frobenius.

Before final confirmatory use, the implemented DM procedure receives a bounded dependence / calibration review. The reported persistent-AR(1) size distortion is treated as a limitation to investigate rather than as satisfactory nominal calibration or as proof of a coding defect. The current Bartlett / Newey-West DM remains the baseline unless a separately reviewed robustness procedure is explicitly adopted. No new default is frozen in this draft.

SPA and the Model Confidence Set are implemented for Block 3B screening. They are applied separately to each pre-specified loss/proxy channel. Reduced QLIKE and squared Frobenius are never averaged, and distinct covariance proxies are never pooled.

The SPA benchmark is supplied by the caller. HAR-DRD remains the planned empirical benchmark unless changed explicitly before empirical results. The SPA differential is $d_{k,t}=L_{0,t}-L_{k,t}$. A positive value means the alternative beats the benchmark. The null is that no alternative has strictly lower expected loss. The studentized Hansen (2005) statistic, stationary-bootstrap geometric long-run variance, and three recenterings are used. The primary p-value is the consistent recentering. The headline SPA level is $0.05$. Bootstrap p-values use $\mathrm{mean}(T^\ast>T)$. Production resampling uses $B=5000$ draws, seed $20260913$, and expected block length $\max(2,\lfloor T^{1/3}\rfloor)$.

The Model Confidence Set implements both coherent Hansen–Lunde–Nason pairs. The primary SCREEN procedure is $(T_R,e_R)$. The companion is $(T_{\max},e_{\max})$. The frozen SCREEN membership level is $\alpha=0.10$. Model p-values are stored so membership at $0.05$ and $0.25$ can be reported without rerunning the bootstrap. MCS bootstrap p-values use $\mathrm{mean}(T^\ast\ge T)$. Bootstrap indices are drawn once and reused through elimination. Ties are broken by original column index. Identical loss columns are ties. A nonzero constant pairwise differential is rejected as degenerate.

Block 3C contains two separately pre-specified Giacomini-White specifications. The market-state instrument vector is `[1, log(average realized variance)_{t-1}, average realized correlation_{t-1}]`. The measurement / stress vector is `[1, log(aggregate realized quarticity)_{t-1}, jump indicator_{t-1}]`. Bonferroni adjustment is used across these two GW specifications. The exact multivariate quarticity aggregation and jump-statistic definition are not yet frozen. Every conditioning variable must be observable at the forecast origin; target-day information and full-CONFIRM quantiles are forbidden in state construction.

Mincer-Zarnowitz calibration remains planned for Block 3C. The required family now includes standard MZ, a state-augmented MZ specification for conditional miscalibration, and covariance-aware MZ-GLS. Augmentation and GLS weighting are separate design choices. Any GLS or weighted specification must state and justify the conditional residual / proxy-error variance model; approximate weights must not be mislabeled as exact GLS. The final covariance calibration representation, portfolio projection set if used, weighting rule, and multiplicity treatment are not yet frozen and must be resolved before final `PREREGISTRATION.md`.

Giacomini-Rossi fluctuation analysis is a required Block 3C extension for formal time-local comparison of finalist forecast performance. Its window length or window fraction, admissible endpoints, direction, and reference critical values are not yet frozen. A moving sequence of ordinary pointwise DM tests is not an acceptable substitute for the formal fluctuation procedure. The cumulative loss-differential path remains a descriptive diagnostic alongside the formal time-local test.

State-dependent threshold forecast evaluation in the style of Odendahl-Rossi-Sekhposyan is an optional robustness extension, not a mandatory demo blocker. If later authorized, the state variable, timing, threshold family, search range, trimming or minimum state occupancy, and inference accounting for threshold search must be frozen before use. Post-hoc crisis thresholds with ordinary unadjusted p-values are forbidden.

## Data-source gate

Block 2 and Block 3 may be built and validated on synthetic known-truth data.

Block 4 must not begin until the empirical panel source is committed and verified for chronology, usable history length, cross-sectional dimension, security identity and universe construction, access and legal feasibility, and the ability to create the required forecasting and evaluation dates.

The long historical dataset is unresolved. This draft does not choose one.

---

Simulation is required to validate estimators, losses, and inference against known truth before relying on market data. Simulation never substitutes for the project's final empirical finding.
