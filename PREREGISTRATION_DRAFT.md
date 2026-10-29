# Preregistration draft

This document records protocol decisions that are frozen for implementation. It is a draft. It is not immutable and it is not the final preregistration.

Final `PREREGISTRATION.md` is created once, after the exact graph-neural model specification and the empirical dataset are frozen, and before any Block-4 empirical model fitting begins. Once that file is created it is never edited. Later departures must be recorded separately as deviations.

The graph-neural deep-learning slot is still unresolved. The long historical U.S.-equity panel is still unresolved. Any optional state-dependent threshold design is also not yet frozen. Therefore a final preregistration would be false.

Status on 2026-09-21. A later Binance open-data first-stage implemented XGBoost-DRD and LSTM-BEKK, ran VALIDATION and SCREEN under `docs/BINANCE_OPEN_DATA_PROTOCOL.md`, and left CONFIRM locked. The U.S.-equity DATA GATE remains closed. Final `PREREGISTRATION.md` remains absent. Sections below retain the draft commitments as recorded. They are not rewritten as if the Binance branch had been the original preregistration.

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

Later status. XGBoost-DRD was subsequently implemented in `covharness.models.xgboost_drd` and selected as `XGB08` on Binance VALIDATION. The two sentences above are the original draft text. They are not a current implementation claim.

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

Block 3C contains two separately pre-specified Giacomini-White specifications. They are run as two tests, not as one stacked six-instrument regression. The market-state instrument vector is `[1, log(average realized variance)_{t-1}, average realized correlation_{t-1}]`, constructed from origin-day realized covariance only and not standardized. The measurement / stress vector is `[1, log(RQ_agg)_{t-1}, BNS market jump indicator_{t-1}]`, constructed from origin-day synchronized intraday returns only and not standardized. $\mathrm{RQ}_{\mathrm{agg}}$ is the cross-sectional mean of per-asset realized quarticity $\mathrm{RQ}_i=(M/3)\sum_j r_{j,i}^4$. The jump state is the one-sided 1 percent Barndorff-Nielsen and Shephard (2006) adjusted-ratio indicator on the equal-weight intraday market return. Large negative standardized values indicate jumps. This detects a jump in the common-market portfolio. It is not an any-constituent-jumped indicator. A name-specific jump that is small in the equal-weight average can be missed, and a common jump can be flagged even if some constituents did not jump. Bonferroni adjustment is used across these two GW specifications. Family size is 2, family $\alpha=0.05$, and the per-test cutoff is $0.025$. Adjusted p-values are $\min(1,2p)$. Every conditioning variable must be observable at the forecast origin. Target-day information and full-CONFIRM quantiles are forbidden in state construction.

The primary Mincer-Zarnowitz representation is the Patton–Sheppard common-coefficient pooled-vech regression on unique entries $i\le j$, $S_{ij,t}=\alpha+\beta H_{ij,t}+e_{ij,t}$, with null $(\alpha,\beta)=(0,1)$. The 100-random-portfolio MZ design is not used. Inference uses daily scores that aggregate unique-entry contributions within each date, then Bartlett / Newey-West HAC on that daily sequence. Exact calibration returns Wald 0, p-value 1, and no numeric sandwich. An exact linear violation returns Wald $\infty$, p-value 0, and no numeric sandwich. Ordinary cases return the HAC sandwich. Headline standard MZ on the two finalists is a Bonferroni family of size 2. Diagonal-only and off-diagonal-only pooled coefficients are descriptive diagnostics and do not carry a headline familywise claim.

State-augmented MZ uses the same pooled representation with origin-day $z_t=[\log(\mathrm{mean}_i S_{ii,t}),\mathrm{mean}_{i<j}R_{ij}(S_t)]$ and common project-specific $\gamma$ coefficients. That pooled state restriction is not Patton–Sheppard. The null is $\alpha=0$, $\beta=1$, $\gamma_1=0$, $\gamma_2=0$. Augmented MZ on the same two finalists is a separate Bonferroni family of size 2. GW tests whether state predicts relative loss. Augmented MZ tests whether state predicts one model's calibration error.

Approximate Patton–Sheppard equation-21 weighting uses $s_{ij,t}=\sqrt{H_{ii,t}H_{jj,t}}$ and divides the dependent variable and every regressor by that scale. The label is `approximate_ps21`. It is not exact GLS. Forecast diagonals must be finite and strictly positive. Approximate PS21 is a robustness re-estimation of standard calibration. OLS versus WLS is not selected by whichever rejects.

Giacomini-Rossi fluctuation analysis is Proposition 1 of Giacomini–Rossi (2010) in the GW-method version. The frozen window fraction is $\mu=0.30$. The integer centered-window length is $m=2\lfloor 0.30 P/2\rfloor$. The formal test is two-sided only. At $\mu=0.30$ and $\alpha=0.05$ the Table I critical value is $k=3.012$. Rejection is $\max_t|F_t|>3.012$. Endpoints use centered windows only. The first and last $m/2$ positions are not test centers. The global long-run variance is the uncentered Bartlett / Newey-West estimator of the full confirmatory differential. It is not the demeaned Block 3A HAC estimator and it is not re-estimated inside local windows. A moving sequence of ordinary pointwise DM tests is not a substitute. The cumulative loss-differential path remains a descriptive diagnostic. The One-Time Reversal test is not implemented in Block 3C.

State-dependent threshold forecast evaluation in the style of Odendahl-Rossi-Sekhposyan is an optional robustness extension, not a mandatory demo blocker. If later authorized, the state variable, timing, threshold family, search range, trimming or minimum state occupancy, and inference accounting for threshold search must be frozen before use. Post-hoc crisis thresholds with ordinary unadjusted p-values are forbidden.

## Data-source gate

Block 2 and Block 3 may be built and validated on synthetic known-truth data.

Block 4 must not begin until the empirical panel source is committed and verified for chronology, usable history length, cross-sectional dimension, security identity and universe construction, access and legal feasibility, and the ability to create the required forecasting and evaluation dates.

The long historical dataset is unresolved. This draft does not choose one.

---

Simulation is required to validate estimators, losses, and inference against known truth before relying on market data. Simulation never substitutes for the project's final empirical finding.

---

## Post-SCREEN amendment register

This section is the written record for work proposed after the Binance selection period was seen. It does not rewrite the earlier draft commitments. It is not the final preregistration.

SCREEN results on targets 2023-11-30 through 2025-04-12 were seen on 2026-09-21. Every entry below was proposed after that date. Those SCREEN quantities remain selection data. They are not confirmatory rankings. Confirmatory claims rest on CONFIRM only. CONFIRM targets 2025-04-13 through 2026-08-25 remain locked.

A later task may be added only by appending a dated spec to this register and committing that spec before any run that produces empirical numbers. The report for that run records the spec-commit hash.

### Standing rules

The following rules bind every post-SCREEN task.

CONFIRM stays locked until gate G1 records an authorization decision. No task reads CONFIRM-dated measurements, forecasts, or losses. Data-only diagnostics stop at 2025-04-12. Callers must not pass `unlock_confirm=True`.

The 2026-09-21 VALIDATION and SCREEN artifacts are read-only. New outputs go under `results/post_screen/<task_id>/`. New candidates receive new IDs. Existing IDs are not reused for a different specification.

Specification precedes results. Each task writes its spec as a dated entry in this register and commits that entry before any run that produces empirical numbers. The corresponding report records that commit hash.

Every variant tried is reported, including failures. Stochastic methods use seeds $(0,1,2,3,4)$. Aggregation is the equal-weight covariance ensemble. Best-seed selection is forbidden.

Equal treatment is required. The rolling window remains $m=250$. Parameters are refit every 21 origins. Daily state updates, the same targets, and the same losses are reused. A search budget above 20 configurations for any family is recorded with the true total, and parity extensions are applied where the same edge problem exists.

One estimation-failure policy applies to all likelihood-estimated models. That policy is fixed in task C1 and reused by later tasks.

SCREEN-based outputs are labeled descriptive or selection, never confirmatory. Recursive state and statistical lags must not pack across the 2023-03-24 halt.

Reusable code goes in the existing `src/covharness` package that owns it. Scripts remain thin entry points.

### Planned entries

Status values start as planned except for task 0, which is this documentation record. Spec-commit and result-commit cells remain blank until the corresponding git hashes exist.

| Task ID | TODO items | Status | Spec commit | Result commit |
| --- | --- | --- | --- | --- |
| 0 | 0 | completed |  |  |
| A1 | 1-2 | completed | `80612248ae6f772262c4eadc8b48c6a12a9887c0` | `084e3915656be4d326ac242fc8bc709d86e335b3` |
| A1-addendum | A1 Wald check | completed | `0187bb1a5921bba53152f2610343cb0c8b4aaeb0` | `e0447f94ebbaa2e6b35799b8ae6660398e83f5ba` |
| A2 | 3 | completed | `75240e13873322df8d93f4dfb3bb4a76626a3bb4` | `80f03b0acc0231a660ae72e13ab1957133fef31f` |
| A3 | 4-5 | completed | `c30b27a5739d803458006853c2d8bbc80c80d32b` | `af3d14a2dca15cdf589586151bb8414e0ed13ee1` |
| A4 | 6 | completed | `d0d3f70547038bee845b2984c1ffdb2349f61650` | `71bdbc200c41224545116adfc0aef24ee0936690` |
| B1 | 14 | planned |  |  |
| B2 | 15 | completed | `8d0cbfb7184290f7db819e56c45e0da3548c831c` | `f9c6b58a79941bcc72392f6d95943d54b336a9dd` |
| C1 | 8 | planned |  |  |
| C2 | 7 | planned |  |  |
| C3 | 12 | planned |  |  |
| D1 | 13 | planned |  |  |
| D2 | 9 | planned |  |  |
| D3a | 10 | planned |  |  |
| D3b | 10 | planned |  |  |
| D4 | 11 | planned |  |  |
| E1 | 16 | planned |  |  |
| G1 |  | planned |  |  |
| F1 | 17 | planned |  |  |

G1 is the CONFIRM authorization gate. It has no numbered TODO item in the source list. Recording the gate in this table does not unlock CONFIRM.

### Task 0 artifact inventory

Inventory date 2026-10-28. Files were hashed on disk. Numpy arrays were not loaded. CONFIRM-dated slices were not read. The 2026-09-21 VALIDATION and SCREEN artifacts were not rewritten.

| Artifact | Path | Bytes | SHA-256 | Manifest match |
| --- | --- | --- | --- | --- |
| Production panel NPZ | `data/processed/binance_five_asset_panel.npz` | 359563 | `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a` | match with core YAML, VALIDATION manifest, and SCREEN manifest |
| Production panel sidecar JSON | `data/processed/binance_five_asset_panel.json` | 29810 | `38ed1d4843ea6138d7f87fc0405b668b1ee2fd7403d4cddfba0ca49160737394` | match with the 2026-09-21 panel record in `docs/PROJECT_STATE.md` |
| VALIDATION forecasts NPZ | `results/binance_validation_forecasts.npz` | 6693176 | `564afb691b138af0dba44fc1009f1e30c8f1887670e02254773957d9a77a9773` | match with VALIDATION manifest `forecasts_sha256` |
| VALIDATION QLIKE losses NPZ | `results/binance_validation_losses_qlike.npz` |  |  | absent. VALIDATION did not store a loss NPZ |
| VALIDATION Frobenius losses NPZ | `results/binance_validation_losses_frobenius.npz` |  |  | absent. VALIDATION did not store a loss NPZ |
| SCREEN forecasts NPZ | `results/binance_screen_forecasts.npz` | 907754 | `118f9eb4f5cfc8f2ffa1b1e73128358725aa179d91ebea8149c1f997a89e6c94` | match with SCREEN manifest `forecasts_sha256` |
| SCREEN QLIKE losses NPZ | `results/binance_screen_losses_qlike.npz` | 33884 | `3fa68921871277b52e60412bae9fb23f31e7b26e39bec446d1051f7f6e8af61d` | match with SCREEN manifest `losses_qlike_sha256` |
| SCREEN Frobenius losses NPZ | `results/binance_screen_losses_frobenius.npz` | 34731 | `63a4de10db499b5527e5f1c276c17460481d85a7c3699a7a6226a97819729713` | match with SCREEN manifest `losses_frobenius_sha256` |
| VALIDATION checkpoints | `results/binance_validation_checkpoints/` | 6883173 | directory of 167 NPZ files and 167 JSON files | no directory hash in the manifests |
| SCREEN checkpoints | `results/binance_screen_checkpoints/` | 873212 | directory of 13 NPZ files and 15 JSON files | no directory hash in the manifests |
| SCREEN HARQ-DRD checkpoint NPZ | `results/binance_screen_checkpoints/HARQ-DRD_HARQDRD01.npz` | 75428 | `d2232e6e3e6aa013ff705ff4519312c3dd72cdfcd0e34eed9d50b714eaa0cc14` | not a manifest field. Recorded here for later repair diagnostics |
| SCREEN HARQ-DRD checkpoint JSON | `results/binance_screen_checkpoints/HARQ-DRD_HARQDRD01.json` | 874 | `9810a5cd63c7f97a2956fca0d4a63891f18f20dc0704369ac262ad1bf6704e5f` | not a manifest field. Recorded here for later repair diagnostics |
| Binance 1-minute kline cache | `data/cache/open_data/binance/spot/monthly/klines/1m/` | 531001620 | directory of 290 ZIP files and 290 CHECKSUM files for BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, and ADAUSDT, months 2021-11 through 2026-08 | no cache hash in the manifests |

SCREEN DCC and DCC-NL checkpoints are JSON diagnostics only. They have no NPZ because those families failed at origin 2023-12-20. The SCREEN NPZ set is the nine complete representatives, with LSTM-BEKK stored as five seed files.

The SCREEN manifest file itself hashes to `dc54b52b245279f1b36a69d16ae1d3f88f8424dc6e1c4078124fe1755d12ea36`, matching the 2026-09-21 record. The tracked core YAML remains `8d63eacacf13a876651f9c4eb5399a8527d8458278974ff069dd0701dd5cbf07`. The VALIDATION selection file remains `f6d9830302b03afa2f3f45cd15576d70fbae4b96482af5c71d935822bc132f4b`.

### Task A1 spec. Persistence of SCREEN loss differentials and refit-cycle pattern

Spec date 2026-10-28. This entry is written before any A1 empirical number is computed. The 2026-09-21 SCREEN loss and forecast NPZ files remain read-only. CONFIRM is not read. The analysis is descriptive. It does not choose a Diebold–Mariano lag, a bootstrap block length, or a confirmatory default. Diebold–Mariano, SPA, and MCS are not run.

Sign convention. $d_t=L_{A,t}-L_{B,t}$. Then $d_t<0$ means A has lower loss on SCREEN target $t$.

Pairs, frozen now.

| Pair ID | A | B | Role |
| --- | --- | --- | --- |
| headline | HARQDRD01 | LSTM19 | SCREEN headline comparison. LSTM19 is the stored equal-weight five-seed covariance ensemble column |
| quarticity | HARQDRD01 | HARDRD01 | realized-quarticity ladder |
| nonlinear | HARDRD01 | XGB08 | nonlinear HAR-DRD ladder |
| qlike_mcs | HARQDRD01 | EWMA01 | primary reduced-QLIKE MCS pair at $\alpha=0.10$ |

Losses. Reduced multivariate QLIKE and squared Frobenius, each as the stored SCREEN loss panel. Both losses are scored for every pair. The two losses are never averaged.

Inputs. `results/binance_screen_losses_qlike.npz` and `results/binance_screen_losses_frobenius.npz`, whose 2026-09-21 hashes are recorded in Task 0. Column identities are the stored SCREEN model keys that correspond to those configuration IDs. Refit-phase labels come from the SCREEN schedule. The first SCREEN origin is a refit. Later refits occur every 21 forecast origins. The same $k$ sequence can be recovered from stored SCREEN `refit` flags. Those flags are not rewritten. Dates after 2025-04-12 are not read.

Persistence quantities, for each pair and each loss.

Sample ACF of $d_t$ at lags $1,\ldots,30$, using the existing $1/T$ autocovariance divisor already implemented in `covharness.inference.hac.sample_autocovariances`. $\hat\rho$ is the lag-1 ACF.

The AR(1) effective-sample approximation is $T_{\mathrm{eff,AR1}}=T(1-\hat\rho)/(1+\hat\rho)$. It is labeled as an AR(1) approximation. It is undefined when $|\hat\rho|\ge 1$ or when $1+\hat\rho=0$. It is not used to select a lag.

The existing $\kappa$-based $T_{\mathrm{eff}}$ is taken from `loss_differential_diagnostics` with its default Newey–West 1994 HAC lag. $\kappa=\hat\omega/\hat\gamma_0$ and $T_{\mathrm{eff}}=T/\kappa$ when those ratios are defined. $T_{\mathrm{eff}}$ is not clipped.

Bartlett long-run variance of $d_t$ is reported at four frozen lags. Newey–West 1994 automatic lag, which is $5$ at $T=500$. Twice that automatic lag, $10$. Four times that automatic lag, $20$. $\lfloor 1.3\sqrt{T}\rfloor=29$. Those lags are not chosen from the ACF. Reporting them does not adopt a new Diebold–Mariano default.

Refit cycle. Let $k$ be days since the last SCREEN refit, with $k=0$ on a refit origin and $k\in\{0,\ldots,20\}$ inside a complete 21-origin cycle. The SCREEN target count is $500=23\times 21+17$. The final cycle is therefore partial and occupies $k=0,\ldots,16$ only. Occupancy of $k=0,\ldots,16$ is $24$. Occupancy of $k=17,\ldots,20$ is $23$. For each pair and loss, report the mean of $d_t$ by $k$ together with the count. One descriptive Wald test asks whether those $k$-means are equal. The regression is $d_t$ on a full set of occupied $k$ dummies with no intercept. The null is that all dummy coefficients are equal, implemented as $20$ contrasts against $k=0$ when every $k$ is occupied, otherwise as occupied-count minus one. The sandwich uses the existing Bartlett daily-score HAC at lag $29$. The test is descriptive. It is not a confirmatory Diebold–Mariano test and does not select a lag.

Implementation. Helpers extend `src/covharness/inference/diagnostics.py`. The entry script is `scripts/post_screen_loss_dependence.py`. Outputs go under `results/post_screen/A1/` as JSON, CSV, and PNG. PNG files are ACF plots and mean-$d_t$-by-$k$ plots. SCREEN arrays are read only. New candidates are not created.

Tests, before the empirical run. Sample ACF on a synthetic AR(1) recovers $\rho$ within a fixed-seed Monte Carlo tolerance. The $T_{\mathrm{eff,AR1}}$ formula is checked on known $(\rho,T)$ values. Refit-phase labels are checked on a synthetic schedule that includes a partial final cycle of length $17$ after $23$ complete cycles of length $21$.

A1 empirical run date 2026-10-28. Spec commit `80612248ae6f772262c4eadc8b48c6a12a9887c0`. `tests/unit/test_inference_diagnostics_a1.py` and `tests/unit/test_inference.py` were run first. 27 passed, 1 existing overflow warning. The SCREEN loss NPZs matched the Task 0 hashes. Dates were 2023-11-30 through 2025-04-12. CONFIRM was not read. Diebold–Mariano, SPA, and MCS were not run. No lag or block length was adopted. Outputs are under `results/post_screen/A1/` and are SCREEN descriptive diagnostics.

### Task A1-addendum spec. Check the refit-phase Wald result

Spec date 2026-10-28. This entry is written before any addendum empirical number is computed. It does not rewrite A1 committed files or numbers. CONFIRM is not read. Diebold–Mariano is not run. No lag is adopted. The exercise remains descriptive SCREEN diagnostics for the same four pairs and two losses as A1.

Placebo. The A1 equal-mean Wald is invariant to permuting dummy names, so adding a constant to $k$ modulo $21$ is not a placebo. The placebo circularly shifts the phase series in time. For shift $r=0,1,\ldots,20$, replace the label path by `roll(k, r)` and recompute the A1 Wald at HAC lag $29$, leaving $d_t$ unchanged. $r=0$ is the true SCREEN refit calendar. Report the true statistic, the 20 shifted statistics, the rank of the true statistic among the 21 values, from largest to smallest, and how many of the 21 have $p<0.05$. A calendar-locked refit pattern should place the true statistic among the largest. Widespread small placebo $p$-values would indicate size distortion rather than a distinctive refit alignment.

One-degree-of-freedom contrasts. Candidate B1 HAC lags are $5$ and $29$. Both are reported. Neither is selected. Each contrast is OLS of $d_t$ on an intercept and one regressor, with the same Bartlett daily-score sandwich as A1, and a Wald test that the slope is zero.

Fresh-refit effect. The regressor is $1\{k=0\}$.

Linear staleness. The regressor is $k$ itself.

Influence sensitivity. Identify the ten SCREEN dates with the largest $|d_t|$. Report those dates, their $k$, and $d_t$. Recompute the A1 equal-mean Wald at lag $29$ after dropping those ten dates. Occupancy of $k$ is allowed to change. This is a sensitivity. It is not a cleaning rule and does not revise A1.

Implementation. Helpers extend `src/covharness/inference/diagnostics.py`. The entry script is `scripts/post_screen_refit_wald_check.py`. New outputs go under `results/post_screen/A1/` with the prefix `a1_addendum_`. Existing A1 JSON, CSV, and PNG files are not rewritten.

Tests, before the empirical run. A time-roll of synthetic phase labels changes the Wald when $d_t$ has a $k=0$ mean shift, while adding a constant to $k$ modulo the cadence does not. The one-degree-of-freedom slope Wald rejects a known intercept-plus-dummy mean shift and a known linear $k$ effect under a fixed seed. Dropping ten planted outliers changes the equal-mean Wald relative to the full sample.

A1-addendum empirical run date 2026-10-28. Spec commit `0187bb1a5921bba53152f2610343cb0c8b4aaeb0`. `tests/unit/test_inference_a1_addendum.py` and `tests/unit/test_inference_diagnostics_a1.py` were run first. 10 passed. Committed A1 `a1_summary.json` hash `58e4484211573c071435d38e79c77c9d35056f8999de667e57321f0bc5805cb9` was unchanged after the run. CONFIRM was not read. Diebold–Mariano was not run. No lag was adopted. New files use the prefix `a1_addendum_`.

Placebo. For every reduced-QLIKE pair, all 21 time-rolls reject equal phase means at $5$ percent. The true SCREEN alignment is not the largest statistic except that it is near the middle or the smallest. Headline QLIKE rank from largest is $5$. Quarticity $13$. Nonlinear $21$. HARQ–EWMA $14$. That pattern is not a distinctive refit-calendar lock. It is consistent with the 21-dummy Wald rejecting under this $d_t$ dependence for almost any roll of the same occupancy path. Headline squared Frobenius has rank $2$ and $0$ of $21$ rolls with $p<0.05$. Quarticity squared Frobenius has all $21$ rolls below $0.05$, with true rank $19$.

One-degree-of-freedom contrasts. Fresh-refit and linear-$k$ slopes are indistinguishable from zero at lags $5$ and $29$ for seven of eight pair-loss cells. The exception is HARQDRD01 minus EWMA01 reduced QLIKE. The $1\{k=0\}$ slope is $0.411$ with $p=0.0028$ at lag $5$ and $p=0.00084$ at lag $29$. Linear $k$ is not significant there. These contrasts do not select a lag.

Influence. Dropping the ten largest $|d_t|$ dates does not remove the reduced-QLIKE equal-mean rejections. Quarticity QLIKE Wald increases. Headline squared Frobenius moves from $p=0.081$ to $p=0.0026$. Remaining dates are concatenated for HAC. That construction is a sensitivity, not a cleaning rule.

### Task A2 spec. Pooled Mincer–Zarnowitz calibration on SCREEN

Spec date 2026-10-28. This entry is written before any A2 empirical number is computed. The exercise is descriptive. It does not rank models. It does not recalibrate forecasts. It does not change any model. SCREEN ratios from this task will not set any recalibration constant. Task C3, if later run, fits such a constant on past data only.

CONFIRM is not read. Production-panel realized covariance is copied only for SCREEN target dates 2023-11-30 through 2025-04-12. Dates after 2025-04-12 are not copied. The SCREEN forecast NPZ remains read-only.

Roster. The nine complete SCREEN columns are RW, EWMA, HAR-DRD, HARQ-DRD, LW-linear, LW-NL, Ridge-DRD, XGBoost-DRD, and LSTM-BEKK. LSTM19 is the stored five-seed equal-weight covariance ensemble. The five LSTM19 seeds are reported in an appendix table only. DCC01 and DCCNL01 are listed as having no forecasts. They are not scored. Models are reported in roster order, not by $p$-value.

Estimator. Reuse `mincer_zarnowitz_pooled_vech`. Do not write a new MZ estimator. OLS uses `weighting="none"`. Approximate PS21 uses `weighting="approximate_ps21"`. Unique entries are the $N(N+1)/2=15$ pairs $i\le j$ at $N=5$. Null $(\alpha,\beta)=(0,1)$. Daily-score Bartlett HAC uses the current automatic Newey–West 1994 lag, which is $5$ at $T=500$, and lag $29$ as a sensitivity. Neither lag is adopted as a new default. Diagonal-only and off-diagonal-only pooled coefficients are descriptive. Nine models are reported separately. This is not the confirmatory two-finalist Bonferroni family.

Whitened ratio. For each date, $r_t=\mathrm{tr}(H_t^{-1}S_t)/N$ is computed from the same Cholesky solves already used by reduced QLIKE. `inv(H)` is not formed. Report the mean of $r_t$, a HAC $95$ percent interval for that mean at the automatic lag and at lag $29$, and the share of days with $r_t>1$. Under correct calibration and an unbiased proxy the mean should be near $1$. Log and square-root transforms of $S$ or $H$ are not used, because they are biased under a noisy proxy.

HARQ-DRD fallbacks. Targets 2024-12-10 and 2025-03-03 remain in the sample and are flagged. They are not dropped.

Implementation. The whitened-ratio helper uses the existing Cholesky path in `covharness.losses`. The entry script is `scripts/post_screen_mz.py`. Outputs go under `results/post_screen/A2/`.

Tests, before the empirical run. Existing MZ tests must still pass. A new ratio test draws a synthetic proxy with $\mathrm{E}[S]=H$ and checks that the mean ratio is near $1$. Scaling $H$ by $2$ yields a mean ratio near $0.5$.

### Task A2 results. Pooled Mincer–Zarnowitz calibration on SCREEN

Result date 2026-10-28. Spec commit `75240e13873322df8d93f4dfb3bb4a76626a3bb4`. The estimator is the existing `mincer_zarnowitz_pooled_vech`. No new MZ estimator was written. Tests `tests/unit/test_inference_mz.py` and `tests/unit/test_losses.py` 48 passed before the SCREEN run. The whitened-ratio helper is `covharness.losses.qlike.whitened_trace_ratio`. Entry script `scripts/post_screen_mz.py`. Outputs `results/post_screen/A2/`. Production-panel realized covariance was copied only for SCREEN targets 2023-11-30 through 2025-04-12 and matched the stored SCREEN `proxy_rcov` cube exactly. Dates after 2025-04-12 were not copied. CONFIRM remained locked. Forecasts were not recalibrated. Models were not changed. Models were not ranked.

DCC01 and DCCNL01 have no forecasts.

Nine complete SCREEN columns, roster order, $T=500$, $N=5$, $15$ unique entries. Null $(\alpha,\beta)=(0,1)$. Automatic HAC lag $5$. Headline OLS and approximate PS21 Wald $p$-values use that lag. The mean whitened ratio $r_t=\mathrm{tr}(H_t^{-1}S_t)/N$ is reported with a HAC $95$ percent interval at the same lag.

| Model | Config | $\alpha$ OLS | $\beta$ OLS | Wald $p$ OLS | Wald $p$ PS21 | Mean $r$ | Mean $r$ interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW | RW01 | $0.000704$ | $0.435$ | $6.21\times 10^{-24}$ | $2.55\times 10^{-17}$ | $1.498$ | $[1.401,1.595]$ |
| EWMA | EWMA01 | $0.000303$ | $0.764$ | $6.98\times 10^{-4}$ | $1.84\times 10^{-3}$ | $1.148$ | $[1.041,1.255]$ |
| HAR-DRD | HARDRD01 | $0.000165$ | $0.943$ | $0.188$ | $0.340$ | $1.073$ | $[0.964,1.182]$ |
| HARQ-DRD | HARQDRD01 | $0.000142$ | $0.954$ | $0.243$ | $0.351$ | $1.062$ | $[0.962,1.161]$ |
| LW-linear | LWLIN01 | $0.000184$ | $1.379$ | $9.51\times 10^{-6}$ | $3.99\times 10^{-6}$ | $1.391$ | $[1.158,1.624]$ |
| LW-NL | LWNL01 | $0.000183$ | $1.339$ | $2.76\times 10^{-5}$ | $1.01\times 10^{-5}$ | $1.473$ | $[1.229,1.716]$ |
| Ridge-DRD | RIDGE01 | $0.000165$ | $0.943$ | $0.188$ | $0.340$ | $1.073$ | $[0.964,1.182]$ |
| XGBoost-DRD | XGB08 | $0.000338$ | $0.764$ | $6.94\times 10^{-3}$ | $0.209$ | $1.098$ | $[0.998,1.198]$ |
| LSTM-BEKK | LSTM19 | $0.000498$ | $0.764$ | $9.40\times 10^{-5}$ | $0.0280$ | $1.066$ | $[0.947,1.185]$ |

Ridge-DRD matches HAR-DRD exactly, as on the stored SCREEN losses.

HAR-DRD and HARQ-DRD do not reject $(\alpha,\beta)=(0,1)$ at $5$ percent under OLS or approximate PS21, at lag $5$ or lag $29$. Their mean-ratio intervals include $1$. HARQ-DRD fallback targets remain in the sample. On 2024-12-10, $r_t=2.353$. On 2025-03-03, $r_t=2.899$. Those two dates are flagged, not dropped.

LSTM19 is the stored five-seed covariance ensemble. OLS rejects $(0,1)$ with slope $0.764$. Approximate PS21 also rejects at lag $5$ ($p=0.028$). The mean whitened ratio is $1.066$ and its interval includes $1$. The OLS slope and the quadratic-form size therefore disagree. That is a calibration description. It is not a ranking. Lag $29$ does not reverse the OLS rejection. The five seeds are in `results/post_screen/A2/a2_appendix_lstm_seeds.csv`. All five reject OLS $(0,1)$. Seed 2 is the only seed whose mean-ratio interval excludes $1$.

XGBoost-DRD rejects OLS $(0,1)$ at lag $5$ ($p=0.00694$) and does not reject approximate PS21 ($p=0.209$). At lag $29$ the OLS $p$-value is $0.0587$. The mean-ratio interval includes $1$. RW, EWMA, LW-linear, and LW-NL reject both OLS and PS21 at both lags. Their mean-ratio intervals lie above $1$.

Diagonal-only and off-diagonal-only coefficients are in `results/post_screen/A2/a2_sensitivity.csv`. They are descriptive.

SCREEN ratios from this task do not set any recalibration constant. Task C3, if later run, fits such a constant on past data only. Log and square-root transforms were not used.

Report `results/post_screen/A2/REPORT.md`. The result commit is `80f03b0acc0231a660ae72e13ab1957133fef31f`. That commit left the result-commit cell blank because the hash did not exist until the commit was created. The cell above records that hash. The report commit does not replace it.

### Task A3 spec. Signature plot and return-sign asymmetry before CONFIRM

Spec date 2026-10-28. This entry is written before any A3 empirical number is computed. The exercise is descriptive. It documents the noise-versus-Epps trade-off behind the frozen five-minute proxy and describes return-sign asymmetry. The five-minute proxy is not changed because of the plot. The production panel file is not rewritten. CONFIRM is not read into an output row.

Dates. Inclusive HISTORY_A through SCREEN, 2021-11-09 through 2025-04-12. The halt 2023-03-24 is excluded as a statistical day. Its 23:59 close remains the calendar anchor for 2023-03-25, as in the frozen measurement specification. Expected length is 1250. CONFIRM dates 2025-04-13 through 2026-08-25 are not measured. The April 2025 monthly archive is opened because SCREEN occupies 2025-04-01 through 2025-04-12. Timestamps after 2025-04-12 23:59 UTC are skipped by the loader end bound and are not stored.

Signature grids. Intervals are 1, 5, 15, and 30 minutes. Each divides 1440. Endpoints are the minute-of-day closes `interval-1, 2*interval-1, ..., 1439`. The anchor on every grid is the prior calendar day's 23:59 one-minute close. No forward fill. A missing or nonpositive required close raises. Log returns are first differences of log closes on that grid, using one implementation shared with the frozen five-minute function. Realized covariance is the existing `realized_covariance`. Interval 5 returns the frozen `FIVE_MINUTE_END_MINUTES` tuple rather than a recomputed copy. \(M=1440/\mathrm{interval}\) and \(N=5\). Report \(M\) and \(M/N\).

Signature summaries. For each asset and interval, the mean over days of realized variance \(S_{ii}\). For each interval, the mean over days of the average strict-upper-triangle realized correlation. The plot shows both against the interval. Days are not dropped. No repair.

Asymmetry series. Daily UTC close-to-close log returns, and the 288 five-minute log returns. Primary tables are per asset. A pooled-across-assets table is secondary and does not replace the per-asset table. Within each series, the thresholds are the 95th and 99th percentiles of \(|r|\), using NumPy's linear quantile. Those two probabilities are fixed here and are not chosen after seeing counts. An exceedance is strict, \(r>c\) or \(r<-c\). Mean size on a side is the mean of \(|r|\) among exceedances on that side. A two-sided binomial test compares up and down counts under null probability \(1/2\). No multiplicity adjustment is applied. The tests are descriptive. Skewness is the bias-corrected Fisher–Pearson moment, `scipy.stats.skew` with `bias=False`.

Semivariance. On the five-minute returns, for each asset and day, \(\mathrm{RS}^{+}\) is the sum of squares of strictly positive returns and \(\mathrm{RS}^{-}\) is the sum of squares of strictly negative returns. Zeros contribute to neither. Their sum equals realized variance. Report the mean of each and a Bartlett HAC 95 percent interval for the mean of \(\mathrm{RS}^{+}-\mathrm{RS}^{-}\), using `hac_long_run_variance` at its automatic Newey–West 1994 lag. The lag is not chosen from this sample's autocorrelation. The normal critical value is 1.96.

Implementation. Add the interval parameter on the existing path in `covharness.data.binance_calendar` and `covharness.realized.binance_panel`. Semivariance identity lives next to `realized_covariance`. The entry script is `scripts/post_screen_measurement.py`. Outputs go under `results/post_screen/A3/`.

Tests, before the empirical run. Interval 5 endpoints equal the frozen tuple. Return counts per day equal \(1440/\mathrm{interval}\). Semivariances sum to realized variance. Five-minute realized covariance on these dates is bitwise equal to the production panel. The frozen proxy and the production panel file are not modified.

### Task A3 results. Signature plot and return-sign asymmetry before CONFIRM

Result date 2026-10-28. Spec commit `c30b27a5739d803458006853c2d8bbc80c80d32b`. The interval parameter was added on the existing measurement path. `grid_log_returns` is the only log-difference implementation. The five-minute wrapper calls it. `tests/unit/test_post_screen_a3.py` and `tests/unit/test_binance_measurement.py`, excluding the production-panel parity test, 19 passed. The parity test then passed in 20.22 seconds. The entry script `scripts/post_screen_measurement.py` exited 0. Five-minute realized covariance on 2021-11-09 through 2025-04-12 is bitwise equal to the production panel. \(T=1250\). The halt is not a row. CONFIRM rows copied are 0. The production panel file was not rewritten. The frozen five-minute proxy was not changed.

Mean pairwise realized correlation rises with the interval. It is 0.654 at 1 minute, 0.708 at 5 minutes, 0.722 at 15 minutes, and 0.725 at 30 minutes. \(M/N\) is 288, 57.6, 19.2, and 9.6. Mean realized variance changes little across these grids. That is the Epps pattern on this panel. It does not select a new proxy.

Daily close-to-close skewness is negative for BTC, ETH, BNB, and LTC, and positive for ADA. Per-asset binomial tests of daily exceedance counts do not reject equal up and down counts at 5 percent at either frozen percentile. The pooled daily 95th-percentile test has \(p=0.054\). Five-minute skewness is negative for every name. At the 95th percentile of absolute five-minute returns, down counts exceed up counts, and the binomial test rejects at 5 percent for ETH, BNB, LTC, and ADA. BTC does not reject. At the 99th percentile only BNB rejects. These tests are descriptive. No multiplicity adjustment was applied.

Five-minute mean \(\mathrm{RS}^{+}-\mathrm{RS}^{-}\) is negative for every asset. The automatic HAC lag is 7. The 95 percent interval excludes 0 for BNB and LTC and includes 0 for BTC, ETH, and ADA. Semivariances partition the squared returns exactly at the element level.

Outputs are under `results/post_screen/A3/`. Result commit `af3d14a2dca15cdf589586151bb8414e0ed13ee1`. Report `results/post_screen/A3/REPORT.md`.

### Task A4 spec. Leverage asymmetry in SCREEN variance and in LSTM-BEKK

Spec date 2026-10-28. This entry is written before any A4 empirical number is computed. The exercise is descriptive. It does not change a model. It does not assume that crypto has the equity leverage sign. CONFIRM is not read.

Sample. SCREEN target dates 2023-11-30 through 2025-04-12. Predictors may use HISTORY_B rows, which begin 2023-03-25. They may not use a date after the target's origin, a VALIDATION date, or the halt. The 22-day HAR lag from the first SCREEN target stays inside HISTORY_B. Production-panel rows after 2025-04-12 are not copied.

Equation, for each asset \(i\).

\[
\log \mathrm{RV}_{i,t+1} = a + b_d \log \mathrm{RV}_{i,t} + b_w \log \mathrm{RV}_{i,t}^{(w)} + b_m \log \mathrm{RV}_{i,t}^{(m)} + g_{\mathrm{pos}} r_{i,t}^{+} + g_{\mathrm{neg}} r_{i,t}^{-} + e_{i,t+1}
\]

HAR blocks match HAR-DRD. At response index \(k\), the daily lag is observation \(k-1\), the weekly block is the mean of the four variances \([k-5, k-1)\), and the monthly block is the mean of the seventeen variances \([k-22, k-5)\). The blocks do not overlap. Averages are taken on the variance scale, then logged. \(r^{+}=\max(r,0)\) and \(r^{-}=\min(r,0)\) use the origin-day UTC log return. A nonpositive variance or forecast diagonal raises. No repair and no epsilon floor.

Three dependent variables, same regressors. Data uses \(\log \mathrm{RV}_{i,t+1}\). LSTM19 uses \(\log H_{ii,t+1\mid t}\) from the stored five-seed ensemble. The control uses \(\log H_{ii,t+1\mid t}\) from HARQDRD01. HARQ-DRD fallback targets 2024-12-10 and 2025-03-03 stay in every regression and are flagged.

Leverage, as a label rather than an assumption, means \(|g_{\mathrm{neg}}| > g_{\mathrm{pos}}\) and \(g_{\mathrm{neg}}<0\). The reported gap is \(g_{\mathrm{pos}}+g_{\mathrm{neg}}\), with a HAC 95 percent interval. Primary fits are per asset. A pooled fit with common coefficients is secondary. Its daily score sums the asset scores before HAC, so the day remains the sampling unit.

Inference. OLS. Bartlett HAC of the daily score reuses `hac_long_run_covariance` from the Mincer–Zarnowitz module, with the same \((X'X)^{-1}(T\hat\Omega)(X'X)^{-1}\) scaling. The lag is the automatic Newey–West 1994 lag. It is not chosen from this sample. The critical value is 1.96. A rank-deficient design raises. No ridge.

Implementation. The regression helper lives in `covharness.inference`. The entry script is `scripts/post_screen_leverage.py`. Outputs go under `results/post_screen/A4/`.

Tests, before the empirical run. On a synthetic series with a planted negative \(g_{\mathrm{neg}}\) larger in magnitude than \(g_{\mathrm{pos}}\), the fit recovers that sign pattern and a negative gap. A planted positive \(g_{\mathrm{neg}}\) is not flipped.

### Task A4 results. Leverage asymmetry in SCREEN variance and in LSTM-BEKK

Result date 2026-10-28. Spec commit `d0d3f70547038bee845b2984c1ffdb2349f61650`. Tests `tests/unit/test_post_screen_a4.py` 3 passed before the SCREEN fit. Entry script `scripts/post_screen_leverage.py` exited 0. The lag sample is 2023-03-25 through 2025-04-12. SCREEN responses are the 500 targets beginning 2023-11-30. The halt is absent. CONFIRM rows copied are 0. Forecast diagonals were strictly positive. Models were not changed. HARQ fallback dates remain in the sample.

The leverage label is the spec boolean \(g_{\mathrm{neg}}<0\) and \(|g_{\mathrm{neg}}|>g_{\mathrm{pos}}\). It is not a finding by itself when \(g_{\mathrm{pos}}\) is negative. HAC lag is 5. Intervals are 95 percent.

Data, per asset. Every name has \(g_{\mathrm{neg}}<0\) and \(|g_{\mathrm{neg}}|>g_{\mathrm{pos}}\). The gap interval excludes 0 for BTC, ETH, BNB, and ADA. LTC's interval includes 0. The pooled data gap is \(-4.24\) with interval \([-6.87,-1.61]\).

LSTM19. Point estimates have \(g_{\mathrm{neg}}<0\) for every name. The magnitude condition fails for ETH, where \(g_{\mathrm{pos}}=3.77\) and \(g_{\mathrm{neg}}=-3.39\). Every per-asset gap interval includes 0. The pooled gap is \(-0.26\) with interval \([-1.00,0.48]\). The ensemble's signed-return slopes are much smaller than the data slopes. That is a description of the stored forecasts. It is not a refit.

HARQDRD01. This control was expected to show little residual sign slope beyond the variance lags. It does not do so uniformly. BTC's gap interval excludes 0, at \([-2.98,-0.49]\). ETH, BNB, and ADA intervals include 0. LTC has \(g_{\mathrm{neg}}>0\). The pooled HARQ \(g_{\mathrm{pos}}\) is negative, so the mechanical label is not a large downside coefficient. The control is reported as estimated.

Outputs are under `results/post_screen/A4/`. Result commit `71bdbc200c41224545116adfc0aef24ee0936690`. Report `results/post_screen/A4/REPORT.md`.

### Task B2 spec. SCREEN SPA and MCS at longer bootstrap blocks

Spec date 2026-10-28. This entry is written before any B2 empirical number is computed. The exercise is a sensitivity. It does not replace the frozen 2026-09-21 SPA or MCS files. It does not change the finalist rule. The finalist rule uses sub-block ranks, not MCS.

Universe. The nine complete SCREEN columns in `complete_labels`. DCC and DCC-NL are excluded because they have no forecasts. Both stored losses are used, reduced QLIKE and squared Frobenius, separately. They are not averaged.

Procedures. SPA benchmark is HAR-DRD. Consistent recentering is primary. Lower and upper recenterings are reported. RIDGE01 keeps the existing exact-benchmark-tie exclusion from the SPA statistic. MCS primary is \((T_R, e_R)\). Companion is \((T_{\max}, e_{\max})\). The pairs are not crossed. \(B=5000\). Seed \(20260913\). Mean block lengths are 7, 14, and 30. At length 7 the saved p-values must be reproduced exactly, because 7 is the cube-root default at \(T=500\).

Implementation. Pass `block_length` through the existing `screen_spa` and `screen_mcs` wrappers into `superior_predictive_ability` and `model_confidence_set`. No new bootstrap and no new statistic. The entry script is `scripts/post_screen_block_sensitivity.py`. Outputs go under `results/post_screen/B2/`. The 2026-09-21 JSON files are read and not rewritten.

CONFIRM block length. It is not chosen here. It is not chosen by which of 7, 14, or 30 gives smaller p-values. The rule is decided at G1 and tied to B1's frozen bandwidth. G1 is not run in this task. B1 is not run in this task.

Tests, before the empirical run. A wrapper call with an explicit block length records that length. The length-7 SCREEN run is compared with the saved SPA and MCS JSON after the script starts, and a mismatch stops the task.

### Task B2 results. SCREEN SPA and MCS at longer bootstrap blocks

Result date 2026-10-28. Spec commit `8d0cbfb7184290f7db819e56c45e0da3548c831c`. `tests/unit/test_post_screen_b2.py` passed before the SCREEN rerun. Entry script `scripts/post_screen_block_sensitivity.py` exited 0 in 6.5 seconds. Length 7 reproduced the saved SPA and MCS p-values exactly on both losses. RIDGE01 remained an exact benchmark tie and was excluded from the SPA statistic. The frozen JSON and sub-block rank files were not rewritten. Their hashes match the 2026-09-21 record.

QLIKE consistent SPA p-values are 0.0016, 0.0030, and 0.0022 at lengths 7, 14, and 30. All three recenterings reject at 0.05 at every length. Frobenius consistent p-values are 0.8792, 0.8574, and 0.8688. None reject. Longer blocks do not change those conclusions.

QLIKE range MCS at \(\alpha=0.10\) contains EWMA and HARQ-DRD at every length. The max companion also contains HAR-DRD, Ridge-DRD, and XGBoost-DRD at every length. LSTM-BEKK is outside both. Frobenius range membership changes in one place. RW is inside at length 7, with p-value 0.1084, and outside at lengths 14 and 30. The other Frobenius range members do not change. The max companion's 0.10 membership does not change with block length.

The CONFIRM block length is not chosen. It remains a G1 decision tied to B1. The finalist rule is unchanged.

Outputs are under `results/post_screen/B2/`. Result commit `f9c6b58a79941bcc72392f6d95943d54b336a9dd`. Report `results/post_screen/B2/REPORT.md`.
