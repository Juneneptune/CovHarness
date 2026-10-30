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

One estimation-failure policy applies to all likelihood-estimated models. The policy is carry-forward of the last admissible parameter vector, as recorded on 2026-10-29. Task C1 implements that policy. Later tasks reuse it.

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
| B1 | 14 | completed | `3ccffdbd84398a82879cd16d7da8d41c6ba8c5b6` | `9169b92ad442ffa95d4f7a317aff55484d520463` |
| B2 | 15 | completed | `8d0cbfb7184290f7db819e56c45e0da3548c831c` | `f9c6b58a79941bcc72392f6d95943d54b336a9dd` |
| C1 | 8 | completed | `ed7d2a4761fa5702e7c18e55c48eb2d7cc0c50b7` | `7555cc91925967b531d6e3bba33c8e1bf47e73a2` |
| C2 | 7 | planned |  |  |
| C3 | 12 | planned |  |  |
| D1 | 13 | completed | `e4f8328eacdf6088eb8765bae026f17e97c1464b` | `afb951a005b6ad22e1b12097eb579ef01aef852f` |
| D2 | 9 | planned |  |  |
| D3a | 10 | completed | `d45a5bce027b50be38bc5aaf722ddbd183392d10` | `b3cc780d971ec76844b6bf1f77a899a6632616e1` |
| D3b | 10 | deferred |  |  |
| D4 | 11 | deferred |  |  |
| E1 | 16 | planned |  |  |
| G1 |  | planned |  |  |
| F1 | 17 | planned |  |  |
| R1 | amendment R-2026-10-29 | completed | `a7f8e4e21cbfc8145261756e9b030126948ebc86` | `a7f8e4e21cbfc8145261756e9b030126948ebc86` |
| R2 | amendment R-2026-10-30 | completed | `9d50128d0a01a80396929381314aac9fc4b01125` | `9d50128d0a01a80396929381314aac9fc4b01125` |
| B1b | P5 size-adjusted cutoffs | specified |  |  |
| C4 | A4 follow-up | planned |  |  |

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

### Task D3a spec. Shifted-day panels for training augmentation

Spec date 2026-10-28. This entry is written before any D3a phase panel is built. The panels are training material. Evaluation targets remain UTC-day realized covariance. No model is trained. CONFIRM-period phase days are not built.

Label convention. The existing phase 00 is the production UTC day. Row \(d\) is the interval \([00{:}00\) on \(d,\ 24{:}00\) on \(d)\), anchored on the prior 23:59 close. That is the object the bitwise test names. The seven shifted phases use the stated window. For \(h\in\{3,6,9,12,15,18,21\}\), the phase-\(h\) day labeled \(d\) is \([h{:}00\) on \(d-1,\ h{:}00\) on \(d)\).

Measurement. Five-minute grid, 288 returns, anchored on the previous one-minute close at the phase boundary, no forward fill. Realized covariance, realized quarticity, and the log return are taken over that same window. The log-return and realized-covariance functions are the ones already used for the production day. A missing close raises. No repair.

Dates. Labels run from HISTORY_A through SCREEN, 2021-11-09 through 2025-04-12. A phase day is dropped when its half-open window intersects 2023-03-24. Under the label convention, phase 00 keeps 2023-03-25, because the production day does. Every shifted phase drops the label 2023-03-25, because that window includes hours on the halt date. Dropped labels are stored. Successive kept rows are not treated as adjacent for lags when a label between them was dropped. There is no packing across the halt.

Eligibility. A phase day labeled \(d\) ends at or before 24:00 on \(d\), so it is first usable at origin \(d\) and not at origin \(d-1\). The build checks that no kept row ends after 24:00 on its label. CONFIRM labels are not built. The April 2025 archive is read only through 2025-04-12 23:59.

Overlap. Phase offsets three hours apart share 21 hours. Same-phase successive days share only the boundary. The samples are correlated. They do not multiply the effective sample size. No effective-sample-size number is computed in this task.

Implementation. The phase window is added beside the interval parameter in `covharness.realized.binance_panel`. The entry script is `scripts/post_screen_phase_panels.py`. Outputs go under `results/post_screen/D3a/`.

Tests, before the panels are accepted. Phase 00 matches the production realized-covariance panel bitwise on the pre-CONFIRM dates. The halt exclusion matches the rule above. No kept phase day ends after its label date.

### Task D3a results. Shifted-day panels through SCREEN

The build kept the registered label convention. Phase 00 has 1250 rows and matches the production realized-covariance panel bitwise. Each shifted phase has 1249 rows. The only dropped label is 2023-03-25. Phase 00 keeps that date. No phase contains 2023-03-24 or a date on or after 2025-04-13. Every kept row ends by 24:00 on its label and is not usable at the previous origin. The halt leaves one lag break in every phase, plus the first row, which has no predecessor. Phase offsets three hours apart share 21 hours. The samples are correlated and do not multiply the effective sample size. No model was trained. The production panel hash is unchanged.

Outputs are under `results/post_screen/D3a/`. The phase arrays are local `results/**/*.npz` files and are not tracked. Their SHA-256 values are in `d3a_summary.json`. Result commit `b3cc780d971ec76844b6bf1f77a899a6632616e1`. Report `results/post_screen/D3a/REPORT.md`.

### Decisions recorded on 2026-10-29

These choices are fixed before tasks B1, C1, C3, and D2 run. No empirical run was performed for this entry. CONFIRM stays locked. The 2026-09-21 artifacts are unchanged. B1, C1, C3, and D2 remain planned until each task writes its own spec.

B1 design autocorrelation. The design persistence is an AR(1) coefficient of 0.6. The value 0.3 is closed. The stored synthetic calibration at \(T=500\) has automatic-lag rejection 0.0690 at 0.3 and 0.1104 at 0.6, so the bandwidth study is aimed at the stronger of those two documented cells. B1 has not been run. Gate G1 still ties the CONFIRM block length to the bandwidth B1 freezes under this coefficient. That length is not chosen from SCREEN p-values.

C1 estimation failure. A failed or inadmissible likelihood fit carries forward the last admissible parameter vector for that model, records the failure, and forms the forecast from those parameters and the current admissible data. An interior constraint map is closed. If no admissible vector exists yet, the origin is a recorded failure and no substitute forecast is created. The policy applies from C1 onward. The stored DCC and DCC-NL selection failure remains the report that was already published.

Robustness proxy. The primary proxy remains the unscaled five-minute realized covariance. The robustness proxy is the unscaled 15-minute realized covariance from the same measurement path, at interval 15, with the same anchor rule and no forward fill. Task A3 is the descriptive basis. On the pre-holdout sample the mean pairwise correlation is 0.708 at five minutes and 0.722 at fifteen minutes, and \(M/N\) at fifteen minutes is 19.2. No model forecast has been scored on the 15-minute proxy. Fifteen-minute results do not select representatives and do not replace primary confirmatory claims. CONFIRM-dated 15-minute rows are not built before G1.

SHAR-DRD. The econometric side gains SHAR-DRD before D2 is fully specified. It is HAR-DRD in levels. The correlation equation is the HAR-DRD equation. The daily variance lag is replaced by the two same-day semivariances \(\mathrm{RS}^{+}\) and \(\mathrm{RS}^{-}\) from the primary five-minute returns. Weekly and monthly variance lags remain the HAR-DRD averages of realized variance. The four variance slopes are shared. Intercepts stay asset-specific. The estimator, window, cadence, pair order, and estimation-window-mean fallback remain those of HAR-DRD. A zero return is in neither semivariance. The model receives a new ID when it is implemented. It is not implemented in this entry.

D2 variant V10. The information-rich LSTM input includes those same origin-dated per-asset semivariances. Target-day semivariances are excluded. D2 does not run until the rest of the V10 input list is written in its own spec. The faithful daily-return LSTM-BEKK is unchanged.

C3 recalibration. Two versions are pre-declared. The diagnostic basis is the A2 pooled Mincer–Zarnowitz slope, not a model ranking. LSTM19 has OLS slope \(0.764\). A scalar multiplier rescales that slope and leaves the intercept unchanged. Version 2 is declared beside it. Both are fit on past forecasts only, by primary reduced QLIKE, with the adjusted date excluded from the fit. Version 1 is \(H^{*}=cH\) with \(c>0\). Version 2 is

\[
H^{*}=c\left[\lambda H+(1-\lambda)\bar H\right]
\]

with \(c>0\) and \(\lambda\in[0,1]\). \(\bar H\) is the equal-weight mean of that model's own one-day-ahead forecasts on the preceding 250 origins whose targets fall on or before the current origin. An origin with fewer than 250 such forecasts has no version-2 forecast, and the gap is reported. Both versions are reported. The version is not chosen after seeing which one scores better. No recalibration was fit in this entry.

### Amendment R-2026-10-29

Amendment date 2026-10-29. This entry is written before tasks C1, B1, D1, and C4 run. Every decision below rests on pre-holdout diagnostics from A1, A2, A3, and A4. None rests on a model ranking. CONFIRM stays locked. The 15-minute panel and the semivariance model are not built in this amendment. Task T1 remains the unmerged editorial branch `docs/editorial-pass` and is outside this register.

Pairwise test persistence. Task B1 uses \(\rho_{\star}=0.6\). The largest lag-1 autocorrelation among the eight pre-specified A1 gaps is 0.513, on HARDRD01 versus XGB08 under squared Frobenius. The value 0.6 is the smallest calibration cell at or above that estimate. At \(T=500\) the current automatic-lag Diebold–Mariano test rejects at 0.069 when \(\rho=0.3\) and at 0.110 when \(\rho=0.6\). Cells 0.8 and 0.9 are reported and do not set the requirement.

Estimation-failure policy. Carry-forward of the last accepted parameters is the primary rule for every likelihood-estimated model, including DCC, DCC-NL, BEKK, and HEAVY. An interior softmax map with \(\varepsilon=10^{-6}\) runs as a labeled robustness variant. Primary identifiers are eligible for a later headline. Robustness identifiers are not. DCC01 and DCCNL01 remain the failed selection columns.

Secondary evaluation proxy. Fifteen-minute realized covariance, built by the frozen measurement path with the interval parameter from A3, is a CONFIRM robustness proxy. Five-minute realized covariance stays primary. A3 reports mean pairwise correlation 0.708 at 5 minutes, 0.722 at 15 minutes, and 0.725 at 30 minutes, with realized variance nearly flat. Ranking and inference are reported separately for each proxy and are never pooled. A conclusion is described as proxy-robust only when it holds under both.

Sign asymmetry. The basis is A4. SHAR-DRD, identifier SHARDRD01, enters the econometric eligible pool under task C4. The D2 variant V10 replaces log realized variance in the LSTM input with \(\log\mathrm{RS}^{+}\) and \(\log\mathrm{RS}^{-}\) from the same five-minute returns, so both sides receive the same semivariance information.

Recalibration. Task C3 reports two out-of-sample versions for every model. Recalibration R1 is the scalar map. Recalibration R2 shrinks the forecast toward the trailing proxy mean. Both are fit on past out-of-sample pairs only. The basis is A2. LSTM-BEKK has mean whitened ratio 1.066 and Mincer–Zarnowitz slope 0.764. The headline forecast stays raw unless this register adopts one version for every model before G1.

Completed rows already in the planned-entry table keep the hashes from `results/post_screen/OVERNIGHT_SUMMARY.md` and the A1 reports. A1 spec `80612248ae6f772262c4eadc8b48c6a12a9887c0`, result `084e3915656be4d326ac242fc8bc709d86e335b3`. A1-addendum spec `0187bb1a5921bba53152f2610343cb0c8b4aaeb0`, result `e0447f94ebbaa2e6b35799b8ae6660398e83f5ba`. A2 spec `75240e13873322df8d93f4dfb3bb4a76626a3bb4`, result `80f03b0acc0231a660ae72e13ab1957133fef31f`. A3 spec `c30b27a5739d803458006853c2d8bbc80c80d32b`, result `af3d14a2dca15cdf589586151bb8414e0ed13ee1`. A4 spec `d0d3f70547038bee845b2984c1ffdb2349f61650`, result `71bdbc200c41224545116adfc0aef24ee0936690`. B2 spec `8d0cbfb7184290f7db819e56c45e0da3548c831c`, result `f9c6b58a79941bcc72392f6d95943d54b336a9dd`. D3a spec `d45a5bce027b50be38bc5aaf722ddbd183392d10`, result `b3cc780d971ec76844b6bf1f77a899a6632616e1`. Commit `a7f8e4e21cbfc8145261756e9b030126948ebc86` records this amendment. Report `results/post_screen/R1/REPORT.md`.

### Task C1 spec. Estimation-failure policy for DCC and DCC-NL

Spec date 2026-10-29. This entry is written before any C1 fit. It follows amendment R-2026-10-29. DCC01 and DCCNL01 stay failed. No SPA, MCS, or finalist selection is run. CONFIRM stays locked. The 2026-09-21 artifacts are not rewritten.

Primary policy CF. At a refit where any stage fails \(a+b<1\), the model keeps the last accepted parameters for every stage and keeps updating states on the current data. A failure at a period's first refit, with nothing to carry, is a model failure. Each carry-forward is logged with origin, asset, and stage. Forecasts are not repaired.

Robustness policy IM. Parameters are estimated through an unconstrained softmax-type map so \(a+b<1\) holds by construction. The recorded gap is \(\varepsilon=10^{-6}\). A fit with \(a+b>1-\varepsilon\) or \(a+b>0.999\) is counted and reported as near-boundary. IM identifiers are not headline-eligible.

Identifiers. DCC02 and DCCNL02 are CF and are eligible at a later G1. DCC03 and DCCNL03 are IM and are not eligible for the headline. Later likelihood models, including BEKK and HEAVY, inherit CF as primary and IM as robustness.

Implementation. The shared map and the carry-forward log live where C2 and D4 can import them. `dcc.py` keeps the existing reject path as the default, so DCC01 mathematics are unchanged. The entry script is `scripts/post_screen_dcc_failure_policy.py`. VALIDATION runs before SCREEN. Outputs go under `results/post_screen/C1/`. Both losses are scored. SCREEN output is labeled selection.

Tests, before the empirical run. A synthetic GARCH path that drives QMLE to the boundary triggers CF. Carried parameters stay constant while states move. A first-refit failure still fails under CF. IM never returns \(a+b\ge 1\) and counts near-boundary fits. No forecast is repaired.

Commit `7555cc91925967b531d6e3bba33c8e1bf47e73a2` records the execution. Report `results/post_screen/C1/REPORT.md`.

### Task B1 spec. Confirmatory pairwise procedure

Spec date 2026-10-29. This entry is written before any B1 simulation. The persistence requirement is the frozen value \(\rho_\star=0.6\) from amendment R-2026-10-29. Cells 0.8 and 0.9 are reported and do not set the requirement. The choice is made on simulated data. It is not made by which procedure favours a model. CONFIRM stays locked. Default HAC and DM behaviour is unchanged unless this entry's later freeze says so. New behaviour is reached by a new keyword.

Candidates. P1 is Bartlett, Newey-West 1994 automatic lag, and N(0,1) critical values. P2 is Bartlett, four times that automatic lag, and N(0,1). P3 is Bartlett, lag \(\lfloor 1.3\sqrt{T}\rfloor\), and Kiefer-Vogelsang (2005) fixed-\(b\) critical values, the Lazarus, Lewis, Stock, and Watson (2018) recommendation. P4 is Bartlett with the Andrews (1991) AR(1) plug-in lag and the same fixed-\(b\) critical values. P5 is the equal-weighted cosine estimator with \(\nu=\lfloor 0.4 T^{2/3}\rfloor\) and \(t_\nu\) critical values, equation (10) of Lazarus, Lewis, Stock, and Watson (2018). P6 is the recentered stationary-bootstrap mean test. Its block length equals the P3 lag. Its inner bootstrap count is the existing pairwise constant 5000.

Fixed-\(b\). The bandwidth fraction is \(b=(L+1)/T\). The critical-value function is linear interpolation of Kiefer and Vogelsang (2005) Table I, which that paper states may be interpolated between the tabulated \(b\) grid. The two-sided 5 percent column is the tabulated 97.5 percent right-tail value. The \(b=1\) entries are the analytical Abadir and Paruolo values, including 3.764 at 95 percent and 4.771 at 97.5 percent. No coefficient is invented.

Designs. \(T\in\{500,250\}\). \(\rho\in\{0,0.3,0.6,0.8,0.9\}\). D1 is a Gaussian AR(1) null. D2 is a Student-\(t(5)\) AR(1) null. D3 resamples the demeaned SCREEN HARQDRD01 minus LSTM19 QLIKE gap by the stationary bootstrap at block lengths 7 and 29. D3 uses SCREEN only. Replications are 5000. The seed is the existing calibration seed 20260916 in `size.py`, split by a fixed seed sequence so cells are independent. Monte Carlo standard errors are reported. Power, for D1 and D2, uses a mean shift of 0.2 long-run standard deviations.

Selection rule. A procedure qualifies when its two-sided 5 percent rejection rate lies in \([0.03,0.075]\) at \(T=500\) for every \(\rho\le 0.6\) under D1 and D2, and under D3. Among qualifiers, the highest average power is selected. Ties go to the simpler procedure, in the order P1 through P6. If none qualifies, the smallest maximum absolute distortion from 0.05 is selected and recorded as a limitation.

The entry script is `scripts/dm_procedure_calibration.py`. Outputs go under `results/post_screen/B1/`. The frozen choice is written here before the SCREEN demonstration on the four A1 pairs. That demonstration is descriptive. It does not rerun SPA, MCS, or finalist selection.

### Task B1 freeze. Confirmatory pairwise procedure

Freeze date 2026-10-29. This paragraph is written after the calibration and before the SCREEN demonstration. No procedure qualified. Under D1 and D2 at \(T=500\) and \(\rho\le 0.6\), several candidates were inside \([0.03,0.075]\), but every candidate exceeded 0.075 on at least one D3 resample of the SCREEN HARQ-DRD minus LSTM-BEKK QLIKE gap. The selection rule therefore takes the smallest maximum absolute distortion from 0.05. That procedure is P5, the equal-weighted cosine estimator with \(\nu=\lfloor 0.4 T^{2/3}\rfloor\) and \(t_\nu\) critical values. Its maximum distortion on the required cells is 0.0334. This is a limitation. CONFIRM will call P5 through the new keyword. The default automatic-lag \(N(0,1)\) test is unchanged. P6 is not adopted.

Commit `2aae89c62d7c5e5113e4c34e62fefcf8100c3c7f` freezes this choice. Commit `9169b92ad442ffa95d4f7a317aff55484d520463` records the SCREEN demonstration. Report `results/post_screen/B1/REPORT.md`.

### Task D1 spec. Interior search around LSTM19 and EWMA01

Spec date 2026-10-29. This entry is written before any D1 fit. The frozen core YAML is not edited. New identifiers live in `configs/binance_post_screen_d1.yaml`. VALIDATION is the only selection sample. CONFIRM stays locked. No SPA, MCS, or finalist reselection is run. The deep-learning representative is not re-decided here.

LSTM19 is held fixed at 5 layers, dropout 0.2, 75 epochs, learning rate 0.001, and gradient clip 1. One factor moves at a time. The seven configurations are LSTM21 and LSTM22 for depth 6 and 7, LSTM23 and LSTM24 for dropout 0.3 and 0.4, LSTM25 and LSTM26 for 100 and 150 epochs, and LSTM27 for learning rate 0.002. Each uses seeds 0 through 4 and an equal-weight ensemble. EWMA21 through EWMA24 use decay 0.60, 0.65, 0.70, and 0.75. EWMA runs before the LSTM configurations. A checkpoint is written after each configuration-seed pair.

Selection uses the stored VALIDATION rule. The score is ensemble mean QLIKE, and an exact tie uses the lexicographically smaller identifier. LSTM19 and EWMA01 enter that comparison through their stored VALIDATION scores. They are not refit. If the winner is the outward edge of an axis whose step is fixed by this grid, at most one further configuration is fit on that axis. Depth 7 may be followed by depth 8, dropout 0.4 by 0.5, 150 epochs by 200, and EWMA decay 0.60 by 0.55. The learning-rate axis has a single new point, so a further rate is not defined. If LSTM27 wins, the extra step is not invented. The search stops and the edge is reported. If the baseline identifier still wins, no further outward step is fit.

Ridge is not extended. A penalty of 0 is ordinary least squares, not a point on a longer shrinkage path. XGB08 is not extended. Its depth and learning rate sit between other points of the frozen grid. The searched LSTM-BEKK count is 20 plus the number of new configurations actually fit. The searched EWMA count is 20 plus the number of new decays actually fit.

If the selected identifier changes, that identifier is run on SCREEN under its new identifier. Outputs go under `results/post_screen/D1/`.

### Task D1 result. Interior search around LSTM19 and EWMA01

Result date 2026-10-30. VALIDATION selected EWMA21, decay 0.60, and LSTM22, depth 7 with the other LSTM19 settings held fixed. The declared outward steps EWMA25, decay 0.55, and LSTM28, depth 8, were fit and lost. After those steps the winners sit between searched neighbors on the moved factor. The searched totals are EWMA 25 and LSTM-BEKK 28. Both new identifiers were scored on SCREEN. LSTM19 remains the deep-learning representative. Commit `afb951a005b6ad22e1b12097eb579ef01aef852f` records the scores. Report `results/post_screen/D1/REPORT.md`.

### Amendment R-2026-10-30

Amendment date 2026-10-30. The items below were fixed after reviewing the completed tasks R1, C1, B1, and D1. None uses the locked holdout. CONFIRM stays locked. Earlier register entries are unchanged.

Family configurations. Under the VALIDATION rule, the LSTM-BEKK family configuration is LSTM22, depth 7, with the other LSTM19 settings held fixed. The EWMA family configuration is EWMA21, decay 0.60. D1 selected both on VALIDATION. The outward steps, depth 8 and decay 0.55, were fit and lost, so both winners are interior on the extended one-factor grids. LSTM19 and EWMA01 stay in all tables as the original-grid comparisons. The switch follows the VALIDATION rule. On SCREEN, LSTM22 mean QLIKE is −31.298 against LSTM19 at −31.409, and EWMA21 is −31.784158 against EWMA01 at −31.776804. That VALIDATION-to-SCREEN reversal for LSTM is a finding about the reliability of 250-day configuration selection. G1 re-runs the frozen finalist rule with these configurations and accepts the outcome.

Departure from the LSTM-BEKK paper. Depth 7 lies outside the source paper's architecture range of 3 to 5. It is admitted only through the named opt-in `architecture_range` `post_screen_d1`. This is a declared departure.

Pairwise test. P5, the equal-weighted cosine estimator, is the primary confirmatory pairwise procedure. The frozen fallback rule selected it because no candidate qualified. Under the resampled SCREEN headline-gap null at \(T=500\), P5 rejects 0.0834 at block length 7 and 0.0774 at block length 29, against a nominal size of 0.05. Under Student-\(t(5)\) at \(\rho\le 0.6\) it stays inside \([0.03, 0.075]\). A size-adjusted companion is frozen in task B1b. A confirmatory pairwise claim requires rejection at the ordinary P5 critical value and at both size-adjusted critical values. The win rate, subperiod signs, and cumulative-loss plot are always reported beside the test.

DCC carry-forward caveat. DCC02 and DCCNL02 each held parameters at 9 of 24 SCREEN refits and at no VALIDATION refit. Every DCC02 and DCCNL02 result is reported with that count, beside the interior-map variants DCC03 and DCCNL03.

Deferral. D3b, the phase-shifted augmentation, and D4, the HEAVY matched pairs, are deferred until after CONFIRM. They are excluded from the confirmatory roster. A later run is a post-holdout extension and cannot be confirmatory on this holdout. D3a's phase arrays are kept.

D2 configuration. D2's variants use LSTM22's configuration. V00 is LSTM22 itself, already scored in D1, and is not refit.

### Task B1b spec. Size-adjusted critical values for P5

Spec date 2026-10-30. This entry is written before any B1b simulation. The null is the B1 design D3. The SCREEN HARQ-DRD minus LSTM-BEKK QLIKE gap is demeaned and resampled by the existing stationary bootstrap at \(T=500\) and block lengths 7 and 29. The script calls the existing path builder and `diebold_mariano(..., reference="ewc")`. It does not reimplement either. CONFIRM stays locked. The cutoffs are not applied to any model comparison in this task.

For each block length, 20000 null replications are drawn. The size-adjusted two-sided critical value is the 0.95 quantile of the absolute P5 statistic, using NumPy's linear quantile. The ordinary P5 critical value is the existing Student-\(t\) cutoff at \(\nu=\lfloor 0.4 T^{2/3}\rfloor\). The quantile standard error is the sample standard deviation of that same quantile across 1000 bootstrap resamples of the 20000 absolute statistics. The seed is the existing calibration seed 20260916. The null keys are `B1b-null-7` and `B1b-null-29`. The quantile-standard-error keys are `B1b-se-7` and `B1b-se-29`. A fresh check uses 5000 replications at keys `B1b-check-7` and `B1b-check-29`. The check passes when the rejection rate at the frozen cutoff lies within two Monte Carlo standard errors of 0.05, using \(\sqrt{0.05\times 0.95/5000}\). The values are frozen after this run and are not recomputed on CONFIRM. A confirmatory claim still requires rejection at the ordinary P5 cutoff and at both size-adjusted values, as stated in amendment R-2026-10-30.

The entry script is `scripts/p5_size_adjusted_cutoffs.py`. Outputs go under `results/post_screen/B1b/`.
