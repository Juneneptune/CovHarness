# Project state

Last updated 2026-09-21.

Project conda environment is `covharness` (Python 3.11). Recreate with `conda env create -f environment.yml` from the repository root.

## Current milestone

Binance open-data SCREEN execution and Stage-2 finalist selection. VALIDATION configurations are frozen. SCREEN forecasts, SPA, MCS, and the median-rank econometric finalist exist. Those SCREEN quantities are selection data, not confirmation. CONFIRM remains locked. The U.S.-equity DATA GATE remains closed. LSTM-BEKK-RC and GHAR were not begun.

The public research narrative is `README.md`. Exact model contracts are in `docs/MODEL_IMPLEMENTATION.md`. Inference implementation is in `docs/INFERENCE_METHODS.md`. The U.S.-equity quote path is in `docs/EQUITY_MEASUREMENT.md`. Frozen scientific artifacts were not rewritten.

`INITIAL_CORE_BENCHMARK_PLAN.md` is the initial core plan. It is organized by blocks and parts. It is not a locked roster or a final evaluation specification. It does not use writing-day or resume framing. Forecast-horizon and rolling-window lengths in trading days are unchanged.

## Completed

Block 1 measurement, identity repair, and root-only reconciliation are closed.

Block 2A covariance-space losses and the proxy-robustness demonstration are closed. Documentation-only rename after Block 2A. The probability that $H_B$ has lower loss on a single noisy proxy draw is `single_draw_hb_win_rate`, not a "flip rate". Proxy robustness concerns expected loss rankings, not pointwise rankings on every noisy realization. The Block 2A mathematics are unchanged.

Losses implemented in `covharness.losses`.

Squared Frobenius.

```math
L_F(S,H)=\|S-H\|_F^2=\operatorname{tr}((S-H)^{\top}(S-H))=\sum_{ij}(S_{ij}-H_{ij})^2
```

Equal square dimensions, finite entries, symmetry within `SYMMETRY_ATOL=1e-10`. PSD is not required to compute it. No scaling or annualization. Scalar float. Inputs are not mutated.

Reduced multivariate QLIKE, the primary ranking loss.

```math
L_Q(S,H)=\log\det(H)+\operatorname{tr}(H^{-1}S)
```

`S` is square, finite, symmetric, and PSD, and it may be singular. `H` is strictly PD. Cholesky `H=LL^{\top}` supplies both `logdet(H)=2\sum\log\operatorname{diag}(L)` and the triangular solves for `tr(H^{-1}S)`. `inv(H)` is not formed. A failed Cholesky is the PD failure. No jitter, diagonal loading, clipping, or silent fallback.

Full Stein, when both arguments are SPD.

```math
L_S(S,H)=L_Q(S,H)-\log\det(S)-N
```

Reduced QLIKE and full Stein are not numerically equal. For a common SPD target they differ by a target-only term, so rankings and pairwise differentials agree. Singular `S` is rejected by full Stein.

No matrix repair inside evaluation. A non-PD forecast raises `ForecastNotPositiveDefiniteError`.

Descriptive localization. `variance_squared_error` is $\sum_i(S_{ii}-H_{ii})^2$. `correlation_frobenius_squared` is $\|R(S)-R(H)\|_F^2$ with $R(A)=D(A)^{-1}AD(A)^{-1}$. These are not additive pieces of the ranking losses and do not inherit Patton / LRV ranking consistency. Nonpositive diagonals are rejected.

Singular-proxy contract. Rank-1 `S=[[1,1],[1,1]]` with SPD `H` yields finite squared Frobenius and finite reduced QLIKE. Full Stein fails clearly. Rank deficiency does not by itself prevent evaluation. It can still increase proxy noise and reduce power.

Proxy-robustness demonstration. `S=u\Sigma` with `u\sim\mathrm{Exp}(1)`, so $\mathrm{E}[S]=\Sigma$. `H_A=\Sigma`. `H_B=\log(2)\Sigma`. Analytic expected squared Frobenius is 2.18 versus 2.385. Analytic expected reduced QLIKE is 1.906 versus 2.058. Unsquared Frobenius is 1.086 versus 1.023, so the non-robust criterion ranks `H_B` first. Monte Carlo, seed 20260212, 25,000 draws. Mean squared Frobenius 2.227 versus 2.431. Mean reduced QLIKE 1.903 versus 2.055. Mean full Stein 1.185 versus 1.336. Mean unsquared Frobenius 1.098 versus 1.031. Single-draw $H_B$ win rates are 0.576 (squared Frobenius), 0.568 (reduced QLIKE), 0.568 (full Stein), and 0.576 (unsquared Frobenius). For this scale family the per-draw Frobenius ranking coincides while expected rankings differ.

`M/N` is reported by `matrix_eigen_diagnostics` when `n_returns` is supplied, and as `FrequencyEppsResult.m_over_n`. The TAQ measurement investigation was not rerun.

`INITIAL_CORE_BENCHMARK_PLAN.md` no longer states that `M>N` is required for QLIKE evaluation. Kernels remain in the broader plan as alternative proxies.

Block 2B temporal protocol is implemented in `covharness.protocol`.

Four chronological regions. HISTORY (rolling burn-in), VALIDATION, SCREEN, CONFIRM. Ordering is HISTORY < VALIDATION < SCREEN < CONFIRM. No shuffling. No overlapping evaluation dates among VALIDATION, SCREEN, and CONFIRM.

Boundary convention. Half-open intervals $[start, end)$ on a strictly increasing trading-date index. Evaluation blocks are forecast *targets* (day $t+1$). For target index $i$, the origin is $i-1$ and the estimation window is $[i-m, i)$, equivalently the $m$ dates through the origin, excluding the target.

Target length. VALIDATION has target 250 trading-day targets. SCREEN and CONFIRM have committed minima of 500. SCREEN and CONFIRM are never shortened. If the preferred allocation cannot be met, VALIDATION is shortened first and `validation_shortened` is reported. If $T-m<1000$, allocation raises `ProtocolAllocationError`. Surplus evaluation days go to VALIDATION. A zero-length VALIDATION block sets `data_driven_tuning_available=False` and `require_data_driven_tuning()` raises `TuningUnavailableError`. That design is not described as having completed hyperparameter tuning.

Rolling method. $m=250$ trading days. Monthly refit cadence of 21 forecast origins. At origin $t$, inputs may use observations through $t$. The target is $t+1$. The same schedule is reusable by later conventional, ML, and DL models. No forecasting model is implemented in this block.

Preprocessing. `fit_on_estimation_window` fits a scaler on the rolling window through $t$ only. Passing a full-sample series does not leak. VALIDATION may be transformed after that fit. SCREEN and CONFIRM do not enter the fit. A naive full-sample scaler is shown to be pulled by a CONFIRM spike that the protocol scaler ignores.

CONFIRM lock. Default `LOCKED = True`. `confirm_targets()` and `forecast_schedule(CONFIRM)` raise `ConfirmLockedError`. Explicit `unlock_confirm=True` succeeds. Public helpers (`evaluation_targets`, `visible_calendar`, `all_evaluation_targets` without the flag) omit CONFIRM dates. The object cannot be constructed unlocked and cannot be mutated to unlocked. The later runner will expose `--unlock-confirm`. No model runner was added.

Seed contract. Frozen list $(0,1,2,3,4)$. Report all. No best-seed selection. Aggregation is the equal-weight mean ensemble.

Tuning budget. Auditable cap of 20 configurations per model family. No search engine.

Overnight convention (metadata only). Statistical evaluation uses open-to-close RCov. Later economic GMV risk includes the overnight outer product, with open-to-close-only as a robustness channel. GMV and overnight RCov are not implemented.

Preregistration status. `PREREGISTRATION_DRAFT.md` records the frozen protocol decisions. It is not immutable and not final. Final `PREREGISTRATION.md` is created once, after the graph-neural specification and empirical dataset are frozen, and before Block-4 empirical fitting. After that it is never edited.

DATA GATE. Block 2 and Block 3 used synthetic known-truth data. Empirical Block-4 fitting must not begin until the empirical panel is committed and verified. Model-engineering code may be added and validated on synthetic inputs. The long historical dataset remains unresolved. No dataset was chosen in this block.

GHAR is recorded as a Block-4 structured graph / econometric covariance baseline. It is not a deep-learning model. The graph-neural DL slot remains unresolved. iTransformer is optional and not committed.

Block 3A pairwise inference is implemented in `covharness.inference`.

Sign convention. $d_t = L_{A,t}-L_{B,t}$. $d_t<0$ means A wins that date. $d_t>0$ means B wins. The null is $\mathrm{E}[d_t]=0$. Alternatives are two-sided ($\mathrm{E}[d]\neq 0$), A-better ($\mathrm{E}[d]<0$), and B-better ($\mathrm{E}[d]>0$).

Diebold-Mariano statistic $\mathrm{DM}=\bar d / \widehat{\mathrm{se}}_{\mathrm{HAC}}(\bar d)$. p-values use $N(0,1)$. The Harvey-Leybourne-Newbold correction is not applied. The design did not commit to it.

HAC. Bartlett / Newey-West kernel $k(j)=1-j/(L+1)$. Autocovariances use the $1/T$ divisor on the demeaned series. Long-run variance $\hat\omega=\hat\gamma_0+2\sum_{j=1}^L k(j)\hat\gamma_j$. Default lag $L=\lfloor 4(T/100)^{2/9}\rfloor$ (Newey-West 1994). An explicit lag is allowed. $L=0$ is heteroskedasticity-only. If every supplied observation is exactly equal, HAC is undefined and `DegenerateLossDifferentialError` is raised before demeaning. That check exists because mean subtraction of a constant such as $0.1$ can leave a tiny nonzero residual and an enormous finite DM statistic. A genuinely nonconstant series with small variance is not rejected. A non-finite long-run variance is rejected rather than stored. A numerically tiny negative long-run variance is treated as zero and then rejected as degenerate. No jitter is added. When automatic HAC fails, diagnostic `hac_maxlags` reports the HAC lag that was attempted, not the ACF lag count. The intercept-only HAC standard error matches statsmodels OLS with `cov_type='HAC'`, Bartlett kernel, and `use_correction=False`. The live `covharness` environment now has the declared `statsmodels` dev extra installed so that reference test executes.

Clark-West is a separate procedure. The caller must pass `nested=True`. The implemented formula is the classical scalar squared-error adjustment $d_t^{\mathrm{CW}}=(y_t-f_{R,t})^2-(y_t-f_{U,t})^2+(f_{R,t}-f_{U,t})^2$. Matrix losses, including reduced QLIKE and squared Frobenius, are rejected. A future nested covariance comparison may need a loss-specific justified procedure.

Diagnostics. $C_t=\sum_{s\le t}d_s$. Sample ACF of $d_t$. $\kappa=\hat\omega/\hat\gamma_0$ and $T_{\mathrm{eff}}=T/\kappa$ when defined. $T_{\mathrm{eff}}$ is not clipped. Under negative serial correlation $\kappa$ can be below one and $T_{\mathrm{eff}}$ can exceed $T$.

Synthetic size and power, seed 20260912, 2000 replications, $T=250$, two-sided 5 percent. IID null. naive $t$ rejection 0.052, HAC DM 0.0545. AR(1) null with $\rho=0.6$. naive $t$ rejection 0.3185, HAC DM 0.1135. The naive test over-rejects. HAC is closer to nominal. Mean-shift alternative $\mathrm{E}[d]=-0.20$. two-sided HAC DM rejection 0.8965, A-better 0.946.

Block 3B SPA and MCS are implemented in `covharness.inference`. They operate on a finite loss matrix of shape $(T,M)$ for one loss/proxy channel. Lower loss is better. Inputs are copied. Reduced QLIKE and squared Frobenius are never averaged. Distinct covariance proxies are never pooled.

Stationary bootstrap. Joint Politis-Romano time indices, circular wraparound, restart probability $q=1/\ell$ with $\ell=\max(2,\lfloor T^{1/3}\rfloor)$. The cube-root rule is a function of $T$ only. It does not inspect loss ACFs or rankings. $q=T^{-1/2}$ is not used because it would violate $T q_T^2\to\infty$. Defaults are $B=5000$ and seed $20260913$. MCS reuses one index matrix through every elimination step.

SPA. Hansen (2005). Differential $d_{k,t}=L_{0,t}-L_{k,t}$. Positive means the alternative beats the supplied benchmark. Null $\mathrm{E}[d_k]\le 0$ for every $k$. Statistic $T^{\mathrm{SPA}}=\max(0,\max_k \sqrt{T}\bar d_k/\hat\omega_k)$. Long-run variance is the stationary-bootstrap geometric kernel, not Block 3A Bartlett/Newey-West HAC. The same $\hat\omega_k$ is used for the observed statistic and every replicate. Three recenterings $g_l$, $g_c$, $g_u$ are implemented, with consistent threshold $-\sqrt{2\log\log T}$. p-values use $\mathrm{mean}(T^\ast>T)$. Primary p-value is the consistent recentering, with $\hat p^l\le\hat p^c\le\hat p^u$. Headline $\alpha=0.05$. Exact-constant differentials raise `DegenerateLossDifferentialError`.

MCS. Hansen-Lunde-Nason (2011). Differential $d_{ij,t}=L_{i,t}-L_{j,t}$. Positive means $i$ is worse. Both coherent pairs are implemented. Primary SCREEN procedure is $(T_R,e_R)$. Companion is $(T_{\max},e_{\max})$. The pairs are not crossed. Studentization uses standard errors. EPA p-values use $\mathrm{mean}(T^\ast\ge T)$. MCS p-values are the running maximum along elimination, so $i$ is in the MCS at $\alpha$ iff $\hat p_i\ge\alpha$. Last survivor has p-value 1. Frozen SCREEN membership is $\alpha=0.10$. Ties break by original column index. Identical columns are ties with $t_{ij}=0$. A nonzero constant pair raises `DegenerateMCSDifferentialError`.

Synthetic validation, seed 20260913, $B=5000$, $T=120$. Equal-performance SPA consistent p-value 0.4066, not rejected at 0.05. One clearly better alternative, statistic 25.736, consistent p-value 0.0, rejected. Adding ten poor alternatives left the consistent p-value at 0.0. MCS range and max both retained two equivalent best models and eliminated the inferior column. These examples are validation demonstrations, not empirical findings.

Block 3C generic inference is implemented in `covharness.inference`. Blocks 3A and 3B methodology was not changed. The optional Odendahl–Rossi–Sekhposyan threshold extension was not implemented. The One-Time Reversal test was not implemented.

Giacomini-White. One-step only. $d_t=L_{A,t}-L_{B,t}$. $Z_t=h_t d_t$. $\bar Z=\mathrm{mean}_t Z_t$. $\widehat{\Omega}=T^{-1}\sum_t Z_t Z_t^{\top}$ with no demeaning and no Bartlett HAC. Statistic $GW=T\bar Z^{\top}\widehat{\Omega}^{-1}\bar Z\sim\chi^2_q$. Frozen $\alpha=0.05$. A nonzero constant differential with full-rank instruments is a valid test. All-zero moments, singular or non-finite $\widehat{\Omega}$, and rank-deficient instruments raise `DegenerateGWCovarianceError` or `RankDeficientGWInstrumentsError`. No ridge, jitter, pseudo-inverse, or column dropping. Market-state instruments are origin-day $h_t=[1,\log(\mathrm{mean}_i S_{ii,t}),\mathrm{mean}_{i<j}R_{ij}(S_t)]$ and are not standardized. Measurement/stress instruments are origin-day $[1,\log(\mathrm{RQ}_{\mathrm{agg}}),\mathbf{1}\{\mathrm{BNS\ jump}\}]$ from synchronized returns. The family helper runs two already-constructed matrices separately and reports Bonferroni family size 2, family $\alpha=0.05$, cutoff $0.025$, and $p_{\mathrm{adj}}=\min(1,2p)$. Six instruments are never stacked.

Pooled-vech MZ. For unique $i\le j$, $S_{ij,t}=\alpha+\beta H_{ij,t}+e_{ij,t}$ with one common intercept and slope. Null $(\alpha,\beta)=(0,1)$. This is the Patton–Sheppard common-coefficient representation. The 100-random-portfolio design is not used. Stacked unique entries are not treated as independent. Daily scores sum pair-level contributions within each date. Sandwich inference uses Bartlett / Newey-West HAC on the daily score sequence with the Block 3A lag convention. Frozen $\alpha=0.05$. Diagonal-only and off-diagonal-only pooled coefficients are descriptive. Exact calibration returns Wald $0$, p-value $1$, and `covariance=None`. An exact linear violation returns Wald $\infty$, p-value $0$, and `covariance=None`.

State-augmented MZ adds origin-day $z_t=[\log(\mathrm{mean}_i S_{ii,t}),\mathrm{mean}_{i<j}R_{ij}(S_t)]$ with common project-specific $\gamma_1,\gamma_2$. That pooled restriction is not Patton–Sheppard. Null $\alpha=0$, $\beta=1$, $\gamma=0$. Same daily-score HAC sandwich. GW tests whether state predicts relative loss. Augmented MZ tests whether state predicts one model's calibration error.

Approximate Patton–Sheppard equation-21 weighting. Scale $s_{ij,t}=\sqrt{H_{ii,t}H_{jj,t}}$, not the product. The same scale divides $y$ and every $X$ column. The transformed forecast is the forecast correlation. On the diagonal this is division by $H_{ii,t}$. Label `approximate_ps21`. Never `exact_gls`. Forecast diagonals must be finite and strictly positive. No epsilon floor. Weighting does not remove serial dependence. Inference remains daily-score HAC. Approximate PS21 is a robustness re-estimation. OLS versus WLS is not selected by which rejects.

Giacomini–Rossi fluctuation. Proposition 1, GW-method version. $d_t=L_{A,t}-L_{B,t}$. Negative local $F_t$ means A is locally better. Frozen $\mu=0.30$. $m=2\lfloor 0.30 P/2\rfloor$, always even. Require $P>m$ and $m\ge 2$. $F_t=\hat\omega^{-1}m^{-1/2}\sum_{j=t-m/2}^{t+m/2-1}d_j$. The limit divides by $\sqrt{\mu}$, not $\mu$. Two-sided only. Critical value $k=3.012$ at $\alpha=0.05$. Reject iff $\max|F_t|>3.012$. Centered windows only. Path length $P-m+1$. Global uncentered LRV $\gamma_j^0=P^{-1}\sum d_t d_{t-j}$ with Block 3A Newey–West 1994 lags and Bartlett weights. The demeaned Block 3A `hac_long_run_variance` is not called. $\hat\omega$ is computed once. Zero, negative, or non-finite $\hat\omega^2$ raises `DegenerateFluctuationVarianceError`.

Multiplicity metadata. GW family size 2. Standard MZ on two finalists family size 2. Augmented MZ on the same two finalists is a separate family of size 2. Diagonal/off-diagonal MZ remains descriptive.

Origin-day measurement/stress state is implemented in `covharness.features`. Inputs are already-synchronized intraday returns of shape $(M,N)$ or a panel $(T,M,N)$. Raw TAQ and the realized-covariance estimator were not changed. Per-asset quarticity is $\mathrm{RQ}_i=(M/3)\sum_j r_{j,i}^4$. Aggregate $\mathrm{RQ}_{\mathrm{agg}}=\mathrm{mean}_i\mathrm{RQ}_i$ must be strictly positive before $\log$. No annualization, winsorization, or clipping. The jump state is the Barndorff-Nielsen and Shephard (2006) adjusted-ratio test on $\bar r_j=\mathrm{mean}_i r_{j,i}$, with $\delta=1/M$, $\mu_1=\sqrt{2/\pi}$, $\vartheta=\pi^2/4+\pi-5$, and $Z_{\mathrm{jump}}=J_{\mathrm{BNS}}/\sqrt{\vartheta}$. Frozen $\alpha_{\mathrm{jump}}=0.01$. Indicator $1\{Z_{\mathrm{jump}}<\Phi^{-1}(0.01)\}$. Large negative values indicate jumps. The tail is not reversed. The threshold is not estimated from the sample. This is a common-market jump in the equal-weight intraday portfolio. It is not an any-constituent-jumped indicator. Zero RV, zero BV, nonfinite or negative QP, and insufficient $M$ raise `InvalidOriginStateError`. The measurement GW vector is origin-day $[1,\log(\mathrm{RQ}_{\mathrm{agg}}),\mathbf{1}\{\mathrm{BNS\ jump}\}]$ and is not standardized. `giacomini_white_two_specifications` is unchanged as a two-test Bonferroni helper.

Block 3C is accepted as closed in the project continuation.

Block 4A-1 covariance-model contract is implemented in `covharness.models`.

Public types. `CovarianceModel` is the forecast and identity contract. `RealizedCovarianceModel.fit` consumes a copied `(T, N, N)` realized-covariance window through the origin. `CovarianceForecast` returns an independent `(N, N)` matrix, `ForecastDiagnostics`, and `ModelIdentity`. Models do not inspect VALIDATION, SCREEN, or CONFIRM. The protocol supplies the window.

Input contract. `T>=1`. Square finite slices. Symmetry within `SYMMETRY_ATOL=1e-10`. PSD within `PSD_ATOL=1e-12`. Strict PD is not required. Asymmetry and indefiniteness raise `InvalidModelInputError`. No symmetrization, shrinkage, jitter, clipping, diagonal loading, or eigenvalue flooring.

Random walk. $H_{t+1\mid t}=S_t$, an independent copy of the last window matrix. Earlier slices are ignored. A singular PSD origin matrix remains singular. QLIKE still requires a PD forecast at evaluation. That limitation is diagnosed (`positive_definite=False`) rather than repaired.

EWMA. $H_0=S_0$ and $H_j=\lambda H_{j-1}+(1-\lambda)S_j$ for $j=1,\ldots,T-1$, with forecast $H_{T\mid T-1}=H_{T-1}$. Decay $\lambda$ is a required constructor argument in $(0,1)$. The RiskMetrics reference $0.94$ may be passed by a caller. It is not a tuned project default. The 20-point VALIDATION grid is not frozen and was not run. This EWMA is on realized covariance, not daily-return outer products. EWMA of PSD inputs is PSD. A shared null space can remain singular.

Diagnostics reuse `matrix_eigen_diagnostics`. Strict PD is a Cholesky test matching evaluation losses. Condition number is omitted when the smallest eigenvalue is not strictly larger than `PSD_ATOL`.

A synthetic protocol check slices a `(T, N, N)` cube with `ForecastStep` on a VALIDATION origin. CONFIRM remains locked. No runner was added.

Files created. `src/covharness/models/exceptions.py`, `base.py`, `random_walk.py`, `ewma.py`, and `tests/unit/test_models_rcov_baselines.py`. `src/covharness/models/__init__.py` now exports the public API.

Block 4A-2 HAR-DRD is implemented in `covharness.models.har_drd` as `HARDRDRealizedCovariance`. Public identity `har_drd`. It consumes the same origin-window contract as random walk and EWMA.

DRD. For each supplied $S_t$, $v_t=\operatorname{diag}(S_t)$, $D_t=\operatorname{diag}(\sqrt{v_t})$, $R_t=D_t^{-1}S_t D_t^{-1}$. Every diagonal must be strictly positive. A PSD matrix with a zero diagonal is rejected with `InvalidModelInputError`. Inputs are not altered.

Pair order. Strict upper triangle $i<j$ via `np.triu_indices(N, k=1)`, the same unique-pair order already used by Epps and Giacomini–White. $P=N(N-1)/2$. Reconstruction uses that order with unit diagonal.

HAR features. Zhang-style non-overlapping lags. Response dates $k=22,\ldots,T-1$. Daily is $k-1$. Weekly is the mean of exactly four observations `[k-5:k-1]`. Monthly is the mean of exactly seventeen observations `[k-22:k-5]`. The blocks do not overlap and do not read before local index 0. For $T=250$ there are $228$ regression dates. Origin predictors for the target after the window are `[T-1]`, `[T-5:T-1]`, and `[T-22:T-5]`.

Equations. Variances have asset-specific intercepts and three shared scalar slopes. Correlations have pair-specific intercepts and three shared scalar slopes. The headline model is in levels. There is no log-variance transform, Fisher transform, ridge, graph term, or HARQ term.

Estimation. Exact fixed-effects within transformation. Group-demean $y$ and the three-column $X$, stack only the demeaned three-column design, solve $\beta$ by `lstsq`, recover intercepts from group means. The dummy-variable helper exists only in tests. The fitted model never builds an $N$-column or $P$-column intercept design. Stored state includes $\alpha_D$, $\beta_D$, $\alpha_R$, $\beta_R$, $N$, $P$, window length, regression-date count, within-design ranks, pair ordering, and lag convention. OLS standard errors are not stored.

Repair. Explicit BPQ-style insanity filter. Raw $H$ is used only when $v_{\mathrm{raw}}$ is finite and strictly positive, $x_{\mathrm{raw}}$ is finite, pairwise correlations lie in $[-1,1]$, and both $R_{\mathrm{raw}}$ and $H_{\mathrm{raw}}$ are strictly PD. Otherwise the headline matrix is the arithmetic mean of the current supplied estimation window, `repaired=True`, `repair_method="estimation_window_mean"`. If that mean is not strictly PD, `InvalidModelForecastError` is raised. No second repair. No clipping, nearest-PD, jitter, diagonal loading, or entrywise replacement. Failure flags are stored. Repair frequency is reportable. Raw components are preserved for diagnostics.

A synthetic protocol check on a $m=250$ VALIDATION window confirms that CONFIRM remains locked. No empirical fitting occurred.

Files created or materially changed. `src/covharness/models/har_drd.py`, `src/covharness/models/exceptions.py` (`InvalidModelForecastError`), `src/covharness/models/__init__.py`, `tests/unit/test_models_har_drd.py`, `README.md`, and `docs/PROJECT_STATE.md`.

Block 4A-3 HARQ-DRD is implemented in `covharness.models.harq_drd` as `HARQDRDRealizedCovariance`. Public identity `harq_drd`. It inherits `CovarianceModel` rather than `RealizedCovarianceModel`, because `fit` requires both a `(T, N, N)` covariance window and a matching `(T, N)` per-asset realized-quarticity window. Random walk, EWMA, and HAR-DRD keep the covariance-only `fit`.

Variance equation. Asset-specific $\alpha_Q$ and four shared slopes $(\beta_{Q,d},\phi_{Q,d},\beta_{Q,w},\beta_{Q,m})$. The extra term is $\phi_{Q,d}(\sqrt{\mathrm{RQ}_{k-1}}\odot v_{k-1})$. No sign constraint on $\phi_{Q,d}$. No weekly or monthly quarticity term. No log-variance transform.

Correlation equation. Unchanged HAR-DRD three-slope map. Same pair order. No Fisher transform and no correlation attenuation.

RQ contract. Precomputed $(T,N)$ panel aligned with the covariance history. Definition label `per_asset_rq_m_over_3_sum_r4`, meaning $\mathrm{RQ}_i=(M/3)\sum_l r_{i,l}^4$. Finite and nonnegative. Zero allowed. No annualization, winsorization, clipping, or standardization. This is not the GW aggregate $\mathrm{RQ}_{\mathrm{agg}}=\mathrm{mean}_i\mathrm{RQ}_i$ and is not logged. The model does not read raw intraday returns. Empirical TAQ-to-RQ assembly waits for the DATA GATE.

Estimation. The HAR-DRD within estimator now accepts any slope width. HARQ variance uses four demeaned columns. Correlation uses the existing three. Dummy intercept matrices are not built. Rank is stored and is not used as an automatic rejection unless `lstsq` already fails.

Repair. Shared `headline_repaired_forecast` with HAR-DRD. Same validity flags. Same estimation-window-mean fallback. No second repair.

A synthetic protocol check on a $m=250$ VALIDATION window confirms that CONFIRM remains locked. No empirical fitting occurred.

Files created or materially changed. `src/covharness/models/harq_drd.py`, `src/covharness/models/har_drd.py` (shared within estimator and headline repair helper), `src/covharness/models/__init__.py`, `tests/unit/test_models_harq_drd.py`, `README.md`, and `docs/PROJECT_STATE.md`.

Block 4A-4 standalone Ledoit-Wolf shrinkage is implemented in `covharness.models.ledoit_wolf`. Public classes are `LedoitWolfLinearCovariance` (identity `lw_linear`) and `LedoitWolfNonlinearCovariance` (identity `lw_nl`). Both inherit `CovarianceModel` and consume a copied `(T, N)` daily-return window. They do not consume realized-covariance histories. They do not shrink a DCC targeting matrix. DCC-NL remains a later model.

Return contract. `T>=2`. `N>=1`. Finite values. Caller column order is asset order. Inputs are copied. Open-to-close versus close-to-close is not chosen here. Helper `as_daily_return_history` lives in `covharness.models.base` next to the realized-covariance helper.

Centering and sample covariance. Inside `fit`, `mean = returns.mean(axis=0)`, `Y = returns - mean`, `n_eff = T-1`, and `S = Y^{\top}Y/(T-1)`. The caller is not assumed to have demeaned the window. There is no annualization, scaling, winsorization, or silent missing-value deletion. The same $S$ is the starting object for both estimators.

LW-linear. Ledoit-Wolf 2004b rotation-equivariant shrinkage $\Sigma_L=(1-\rho)S+\rho\mu I$ with $\mu=\operatorname{tr}(S)/N$. $\rho$ is estimated. It is not a user-chosen intensity. Honey / equicorrelation (2004a) is not the headline estimator. Sample eigenvectors are retained. Every sample eigenvalue receives the same affine map $d_i=(1-\rho)\lambda_i+\rho\mu$. The 2004b coefficient is implemented in-house. sklearn 1.9.1 `LedoitWolf` is the independent test reference only and is a pinned dev extra, not a runtime dependency. sklearn uses a $1/T$ Gram matrix. Tests feed $Y_{\mathrm{ref}}=\sqrt{T/(T-1)}\,Y$ with `assume_centered=True` so that $(1/T)Y_{\mathrm{ref}}^{\top}Y_{\mathrm{ref}}$ equals $S$. Production $\rho$ and $\Sigma_L$ match that reference.

LW-NL. Wrapper of pinned runtime dependency `nonlinshrink==0.7` (MIT, https://github.com/matzhaugen/analytic_shrinkage), a port of the 2018 working paper that became Ledoit-Wolf 2020 analytical nonlinear shrinkage. It is not QuEST and not QIS 2022. The kernel/Hilbert formulas are not transcribed. Centered $Y$ is passed with `k=1` so the reference uses $n_{\mathrm{eff}}=T-1$ and the same $S$. Sample eigenvectors are retained. Shrunk eigenvalues are eigenvalue-specific. The reference requires $n_{\mathrm{eff}}\ge 12$, so $T\ge 13$. The $N\ge T$ supplement branch is exposed by the package and is included. A nonfinite, asymmetric, or non-strictly-PD reference matrix raises `InvalidModelForecastError`. No silent repair, jitter, or eigenvalue floor.

Forecast semantics. $H_{t+1\mid t}=\widehat{\Sigma}_t$ from the current supplied window. The model does not update between scheduled common refits and does not inspect VALIDATION, SCREEN, or CONFIRM labels. No runner was added.

Fit-state metadata. Linear stores the in-window mean, $T$, $N$, $n_{\mathrm{eff}}$, centering, divisor `T_minus_1`, $\rho$, $\mu$, and sample eigenvalues. Nonlinear stores the same centering contract plus reference package, version, analytical method label, sample eigenvalues, and nonlinear shrunk eigenvalues. Full eigenvector matrices are not stored.

A synthetic known-$\Sigma$ draw (seed 20260916, $T=60$, $N=4$, spectrum $(8,3,1.2,0.4)$) records that linear eigenvalues lie on one affine map of the sample spectrum and nonlinear eigenvalues do not. That draw is a demonstration fixture, not a superiority claim.

No empirical fitting occurred. WRDS and market data were not accessed.

Files created or materially changed. `src/covharness/models/ledoit_wolf.py`, `src/covharness/models/base.py` (`as_daily_return_history`), `src/covharness/models/__init__.py`, `tests/unit/test_models_ledoit_wolf.py`, `pyproject.toml` (`nonlinshrink==0.7` runtime, `scikit-learn==1.9.1` dev extra), `README.md`, and `docs/PROJECT_STATE.md`.

Block 4A-5 original Engle DCC and DCC-NL is implemented in `covharness.models.dcc`. Public classes are `DCCCovariance` (identity `dcc`) and `DCCNonlinearCovariance` (identity `dcc_nl`). They inherit `CovarianceModel`, share one engine, and differ only in the intercept $C$. cDCC is not implemented.

Return contract. The existing helper `as_daily_return_history` is reused. $T\ge 2$. $N\ge 2$. Finite values. Caller column order is asset order. Inputs are copied. There is no silent NaN deletion, annualization, or percent scaling. Open-to-close versus close-to-close remains a data-layer choice.

Stage-one mean. At each parameter fit, $\mu_i$ is the mean of the supplied window and $\varepsilon_{i,t}=r_{i,t}-\mu_i$. A ZeroMean Gaussian GARCH(1,1) is fit to $\varepsilon_i$. Between refits, $\mu_i$ is frozen. `update` centers new returns with that same $\mu_i$.

Stage-one backcast. $h_{i,0}=\mathrm{mean}_t\varepsilon_{i,t}^2$ with divisor $T$. Finite and strictly positive. The value is passed through `arch`'s explicit backcast argument. The unconditional GARCH variance $\omega/(1-a-b)$ is not used as an initializer.

Stage-one fitter. Pinned runtime dependency `arch==8.0.0`. The production call is `mean="Zero"`, `vol="GARCH"`, $p=1$, $o=0$, $q=1$, `dist="normal"`, `rescale=False`. Package mean, scaling, and backcast defaults are not inherited. After the package fit, parameters must satisfy $\omega_i>0$, $a_i\ge 0$, $b_i\ge 0$, and $a_i+b_i<1$. The in-window variance path must be finite and strictly positive. A failed asset fails the DCC fit. Assets are not dropped. Fitted parameters are not altered. The GARCH filter is then replayed with those frozen parameters so targeting, composite likelihood, and stored state share the same $s_t=\varepsilon_t/\sqrt{h_t}$.

Plain target. Standardized residuals are not demeaned again. $\widetilde{C}=S_{\mathrm{std}}^{\top}S_{\mathrm{std}}/T$, then diagonal renormalization to $C$. Divisor $T$. Target type `sample_standardized_residual_correlation`. The sample target is guaranteed rank-deficient when $N>T$ and is rejected then. At $N=T$ the helper constructs $C$ and tests strict PD by Cholesky. There is no $N\ge T$ rejection rule, jitter, nearest-PD, or eigenvalue floor.

DCC-NL target. Same $S_{\mathrm{std}}$. Call `nonlinshrink.shrink_cov(Sstd, k=0)` under pinned `nonlinshrink==0.7`. Do not demean or rescale. Require the nonlinear covariance to be finite, symmetric, positive-diagonal, and strictly PD, then diagonally renormalize to $C_{\mathrm{NL}}$. Target type `analytical_nonlinear_standardized_residual_correlation`. Effective divisor $T$. This is not QuEST, not QIS, and not the standalone LW-NL $k=1$ wrapper. Failure does not fall back to plain $C$. The reference requires $n_{\mathrm{eff}}\ge 12$, so DCC-NL requires $T\ge 12$. Nonlinear shrinkage is not applied to final $H$.

Original DCC recursion. $Q_0=C_{\star}$. Window rows $t=0,\ldots,T-1$. $Q_t$ scores observed $s_t$ after $R_t=\mathrm{diag}(Q_t)^{-1/2}Q_t\mathrm{diag}(Q_t)^{-1/2}$. Then $Q_{t+1}=(1-\alpha-\beta)C_{\star}+\alpha s_ts_t^{\top}+\beta Q_t$. Stored `current_q` is $Q_{T-1}$. `forecast()` forms $Q_{T\mid T-1}$ without mutating state. No cDCC transformed shock. No target-day return.

All-pairs composite likelihood. Strict upper triangle $i<j$. Pair count $N(N-1)/2$. Objective $\sum_t\sum_{i<j}l_{ij,t}$ with the bivariate Gaussian correlation NLL. Contiguous $N-1$ 2MSCLE is not used. Rho is not clipped. A nonfinite or $|rho|\ge 1$ evaluation returns an invalid objective.

Second-stage optimizer. Deterministic SciPy SLSQP. Start $(0.05,0.90)$. Bounds $[0,1]\times[0,1]$. Constraint $\alpha+\beta\le 1$. No random restarts. An accepted fit requires `result.success`, finite objective, $\alpha\ge 0$, $\beta\ge 0$, and $\alpha+\beta<1$. The boundary $\alpha+\beta=1$ raises `InvalidModelForecastError`. Parameters are not clipped or moved inward.

Forecast. $h_{t+1\mid t}=\omega+a\varepsilon_t^2+b h_t$, $Q_{t+1\mid t}=(1-\alpha-\beta)C_{\star}+\alpha s_ts_t^{\top}+\beta Q_t$, $H_{t+1\mid t}=D_{t+1\mid t}R_{t+1\mid t}D_{t+1\mid t}$ with $D=\mathrm{diag}(\sqrt{h})$. Finite, symmetric, strictly PD. No repair. Repeated `forecast()` before `update` is identical.

Daily update. `update(new_return)` with shape $(N,)$. No parameter re-estimation. Advance $h$ and $Q$ one step, center the new return with the frozen fit-window mean, and store the new observed state. The future runner, not the model, interprets `ForecastStep.refit`.

No empirical fitting occurred. WRDS and market data were not accessed.

Files created or materially changed. `src/covharness/models/dcc.py`, `src/covharness/models/__init__.py`, `tests/unit/test_models_dcc.py`, `pyproject.toml` (`arch==8.0.0` runtime), `README.md`, and `docs/PROJECT_STATE.md`.

Block 4A-6 Ridge-DRD is implemented in `covharness.models.ridge_drd` as `RidgeDRDRealizedCovariance`. Public identity `ridge_drd`. It inherits `RealizedCovarianceModel` and consumes the same copied `(T, N, N)` realized-covariance window as HAR-DRD.

Scientific contrast. Ridge-DRD is the regularized HAR-DRD control. Targets, non-overlapping $1/4/17$ predictors, asset and pair fixed effects, pair order, reconstruction, and the estimation-window-mean insanity filter are the HAR-DRD objects. The only headline change is OLS shared slopes versus $\ell_2$-penalized shared slopes. There is no quarticity, cross-section, market state, pair-summary, daily-return, or macro feature. There is no log-variance or Fisher transform. $N$ separate ridge regressions are not used. Pair dummy columns are not built.

Penalty. One explicit constructor argument `lambda_ >= 0`, stored as `ridge_lambda`, shared by the variance and correlation maps. There is no tuned default. The future 20-point VALIDATION grid is not frozen and was not run.

Within transform and scaling. Group-demean $y$ and the three-column $X$. Dummy intercepts are not constructed. Predictor column $j$ is then divided by $\mathrm{scale}_j=\sqrt{\mathrm{mean}_m\widetilde X_{m,j}^2}$ with divisor $M$. An exact zero column receives scale 1 and is retained. Responses are not standardized. Scales are fit only on the supplied window.

Objective. On the scaled within design, $\hat\gamma=\arg\min_\gamma\|\widetilde y-X_{\mathrm{scaled}}\gamma\|_2^2+\lambda\|\gamma\|_2^2$. This is sum of squared errors plus the penalty. The loss is not divided by $M$. Raw slopes are $\beta_j=\gamma_j/\mathrm{scale}_j$. Intercepts $\alpha_g=\bar y_g-\bar X_g\beta$ are unpenalized. For $\lambda=0$, production uses the existing HAR `lstsq` path so the model nests HAR-DRD, including rank deficiency. For $\lambda>0$, production solves $(X_{\mathrm{scaled}}^{\top}X_{\mathrm{scaled}}+\lambda I)\gamma=X_{\mathrm{scaled}}^{\top}\widetilde y$ with SciPy `solve`. sklearn `Ridge` remains a dev/test reference only.

Later headline XGBoost-DRD must keep the same level variance target, raw correlation target, $1/4/17$ information set, and validity rule. Its pooling structure remains unresolved. Log-variance XGBoost is demoted to a possible separately named later variant and was not implemented.

No empirical fitting occurred. WRDS and market data were not accessed.

Files created or materially changed. `src/covharness/models/ridge_drd.py`, `src/covharness/models/__init__.py`, `tests/unit/test_models_ridge_drd.py`, `README.md`, and `docs/PROJECT_STATE.md`.

MZ exact-fit API. Coefficients and Wald semantics are unchanged. Exact calibration returns Wald $0$, p-value $1$, `inference_case="deterministic_null"`, `covariance=None`, and `covariance_degenerate=True`. An exact linear violation returns Wald $\infty$, p-value $0$, `inference_case="deterministic_alternative"`, and `covariance=None`. Ordinary cases return a numeric sandwich with `inference_case="regular"`. No jitter.

Synthetic origin-state demonstrations, not empirical findings. Hand quarticity on $[[1,2],[0.5,-1],[0,1]]$ gives $\mathrm{RQ}=(1.0625,18)$ and $\mathrm{RQ}_{\mathrm{agg}}=9.53125$. Hand BNS market path $(0.2,0.1,0.3,0.1,0.2)$ gives $\mathrm{RV}=0.19$, $\mathrm{BV}=0.1$, $\mathrm{QP}=0.006$, $J_{\mathrm{BNS}}=-0.3874$, $Z=-0.4965$, indicator $0$. Seed 20260914 Gaussian continuous panel, $M=80$, has $Z=-1.493$ and indicator $0$. The same panel with a $0.2$ common jump at interval 40 has $Z=-11.237$ and indicator $1$. Exact MZ $S=H$ returns `deterministic_null` with `covariance=None`. $S=2H$ and $S=H+c$ return `deterministic_alternative` with `covariance=None`.

Synthetic validation, seed 20260914. These are validation demonstrations, not empirical findings. Alternating-block state-dependent $d_t$ with near-zero mean. Unconditional DM statistic $-0.0700$, p-value $0.9442$. GW statistic $793.437$, p-value $5.10\times 10^{-173}$, df $2$. Perfect pooled MZ recovers $\hat\alpha=0$, $\hat\beta=1$, p-value $1$. Intercept shift $0.4$ and slope $1.6$ both yield p-value $0$. A state-dependent intercept that averages to zero has standard MZ p-value $0.9988$ and augmented MZ p-value $0$ with $\hat\gamma_1=1.2$. Approximate PS21 off-diagonal transform equals forecast correlation $1/3$ under $H_{12}=2$, $H_{11}=4$, $H_{22}=9$, so $s_{12}=6$ rather than $36$. Giacomini–Rossi stable Gaussian null, $P=400$, $m=120$, $\max|F|=1.840<3.012$, not rejected. Known sign-change break of length $200$, $m=60$, $\max|F|=3.506$, rejected, with early mean $F=-3.401$ and late mean $F=3.401$.

## Files that own the implementation

- `src/covharness/losses/contracts.py`
- `src/covharness/losses/frobenius.py`
- `src/covharness/losses/qlike.py`
- `src/covharness/losses/localization.py`
- `src/covharness/losses/robustness.py`
- `src/covharness/losses/__init__.py`
- `src/covharness/protocol/exceptions.py`
- `src/covharness/protocol/constants.py`
- `src/covharness/protocol/rolling.py`
- `src/covharness/protocol/splits.py`
- `src/covharness/protocol/preprocess.py`
- `src/covharness/protocol/__init__.py`
- `src/covharness/inference/exceptions.py`
- `src/covharness/inference/differentials.py`
- `src/covharness/inference/hac.py`
- `src/covharness/inference/dm.py`
- `src/covharness/inference/clark_west.py`
- `src/covharness/inference/diagnostics.py`
- `src/covharness/inference/size.py`
- `src/covharness/inference/bootstrap.py`
- `src/covharness/inference/spa.py`
- `src/covharness/inference/mcs.py`
- `src/covharness/inference/gw.py`
- `src/covharness/inference/mz.py`
- `src/covharness/inference/fluctuation.py`
- `src/covharness/inference/__init__.py`
- `src/covharness/features/exceptions.py`
- `src/covharness/features/origin_state.py`
- `src/covharness/features/__init__.py`
- `src/covharness/models/exceptions.py`
- `src/covharness/models/base.py`
- `src/covharness/models/random_walk.py`
- `src/covharness/models/ewma.py`
- `src/covharness/models/har_drd.py`
- `src/covharness/models/harq_drd.py`
- `src/covharness/models/ledoit_wolf.py`
- `src/covharness/models/dcc.py`
- `src/covharness/models/ridge_drd.py`
- `src/covharness/models/__init__.py`
- `src/covharness/diagnostics/epps.py` (`m_over_n` added)
- `tests/unit/test_losses.py`
- `tests/unit/test_loss_robustness.py`
- `tests/unit/test_protocol.py`
- `tests/unit/test_inference.py`
- `tests/unit/test_inference_spa_mcs.py`
- `tests/unit/test_inference_gw.py`
- `tests/unit/test_inference_mz.py`
- `tests/unit/test_inference_fluctuation.py`
- `tests/unit/test_origin_state.py`
- `tests/unit/test_models_rcov_baselines.py`
- `tests/unit/test_models_har_drd.py`
- `tests/unit/test_models_harq_drd.py`
- `tests/unit/test_models_ledoit_wolf.py`
- `tests/unit/test_models_dcc.py`
- `tests/unit/test_models_ridge_drd.py`
- `tests/unit/test_epps.py`
- `notebooks/proxy_robust_losses.ipynb`
- `notebooks/dm_hac_size.ipynb`
- `results/proxy_robust_losses.png`
- `results/dm_hac_size.png`
- `PREREGISTRATION_DRAFT.md`
- `README.md`
- `docs/PROJECT_STATE.md`
- `INITIAL_CORE_BENCHMARK_PLAN.md`
- `pyproject.toml`

## Tests run

Targeted Block 3A command on 2026-09-13 after Block 3B, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_inference.py`.

```
.....................                                                    [100%]
21 passed, 1 warning in 2.92s
```

The warning is `RuntimeWarning: overflow encountered in dot` in `test_nonfinite_hac_long_run_variance_is_rejected`. `test_hac_matches_statsmodels_intercept_only_ols` executed rather than skipped.

Targeted Block 3B command `python -m pytest -q tests/unit/test_inference_spa_mcs.py`.

```
................................                                         [100%]
32 passed in 1.08s
```

No warnings and no skips.

Full suite `python -m pytest -q` in the same environment.

```
........................................................................ [ 41%]
........................................................................ [ 83%]
.............................                                            [100%]
173 passed, 1 warning in 17.41s
```

No tests were skipped. The overflow warning is the same targeted non-finite HAC case.

`simulate_hac_size_power()` with unchanged defaults, seed 20260912, 2000 replications, $T=250$. IID naive 0.052. IID HAC DM 0.0545. AR(1) $\rho=0.6$ naive 0.3185. AR(1) HAC DM 0.1135. Mean-shift $-0.20$ two-sided HAC 0.8965. Mean-shift HAC A-better 0.946. These rates were not rerun in Block 3B. Block 3A methodology was not modified.

Targeted Block 3C command on 2026-09-13, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_inference_gw.py tests/unit/test_inference_mz.py tests/unit/test_inference_fluctuation.py`.

```
.........................................                                [100%]
41 passed in 1.06s
```

No warnings and no skips.

Full suite after Block 3C, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 33%]
........................................................................ [ 67%]
......................................................................   [100%]
214 passed, 1 warning in 17.24s
```

No tests were skipped. The overflow warning is the same Block 3A non-finite HAC case. Block 3A and 3B tests were not modified.

Targeted origin-state and Block 3C command on 2026-09-13, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_origin_state.py tests/unit/test_inference_gw.py tests/unit/test_inference_mz.py tests/unit/test_inference_fluctuation.py`.

```
.................................................................        [100%]
65 passed, 1 warning in 0.96s
```

The warning is `RuntimeWarning: overflow encountered in matmul` in `test_gw_nonfinite_omega_raises`. No tests were skipped.

Full suite after the origin-state pass, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 30%]
........................................................................ [ 60%]
........................................................................ [ 90%]
......................                                                   [100%]
238 passed, 2 warnings in 17.16s
```

No tests were skipped. The two warnings are the Block 3A non-finite HAC overflow and the Block 3C non-finite GW Omega overflow. Block 3A and 3B methodology was not modified.

Focused Block 4A-1 command on 2026-09-15, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_models_rcov_baselines.py`.

```
...........................                                              [100%]
27 passed in 0.37s
```

No tests were skipped. No warnings.

Full suite after Block 4A-1, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 27%]
........................................................................ [ 54%]
........................................................................ [ 81%]
.................................................                        [100%]
265 passed, 2 warnings in 17.07s
```

No tests were skipped. The two warnings are the same Block 3A non-finite HAC overflow and Block 3C non-finite GW Omega overflow. They are intentional overflow tests. HAR-DRD was not implemented.

Focused Block 4A-2 command on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_models_har_drd.py`.

```
.......................                                                  [100%]
23 passed in 0.47s
```

No tests were skipped. No warnings.

Full suite after Block 4A-2, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 25%]
........................................................................ [ 50%]
........................................................................ [ 75%]
........................................................................ [100%]
288 passed, 2 warnings in 17.49s
```

No tests were skipped. The two warnings are the Block 3A non-finite HAC overflow (`RuntimeWarning: overflow encountered in dot` in `test_nonfinite_hac_long_run_variance_is_rejected`) and the Block 3C non-finite GW Omega overflow (`RuntimeWarning: overflow encountered in matmul` in `test_gw_nonfinite_omega_raises`). They are intentional overflow tests. Warnings were not suppressed.

Focused Block 4A-3 command on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_models_harq_drd.py tests/unit/test_models_har_drd.py`.

```
....................................................                     [100%]
52 passed in 0.70s
```

No tests were skipped. No warnings. That count is 28 HARQ-DRD tests and 24 HAR-DRD tests.

Full suite after Block 4A-3, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 22%]
........................................................................ [ 45%]
........................................................................ [ 68%]
........................................................................ [ 90%]
.............................                                            [100%]
317 passed, 2 warnings in 17.77s
```

No tests were skipped. The two warnings are the same Block 3A non-finite HAC overflow and Block 3C non-finite GW Omega overflow. They are intentional overflow tests. Warnings were not suppressed.

Focused Block 4A-4 command on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_models_ledoit_wolf.py`.

```
......................................                                   [100%]
38 passed, 1 warning in 1.31s
```

No tests were skipped. The warning is `PendingDeprecationWarning: Importing from numpy.matlib is deprecated` from the pinned `nonlinshrink` package when LW-NL first imports that reference. It is not suppressed.

Full suite after Block 4A-4, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 20%]
........................................................................ [ 40%]
........................................................................ [ 60%]
........................................................................ [ 81%]
...................................................................      [100%]
355 passed, 3 warnings in 18.72s
```

No tests were skipped. The three warnings are the Block 3A non-finite HAC overflow, the Block 3C non-finite GW Omega overflow, and the `nonlinshrink` `numpy.matlib` pending deprecation. Warnings were not suppressed. DCC was not implemented.

Focused Block 4A-5 command on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_models_dcc.py`.

```
................................................                         [100%]
48 passed, 1 warning in 3.56s
```

No tests were skipped. The warning is `PendingDeprecationWarning: Importing from numpy.matlib is deprecated` from the pinned `nonlinshrink` package when DCC-NL first imports that reference. It is not suppressed.

Full suite after Block 4A-5, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 17%]
........................................................................ [ 35%]
........................................................................ [ 53%]
........................................................................ [ 71%]
........................................................................ [ 89%]
...........................................                              [100%]
403 passed, 3 warnings in 20.28s
```

No tests were skipped. The three warnings are the Block 3A non-finite HAC overflow, the Block 3C non-finite GW Omega overflow, and the `nonlinshrink` `numpy.matlib` pending deprecation. Warnings were not suppressed. Ridge-DRD was not begun.

Focused Block 4A-6 command on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_models_ridge_drd.py tests/unit/test_models_har_drd.py`.

```
.................................................                        [100%]
49 passed in 1.63s
```

No tests were skipped. No warnings. That count is 25 Ridge-DRD tests and 24 HAR-DRD tests.

Full suite after Block 4A-6, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 16%]
........................................................................ [ 33%]
........................................................................ [ 50%]
........................................................................ [ 67%]
........................................................................ [ 84%]
....................................................................     [100%]
428 passed, 3 warnings in 19.61s
```

No tests were skipped. The three warnings are the Block 3A non-finite HAC overflow, the Block 3C non-finite GW Omega overflow, and the `nonlinshrink` `numpy.matlib` pending deprecation. Warnings were not suppressed. XGBoost-DRD and LSTM-BEKK were not begun.

Block 4A-7 faithful daily-return LSTM-BEKK is implemented in `covharness.models.lstm_bekk` as `LSTMBEKKCovariance`. Public identity `lstm_bekk`. It consumes the same origin-window daily-return contract as DCC through `as_daily_return_history`, requires $N\ge 2$, and exposes `fit`, `forecast`, and `update`.

Paper recursion in internal percent units $x_t=100(r_t-\mu)$.

```math
H_t=CC^{\top}+C_tC_t^{\top}+a x_{t-1}x_{t-1}^{\top}+b H_{t-1}.
```

$C$ and $C_t$ are lower triangular. $a$ and $b$ are scalars. This is Wang, Liu, Tran, and Wang (2025) scalar BEKK plus an LSTM intercept, not full Engle–Kroner matrix $A,B$. Static $CC^{\top}$ with a strictly positive diagonal is the strict-PD anchor. Dynamic $C_tC_t^{\top}$ is a Gram matrix for any real lower triangle, including a negative Swish diagonal. Failed Cholesky raises. There is no jitter, nearest-PD, eigenvalue flooring, or other repair.

Returns. Fit-window mean $\mu$ is stored and frozen between parameter refits. Native residuals are $r_t-\mu$. Internal recursion uses $x_t=100(r_t-\mu)$. Public forecasts divide internal $H$ by $10000$. A constant or nonpositive centered column is rejected. Inputs are copied. Missing rows are not dropped.

Paper-specified pieces. LSTM input is the lagged internal return. Hidden size equals $N$. Depth is in $\{3,4,5\}$. Dropout is in $[0.1,0.2]$. Dynamic lower triangle with Swish on the diagonal. Gaussian NLL. RMSprop. Cholesky $\log\det$ and quadratic. Gradient clipping. Source empirical scale $\times 100$.

Project completions, not attributed to the paper. Stacked `torch.nn.LSTM` layers. Linear head $\mathbb{R}^N\to\mathbb{R}^{N(N+1)/2}$ with bias. One global Swish $\beta$ initialized at $1$. Softplus static diagonal. Softmax logits $(w,a,b)$ initialized at $(0.05,0.05,0.90)$ with $w$ unused in the recursion. $H_0=X^{\top}X/(T-1)$ when strictly PD, otherwise $\operatorname{diag}(\operatorname{diag}(S_0))$ when variances are strictly positive. $H_0$ is detached data. Static $C$ is initialized so $CC^{\top}=0.05 H_0$. Zero recurrent states on every `fit`. Full-sequence BPTT. Fixed epochs and no rolling early stopping. RMSprop $\alpha=0.99$, $\varepsilon=10^{-8}$, momentum $0$, uncentered, no weight decay. `torch.float64` on CPU. Seed list remains $(0,1,2,3,4)$ at the protocol layer. One instance receives one explicit seed.

Indexing. $x_0$ is scored under $H_0$. No presample return. Training NLL has $T$ terms. $H_T$ is formed after processing $x_{T-1}$ and is not scored in-window. `forecast()` returns $H_T/10000$ without mutation. `update` advances hidden, cell, and $H$ once using the frozen mean and frozen parameters.

Runtime pin. `torch==2.13.0` in `pyproject.toml`. This environment resolved `2.13.0+cpu`. License-Expression is the PyTorch mixed OSS set (Apache-2.0, BSD-2/3, MIT, BSL-1.0, LLVM exception). Requires-Python `>=3.10`, including 3.11. `torch.cuda.is_available()` is False here. The model never moves parameters to CUDA.

LSTM-BEKK-RC is not implemented. An input-builder method `build_lstm_input` is the unused seam. No RCov enters the faithful model.

Files created. `src/covharness/models/lstm_bekk.py` and `tests/unit/test_models_lstm_bekk.py`. Files changed. `src/covharness/models/__init__.py`, `pyproject.toml`, `README.md`, and `docs/PROJECT_STATE.md`.

Focused tests, same interpreter, `python -m pytest -q tests/unit/test_models_lstm_bekk.py`.

```
58 passed in 4.38s
```

No tests were skipped. No warnings in that focused run. Training examples are tiny by design. The $N=20$ case uses one epoch only.

Full suite after Block 4A-7, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 14%]
........................................................................ [ 29%]
........................................................................ [ 44%]
........................................................................ [ 59%]
........................................................................ [ 73%]
........................................................................ [ 88%]
.......................................................                  [100%]
487 passed, 3 warnings in 24.27s
```

No tests were skipped. The three warnings are the Block 3A non-finite HAC overflow, the Block 3C non-finite GW Omega overflow, and the `nonlinshrink` `numpy.matlib` pending deprecation. Warnings were not suppressed. XGBoost-DRD and LSTM-BEKK-RC were not begun.

Block 4A-8 implements daily origin-state APIs and a serial synthetic rolling runner. Parameter refit, daily observable-state update, and forecast formation are distinct. The 21-origin cadence is estimator refit. Forecasts remain daily.

Capability mechanism. `ModelCapabilities` in `covharness.models.capabilities` records `RollingCadence`, `FitInput`, and `UpdateObservable` as class attributes. The four cadences are `ORIGIN_MAP` (random walk), `WINDOW_STATE` (HAR-DRD, HARQ-DRD, Ridge-DRD), `RECURSIVE_STATE` (EWMA, DCC, DCC-NL, LSTM-BEKK), and `REFIT_HOLD` (LW-linear, LW-NL). The runner dispatches from those types. It does not branch on class names. Models do not inspect VALIDATION, SCREEN, or CONFIRM.

State APIs. Random walk `update` stores a copy of the new $S_t$. EWMA `update` applies one frozen-lambda recursion step. HAR, HARQ, and Ridge `update_window` rebuild origin $1/4/17$ predictors and the current-window fallback mean without re-estimating coefficients or Ridge scales. HARQ also refreshes the current per-asset RQ window. DCC, DCC-NL, and LSTM-BEKK keep their existing `update`. Standalone LW has no recursive update.

Frozen Ledoit-Wolf cadence. At `ForecastStep.refit=True` the entire estimator is recomputed from the current 250-day return window. At `refit=False` the stored covariance is returned unchanged. Sample covariance, $\rho$, and nonlinear eigenvalues are not refreshed. There is no frozen-$\rho$ plus refreshed-$S$ rule. Daily-moving-window LW remains a possible later robustness. It is not the headline equal-cadence specification.

Frozen HAR-family fallback. Between coefficient refits the insanity-filter mean is the arithmetic mean of the current origin's 250-day realized-covariance window. It is not the window from the last coefficient fit. HAR coefficients freeze. HARQ coefficients freeze. Ridge lambda, slopes, fixed effects, and predictor scales freeze.

Runner. `covharness.protocol.runner` exposes `OriginPayload`, `RollingForecastRecord`, `run_rolling_forecasts`, and `run_block_forecasts`. The payload has optional return, RCov, and RQ fields. Windows end at origin $t$. Target $t+1$ is never included. Records store a copied covariance. Losses and inference are not computed. Execution is serial. A refit origin calls `fit` once and does not update afterward. A non-refit origin advances state once. CONFIRM remains locked.

No empirical fit. No WRDS access. No market data. VALIDATION was not used for tuning. SCREEN was unused. CONFIRM stayed locked. DATA GATE unresolved.

Files created. `src/covharness/models/capabilities.py`, `src/covharness/protocol/runner.py`, `tests/unit/test_models_daily_state.py`, and `tests/unit/test_rolling_runner.py`. Files changed. `src/covharness/models/base.py`, `src/covharness/models/random_walk.py`, `src/covharness/models/ewma.py`, `src/covharness/models/har_drd.py`, `src/covharness/models/harq_drd.py`, `src/covharness/models/ridge_drd.py`, `src/covharness/models/ledoit_wolf.py`, `src/covharness/models/dcc.py`, `src/covharness/models/lstm_bekk.py`, `src/covharness/models/__init__.py`, `src/covharness/protocol/__init__.py`, `README.md`, and `docs/PROJECT_STATE.md`.

Focused daily-state command on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_models_daily_state.py`.

```
12 passed, 1 warning in 1.91s
```

No tests were skipped. The warning is the pinned `nonlinshrink` `numpy.matlib` pending deprecation. It is not suppressed.

Runner integration command, same interpreter, `python -m pytest -q tests/unit/test_rolling_runner.py`.

```
14 passed, 1 warning in 19.48s
```

No tests were skipped. The same `nonlinshrink` pending deprecation appears when LW-NL is first imported. Warnings were not suppressed.

Full suite after Block 4A-8, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 14%]
........................................................................ [ 28%]
........................................................................ [ 42%]
........................................................................ [ 56%]
........................................................................ [ 70%]
........................................................................ [ 84%]
........................................................................ [ 98%]
.........                                                                [100%]
513 passed, 3 warnings in 39.39s
```

No tests were skipped. The three warnings are the Block 3A non-finite HAC overflow, the Block 3C non-finite GW Omega overflow, and the `nonlinshrink` `numpy.matlib` pending deprecation. Warnings were not suppressed. XGBoost-DRD, LSTM-BEKK-RC, and GHAR were not begun. Final `PREREGISTRATION.md` was not created.

Block 4A-9 implements XGBoost-DRD as the nonlinear architecture-control step after Ridge-DRD. Public class `XGBoostDRDRealizedCovariance`, identity `xgboost_drd`. It inherits `RealizedCovarianceModel` and declares `WINDOW_STATE` / `REALIZED_COVARIANCE` / `REALIZED_COVARIANCE_WINDOW`. The Block 4A-8 runner dispatches from those capabilities. It does not branch on the class name.

Architecture-control definition. HAR-DRD to Ridge-DRD to XGBoost-DRD holds fixed the variance-level and raw-correlation targets, the non-overlapping $1/4/17$ information set, the lag convention, the asset and pair group structure, the dummy-free within transformation, the Ridge RMS-scaled predictor representation, the 21-origin WINDOW_STATE cadence, DRD reconstruction, and the current-origin 250-day mean repair. The linear ridge learner is replaced by a boosted-tree learner. That is the defensible statement. The step is not described as changing only one mathematical parameterization.

Pooling and preprocessing. Variance groups are assets. Correlation groups are unique pairs $i<j$. Group means $\bar y$ and $\bar X$ are computed on the $228$ training rows when $T=250$. Within residuals are $y-\bar y$ and $X-\bar X$. Dummy columns and group IDs are not used. After demeaning, each of the three predictor columns is divided by $\sqrt{\mathrm{mean}(\widetilde X_j^2)}$. An exact zero column receives scale $1$. Responses are not standardized. Scaling is imposed so that the numerical predictor representation matches Ridge exactly. It is not imposed because trees require scaling. Ridge helpers `within_transformed_arrays`, `within_predictor_scales`, and `scale_within_predictors` are reused without mathematical change.

Learners. Exactly two pooled boosters are fit per refit, $f_D\colon\mathbb{R}^3\to\mathbb{R}$ and $f_R\colon\mathbb{R}^3\to\mathbb{R}$, both with `objective="reg:squarederror"` on the within residuals. There is no $N$-model or $P$-model explosion, no eval set, and no early stopping. Origin forecasts are $\bar y_g+f(Z_{\mathrm{origin},g})$ with frozen $\bar y$, $\bar X$, and RMS scales. `update_window` rebuilds current-origin HAR predictors and the current 250-day fallback mean only.

Explicit configuration. Constructor requires `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `reg_lambda`, `reg_alpha`, and `gamma`. Frozen package constants are `booster="gbtree"`, `tree_method="hist"`, `device="cpu"`, `n_jobs=1`, `subsample=1.0`, every `colsample` value $1$, `grow_policy="depthwise"`, `max_bin=256`, `base_score=0.0`, `random_state=0`, `validate_parameters=True`, and `early_stopping_rounds=None`. Headline fitting is deterministic. The seed ensemble $(0,1,2,3,4)$ is not applied. SHAP, ALE, feature importance, and pair subsampling are not implemented.

Runtime pin. `xgboost==3.2.0` in `pyproject.toml`. This environment resolved `3.2.0`. License is Apache-2.0. Requires-Python `>=3.10`, including 3.11.16 here. Training uses CPU hist only. The manylinux wheel also installs `nvidia-nccl-cu12` on Linux. That extra is unused. `device="cpu"` is passed explicitly.

Repair. The shared `headline_repaired_forecast` contract is unchanged. Fallback is the arithmetic mean of the current origin's exact 250-day RCov window. An invalid fallback raises. There is no clipping, nearest-PD, jitter, or second repair.

No empirical fit. No WRDS access. No market data. VALIDATION was not used for tuning. SCREEN was unused. CONFIRM stayed locked. DATA GATE unresolved. The pending bounded DM calibration review remains pending. LSTM-BEKK-RC was not begun. The graph-neural specification remains unresolved. Final `PREREGISTRATION.md` was not created.

Files created. `src/covharness/models/xgboost_drd.py` and `tests/unit/test_models_xgboost_drd.py`. Files changed. `src/covharness/models/__init__.py`, `pyproject.toml`, `tests/unit/test_models_daily_state.py`, `tests/unit/test_rolling_runner.py`, `tests/unit/test_models_lstm_bekk.py` (LSTM-BEKK-RC seam now allows the implemented XGBoost export), `README.md`, and `docs/PROJECT_STATE.md`. Ridge within and scaling helpers were not moved.

Focused Block 4A-9 command on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_models_xgboost_drd.py`.

```
22 passed in 12.75s
```

No tests were skipped. No warnings in that focused run.

HAR, Ridge, daily-state, and runner regression, same interpreter, `python -m pytest -q tests/unit/test_models_har_drd.py tests/unit/test_models_ridge_drd.py tests/unit/test_models_daily_state.py tests/unit/test_rolling_runner.py`.

```
78 passed, 1 warning in 33.34s
```

No tests were skipped. The warning is the pinned `nonlinshrink` `numpy.matlib` pending deprecation. It is not suppressed.

Full suite after Block 4A-9, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 13%]
........................................................................ [ 26%]
........................................................................ [ 40%]
........................................................................ [ 53%]
........................................................................ [ 67%]
........................................................................ [ 80%]
........................................................................ [ 93%]
.................................                                        [100%]
537 passed, 3 warnings in 52.75s
```

No tests were skipped. The three warnings are the Block 3A non-finite HAC overflow, the Block 3C non-finite GW Omega overflow, and the `nonlinshrink` `numpy.matlib` pending deprecation. Warnings were not suppressed. LSTM-BEKK-RC and GHAR were not begun. Final `PREREGISTRATION.md` was not created.

Block 4A-10 wires the existing serial runner to the existing covariance-space losses on synthetic data only. No forecasting model was added. Reduced multivariate QLIKE remains `covharness.losses.qlike.reduced_qlike_loss`. Squared Frobenius remains `covharness.losses.frobenius.squared_frobenius_loss`. Both remain one-date functions of shape `(N, N)`. DM, SPA, MCS, GW, and MZ mathematics were not changed.

Evaluator path. `covharness.evaluation.score` is the adapter. Public constructors are `target_covariance_panel`, `align_forecast_records`, `score_forecast_records`, `build_loss_panel`, `summarize_loss_panel`, and `loss_differential_from_panel`. `TargetCovariancePanel` stores unique calendar keys and a copied `(T, N, N)` cube. Lookup is by timestamp, not by positional alignment with the forecast list. `AlignedForecastTarget` pairs `H_{t+1\mid t}` with `S_{t+1}`. `LossRecord` stores `model_name`, `origin`, `target`, `loss_name`, `loss_value`, `refit`, and `action`. Covariance matrices are not copied into loss records. `repaired` is omitted because `RollingForecastRecord` does not expose it.

Alignment. For every `RollingForecastRecord`, origin $t$ must be the calendar predecessor of target $t+1$, and the scoring proxy is `covariance_at(target)`. Origin-day $S_t$ is never the evaluation target. Missing target dates, duplicate `(model, target)` keys, dimension mismatch, and non-next-date origin/target pairs raise `EvaluationAlignmentError`. Existing loss-domain failures (`ForecastNotPositiveDefiniteError`, `InvalidCovarianceMatrixError`) pass through. Evaluation does not jitter, clip, or nearest-PD a forecast.

Loss panel. `build_loss_panel` returns a `(n_targets, n_models)` matrix in chronological target order. Default `date_support='error'` requires identical target dates across compared models. `date_support='intersection'` is explicit. Silent NA dropping is not implemented. Model column order follows the caller roster when supplied, otherwise first appearance. `summarize_loss_panel` returns roster-ordered count, mean, median, and sample standard deviation. It does not label a winner.

Differentials. `loss_differential_from_panel(panel, A, B)` returns $d_t=L_{A,t}-L_{B,t}$ through the existing Block 3A `loss_differential`. Negative means A has lower loss that date. A focused smoke test passes the resulting series to existing `diebold_mariano_from_losses`. SPA and MCS were not invoked in this block. DM bandwidth and dependence defaults were not changed. The pending DM calibration review remains pending.

Synthetic generator. `covharness.simulation.benchmark.synthetic_benchmark_panel` draws a seeded panel of business-day calendar, daily returns, strictly PD realized covariance, and per-asset $\mathrm{RQ}_i=(M/3)\sum_l r_{i,l}^4$. Latent covariance uses time-varying volatilities and equicorrelation. Interval returns are Gaussian given that latent matrix. The evaluation target is the noisy realized-covariance proxy, not the latent generator. The generator is an integration fixture. It is not a market simulator.

Demo. `scripts/synthetic_benchmark_demo.py` uses `build_schedule`, not a full `TemporalProtocol` allocation, because production $T-m\ge 1000$ is too long for a demonstration. Demonstration window $m=40$, $N=3$, $8$ targets, seed $20260916$. Roster is random-walk RCov, EWMA, HAR-DRD, HARQ-DRD, Ridge-DRD ($\lambda=1$), XGBoost-DRD (`n_estimators=2`, `max_depth=1`), LW-linear, LW-NL, DCC, DCC-NL, and LSTM-BEKK (`max_epochs=1`). Those hyperparameters are demonstration settings. They are not tuned. Printed tables are integration diagnostics, not empirical findings. CONFIRM remains locked. SCREEN is unused.

No empirical fit. No WRDS access. No market data. VALIDATION was not used for tuning. SCREEN was unused. CONFIRM stayed locked. DATA GATE unresolved. The pending bounded DM calibration review remains pending. LSTM-BEKK-RC was not begun. GHAR was not begun. The graph-neural specification remains unresolved. Final `PREREGISTRATION.md` was not created.

Files created. `src/covharness/evaluation/exceptions.py`, `src/covharness/evaluation/score.py`, `src/covharness/evaluation/__init__.py`, `src/covharness/simulation/benchmark.py`, `scripts/synthetic_benchmark_demo.py`, `tests/unit/test_evaluation.py`, and `tests/unit/test_synthetic_benchmark.py`. Files changed. `src/covharness/simulation/__init__.py`, `README.md`, and `docs/PROJECT_STATE.md`.

Focused evaluator command on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_evaluation.py`.

```
18 passed in 2.46s
```

No tests were skipped. No warnings in that focused run.

Synthetic end-to-end command, same interpreter, `python -m pytest -q tests/unit/test_synthetic_benchmark.py`.

```
7 passed in 2.57s
```

No tests were skipped. No warnings in that focused run. The cheap unit roster is random walk, EWMA, HAR-DRD, and Ridge-DRD. LSTM-BEKK is scored through a hand-built forecast record rather than a live unit-level fit.

Executable demo, same interpreter, `python scripts/synthetic_benchmark_demo.py`.

```
success
roster 11 models
forecast count 88
wall time 11.98s
descriptive tables 11 rows by 4 statistics for reduced QLIKE and for squared Frobenius
```

The printed numbers are integration diagnostics. They are not empirical benchmark results and are not recorded here as a ranking.

Full suite after Block 4A-10, same interpreter, `python -m pytest -q`.

```
........................................................................ [ 12%]
........................................................................ [ 25%]
........................................................................ [ 38%]
........................................................................ [ 51%]
........................................................................ [ 64%]
........................................................................ [ 76%]
........................................................................ [ 89%]
..........................................................               [100%]
562 passed, 3 warnings in 53.52s
```

No tests were skipped. The three warnings are the Block 3A non-finite HAC overflow, the Block 3C non-finite GW Omega overflow, and the `nonlinshrink` `numpy.matlib` pending deprecation. Warnings were not suppressed. LSTM-BEKK-RC and GHAR were not begun. Final `PREREGISTRATION.md` was not created.

The bounded Diebold-Mariano dependence / calibration study measures the finite-sample size of the current procedure on synthetic AR(1) loss differentials. Block 3A mathematics were not reopened. `hac_long_run_variance`, `diebold_mariano`, the automatic lag $L=\lfloor 4(T/100)^{2/9}\rfloor$, the $N(0,1)$ reference, alternatives, and diagnostic definitions were not changed. HLN, fixed-$b$, prewhitening, stationary-bootstrap DM, self-normalization, Andrews bandwidth, and Kiefer-Vogelsang critical values were not added.

Historical baseline reproduction, same interpreter, `simulate_hac_size_power()` with seed $20260912$, $B=2000$, $T=250$.

```
IID null naive=0.0520 HAC DM=0.0545
AR(1) rho=0.6 naive=0.3185 HAC DM=0.1135
IID mean shift -0.20 two-sided=0.8965 A-better=0.9460
```

Those six rates match the previously recorded experiment exactly. The historical function API is unchanged.

Expanded null grid. $d_t=\rho d_{t-1}+\varepsilon_t$ with $\varepsilon_t\sim N(0,1)$ and exact stationary start $d_0\sim N(0,1/(1-\rho^2))$. No burn-in. $\mathrm{E}[d_t]=0$. Frozen cells $\rho\in\{0.0,0.3,0.6,0.8,0.9\}$ and $T\in\{250,500,1000\}$. Lag rules $L=0$, $L_{\mathrm{auto}}$, $2L_{\mathrm{auto}}$, $4L_{\mathrm{auto}}$, with explicit lags capped at $T-1$. Automatic lags are $4$, $5$, and $6$ at $T=250$, $500$, and $1000$. $B=5000$. Seed $20260916$. Common random numbers across lag rules within each $(T,\rho)$ cell. $60$ rows. Zero HAC/DM failures.

Automatic-lag two-sided rejection at $\alpha=0.05$, with Monte Carlo $\mathrm{se}=\sqrt{\hat p(1-\hat p)/B}$ and $\hat p\pm 1.96\,\mathrm{se}$.

$T=250$. $\rho=0.0$ rate $0.0558$, interval $[0.0494,0.0622]$, close to nominal. $\rho=0.3$ rate $0.0806$, moderately oversized. $\rho=0.6$ rate $0.1228$, moderately oversized. $\rho=0.8$ rate $0.2378$ and $\rho=0.9$ rate $0.3818$, severely oversized.

$T=500$. $\rho=0.0$ rate $0.0530$, close to nominal. $\rho=0.3$ rate $0.0690$ and $\rho=0.6$ rate $0.1104$, moderately oversized. $\rho=0.8$ rate $0.2004$ and $\rho=0.9$ rate $0.3368$, severely oversized.

$T=1000$. $\rho=0.0$ rate $0.0564$, interval $[0.0500,0.0628]$, close to nominal. $\rho=0.3$ rate $0.0714$ and $\rho=0.6$ rate $0.0862$, moderately oversized. $\rho=0.8$ rate $0.1688$ and $\rho=0.9$ rate $0.2848$, severely oversized.

True long-run variance $\omega=1/(1-\rho)^2$ is simulation truth only. Mean $\hat\omega/\omega$ under the automatic lag falls from about $0.98$ at $\rho=0$ to $0.20$ at $T=250$, $\rho=0.9$. Size distortion tracks systematic LRV underestimation. $\omega$ is not used inside the DM statistic.

No bandwidth was selected from these cells. $L=0$ is a diagnostic heteroskedasticity-only case, not a headline. Larger multiplier lags reduce over-rejection on persistent nulls and can inflate size under $\rho=0$. That pattern is recorded, not optimized.

Evaluator / calibration helpers live in `covharness.inference.size`. The historical `simulate_hac_size_power` contract is unchanged. New public helpers are `stationary_ar1_paths`, `ar1_true_long_run_variance`, `calibration_hac_lag`, `simulate_dm_hac_calibration`, and the two descriptive plotters. The executable study is `scripts/dm_hac_calibration_sensitivity.py`.

No empirical losses. No WRDS. No market data. VALIDATION unused. SCREEN unused. CONFIRM locked. DATA GATE unresolved. Final `PREREGISTRATION.md` was not created.

Files created. `tests/unit/test_inference_dm_calibration.py`, `scripts/dm_hac_calibration_sensitivity.py`, `results/dm_hac_calibration_sensitivity.csv`, `results/dm_hac_calibration_size_vs_rho.png`, and `results/dm_hac_calibration_bandwidth.png`. Files changed. `src/covharness/inference/size.py`, `src/covharness/inference/__init__.py`, `notebooks/dm_hac_size.ipynb`, `README.md`, and `docs/PROJECT_STATE.md`. `src/covharness/inference/hac.py` and `src/covharness/inference/dm.py` were not modified.

Focused calibration tests on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_inference_dm_calibration.py`.

```
15 passed in 1.14s
```

No tests were skipped. No warnings in that focused run.

Calibration grid execution, same interpreter, `python scripts/dm_hac_calibration_sensitivity.py`.

```
historical baseline reproduced
expanded grid B=5000 seed=20260916 rows=60 wall_time_s=38.37
```

Full suite after the calibration study, same interpreter, `python -m pytest -q`.

```
577 passed, 3 warnings in 53.19s
```

No tests were skipped. The three warnings are the Block 3A non-finite HAC overflow, the Block 3C non-finite GW Omega overflow, and the `nonlinshrink` `numpy.matlib` pending deprecation. Warnings were not suppressed. LSTM-BEKK-RC and GHAR were not begun. Final `PREREGISTRATION.md` was not created.

The candidate pairwise companion is a recentered Politis-Romano test of the mean of $d_t=L_{A,t}-L_{B,t}$. Public function `stationary_bootstrap_mean_test` in `covharness.inference.bootstrap_mean`. Observed statistic $S=\sqrt{T}\bar d$. Bootstrap population $d_t-\bar d$. No long-run variance. No normal reference. No bootstrap-$t$. Indices come from existing `stationary_bootstrap_indices` with the expected length passed explicitly as $\ell_T=\max(2,\lfloor T^{1/3}\rfloor)$. $T=250,500,1000$ give $\ell=6,7,10$. Candidate production settings $B=5000$ and seed $20260917$, distinct from SPA/MCS seed $20260913$. $p$-values use weak inequalities and $(1+\mathrm{count})/(B+1)$, so $p\in(0,1]$. This is not Hansen SPA. Constant differentials raise `DegenerateLossDifferentialError`. `dm.py` and `hac.py` were not modified. `bootstrap.py` mathematics were not modified.

Calibration DGP reuses `stationary_ar1_paths`. Outer DGP seed $20260918$. Inner bootstrap seeds are `SeedSequence(20260918, 2, T, round(1000\rho), replication)`. Stage 1 and Stage 2 cells were frozen before any companion size result was seen.

Stage 1. All 15 cells, $M_1=200$, $B_1=499$, wall time $46.64$ seconds. Path `results/bootstrap_mean_calibration_stage1.csv`.

Stage 2 pilot. One cell $T=250$, $\rho=0.6$, $20$ outer replications, $B=1999$, $0.34$ seconds. Extrapolated full Stage 2 about $379$ seconds. Not prohibitive.

Stage 2. The pre-specified 10 cells, $M_2=1000$, $B_2=1999$, wall time $312.37$ seconds. Path `results/bootstrap_mean_calibration_stage2.csv`. Zero failures.

Stage-2 two-sided rejection at $\alpha=0.05$, versus stored automatic-NW rates from the earlier $B=5000$ NW grid.

$T=250$. $\rho=0.0$ companion $0.054$ (NW $0.0558$). $\rho=0.6$ companion $0.094$ (NW $0.1228$). $\rho=0.8$ companion $0.195$ (NW $0.2378$). $\rho=0.9$ companion $0.253$ (NW $0.3818$).

$T=500$. $\rho=0.6$ companion $0.077$ (NW $0.1104$). $\rho=0.8$ companion $0.148$ (NW $0.2004$). $\rho=0.9$ companion $0.240$ (NW $0.3368$).

$T=1000$. $\rho=0.6$ companion $0.064$ (NW $0.0862$). $\rho=0.8$ companion $0.097$ (NW $0.1688$). $\rho=0.9$ companion $0.189$ (NW $0.2848$).

These comparisons are descriptive calibration evidence. The companion is not adopted. No winner label is attached.

Files created. `src/covharness/inference/bootstrap_mean.py`, `tests/unit/test_inference_bootstrap_mean.py`, `scripts/bootstrap_mean_calibration.py`, `results/bootstrap_mean_calibration_stage1.csv`, `results/bootstrap_mean_calibration_stage2.csv`, `results/bootstrap_mean_calibration_size_vs_rho.png`, and `results/bootstrap_mean_versus_nw_size.png`. Files changed. `src/covharness/inference/__init__.py`, `README.md`, and `docs/PROJECT_STATE.md`.

Focused companion tests on 2026-09-16, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_inference_bootstrap_mean.py`.

```
16 passed in 1.06s
```

Full suite after the companion was added and before the calibration grids, same interpreter, `python -m pytest -q`.

```
593 passed, 3 warnings in 53.20s
```

Full suite after the comparison-figure helper filter, same interpreter.

```
593 passed, 3 warnings in 52.99s
```

Bounded empirical-data feasibility audit on 2026-09-20. Local inventory, Block 1 measurement-compatibility review, and an Alpaca credential check. No new WRDS extraction. No Alpaca HTTP request. No model code changed. Tests were not rerun.

Local cache. `data/cache/taq_five_stock_single_exchange_cleaned_20090213.pkl` is 697588 cleaned midquote ticks for AAPL, IBM, JPM, MSFT, and XOM on 2009-02-13, naive exchange-local timestamps. One session per name. No local multi-day RCov, daily-return, or RQ panel.

WRDS inventory reused from 2026-09-14. Sample `taqmsamp_all.cqm_20090213` was previously SELECT-successful and extracted. Production `taqmsec` / `taqm_YYYY` remain catalog-visible without verified SELECT. Three stored production `SELECT 1` attempts failed with schema permission denied. `crsp.dsf` and `crsp.stocknames` remain constant-only SELECT successes without local extracts.

Alpaca. Official historical bars and quotes documentation was read. `feed=sip` would have been required. Credentials `APCA-API-KEY-ID` / `APCA-API-SECRET-KEY` (and `ALPACA_*` aliases) are absent, so the SIP probe was skipped. Untested access is not available.

Files created. `docs/EMPIRICAL_DATA_FEASIBILITY.md`. Files changed. `docs/PROJECT_STATE.md` and `README.md` (repository-layout entry and current-status pointer). `AGENTS.md` was not modified. Final `PREREGISTRATION.md` was not created.

Bounded open-data feasibility audit on 2026-09-21. Official M6 file and official Binance public klines only. No WRDS. No yfinance. No paid API. No model code changed. Tests were not rerun. No empirical fit.

M6. `data/cache/open_data/m6/assets_m6.csv` retrieved 2026-09-21T02:38:13Z from `https://raw.githubusercontent.com/Mcompetitions/M6-methods/main/assets_m6.csv`. SHA-256 `48c67aa0976ae63de3a6d7a42228ef374d39221734a433c9391572d01e85b20a`. 590752 bytes. 26446 rows. 100 unique symbols verified from the file. Dates 2022-01-31 through 2023-02-17. Union 273 dates. All-100 intersection 201 dates (200 successive log returns). Coverage-only 5-name maximum intersection 265 dates. No subset reaches 1,250 or 1,500 common dates. Daily-return models could consume the panel under the present API. RCov and HARQ models could not. Daily outer products are not the high-frequency RCov target.

Binance. Official S3 listings for 12 USDT pairs × {1m, 5m}. Interior missing months 0. Coverage-only 5-name 1m/5m common listed span 2018-04 through 2026-08 (3075 days). Coverage-only 10-name span 2019-01 through 2026-08 (2800 days). Sample 2025-02 1m and 5m for BTCUSDT, ETHUSDT, and BNBUSDT, plus BTCUSDT 5m 2024-12 control. All checksums verified. 2025-02 timestamps are microseconds and every UTC day is complete (1440 1m bars, 288 5m bars). 2024-12 timestamps are milliseconds. Bytes downloaded in the listing-plus-sample script 8640819. Bulk download was not started. Usable 1,500-day coverage remains unverified.

Files created. `docs/OPEN_DATA_FEASIBILITY.md` and `results/open_data_feasibility.json`. Files changed. `docs/PROJECT_STATE.md`. `README.md` was not modified. `AGENTS.md` was not modified. Final `PREREGISTRATION.md` was not created.

Bounded Binance five-asset 1m coverage audit on 2026-09-21. Official `data.binance.vision` monthly ZIPs only. Coverage-audit universe BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, ADAUSDT. Months 2022-01 through 2026-08. 280 ZIPs requested, 280 SHA-256 verified, 0 absent, 0 mismatches. Compressed 512308203 bytes. Parsed from ZIP without retaining CSVs. 12268400 rows. Download 90.672 s. Parse and audit 57.422 s. No model code changed. Tests were not rerun. No empirical fit. No RCov or RQ production.

Each name has 1,703 complete UTC days of 1,440 bars. Five-way intersection 1,703 of 1,704 requested dates (2022-01-01 through 2026-08-31). Sole exclusion 2023-03-24, 80 missing minutes on every name, not forward-filled. Longest common complete streak 1,256 dates from 2023-03-25 through 2026-08-31. Timestamp units milliseconds through 2024-12 and microseconds from 2025-01, with no documented-cut anomalies. Synchronized closes on common complete dates have identical 1,440-minute support and finite within-day log returns. Production return convention was not chosen.

preferred_full_design (≥1500) yes. minimum_confirmatory_design (≥1250) yes. `common_complete_dates - 250 = 1453`. No VALIDATION/SCREEN/CONFIRM split was allocated. Binance empirical DATA GATE not fully passed.

Files created. `results/binance_five_asset_calendar_audit.json` and `results/binance_five_asset_calendar_complete_dates.csv.gz`. Files changed. `docs/OPEN_DATA_FEASIBILITY.md` and `docs/PROJECT_STATE.md`. `README.md` was not modified. `AGENTS.md` was not modified. Final `PREREGISTRATION.md` was not created.

Binance measurement-specification freeze on 2026-09-21. No model fit. No RCov/RQ production. Tests were not rerun. Official 2023-03 monthly 1m ZIPs were re-read only for 2023-03-24 missing-minute forensics. All five names miss 12:40–13:59 UTC (80 contiguous minutes). 23:59 exists. Announced Binance spot resume 14:00 UTC matches the first post-hole bar. Recommended treatment is exclusion of that UTC day as origin and target, retaining 23:59 as the 2023-03-25 anchor. Packed versus calendar one-day lags remain unresolved. Complete Binance empirical DATA GATE has not passed.

Files created. `docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md` and `results/binance_20230324_halt_forensics.json`. Files changed. `docs/OPEN_DATA_FEASIBILITY.md` and `docs/PROJECT_STATE.md`. `README.md` was not modified because the TAQ public measurement definition is unchanged and the Binance arrays are not yet implemented. `AGENTS.md` was not modified. Final `PREREGISTRATION.md` was not created.

Binance segmented calendar freeze and production panel on 2026-09-21. No model fit. No SCREEN. No CONFIRM. Generic `TemporalProtocol` was not modified. TAQ, loss, and inference mathematics were not changed. Official November and December 2021 1m ZIP+CHECKSUM files for BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, and ADAUSDT were downloaded from `https://data.binance.vision` and SHA-256 verified. Existing 2022-01 through 2026-08 archives were reused.

Frozen UTC segments, independently counted before construction. HISTORY_A 2021-11-09 through 2022-07-16 (250). VALIDATION 2022-07-17 through 2023-03-23 (250). Halt 2023-03-24 excluded. HISTORY_B 2023-03-25 through 2023-11-29 (250). SCREEN 2023-11-30 through 2025-04-12 (500). CONFIRM 2025-04-13 through 2026-08-25 (500). Unused tail 2026-08-26 through 2026-08-31. Packed complete-day lags rejected so that no HAR/EWMA/DCC/LSTM state crosses the halt.

Production artifact `data/processed/binance_five_asset_panel.npz`, SHA-256 `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`, 359563 bytes. Sidecar `data/processed/binance_five_asset_panel.json`, SHA-256 `38ed1d4843ea6138d7f87fc0405b668b1ee2fd7403d4cddfba0ca49160737394`. Built $T=1750$, $N=5$. RCov rank 5 on every date. PSD failures 0. Symmetry failures 0. Nonfinite entries 0. Invalid production dates none. 2023-03-25 uses the valid 2023-03-24 23:59 anchor. Raw minute bars are not stored. Raw ZIPs remain gitignored.

Focused tests on 2026-09-21, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_binance_measurement.py`.

```
13 passed in 0.34s
```

Files created. `src/covharness/data/binance_calendar.py`, `src/covharness/data/binance_klines.py`, `src/covharness/realized/binance_panel.py`, `tests/unit/test_binance_measurement.py`, `docs/BINANCE_OPEN_DATA_PANEL.md`, `data/processed/binance_five_asset_panel.json`, and `results/binance_five_asset_panel_validation.json`. Files changed. `docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md`, `docs/OPEN_DATA_FEASIBILITY.md`, `docs/PROJECT_STATE.md`, and `README.md` (repository-structure comments only). `AGENTS.md` was not modified. `PREREGISTRATION_DRAFT.md` was not modified. Final `PREREGISTRATION.md` was not created.

Binance first-stage protocol freeze on 2026-09-21. No model fit. No VALIDATION execution. No SCREEN. No CONFIRM. Generic `TemporalProtocol` was not modified. TAQ, loss, and inference mathematics were not changed.

Branch-specific schedule `covharness.protocol.binance.BinanceSegmentedProtocol`. VALIDATION uses HISTORY_A plus VALIDATION only. SCREEN and CONFIRM use HISTORY_B plus SCREEN plus CONFIRM only. Packed lags across 2023-03-24 are rejected. First origin in each evaluation block is a refit. Cadence 21 resets per block. VALIDATION 250 targets, 12 refits. SCREEN 500 targets, 24 refits. CONFIRM 500 targets, 24 refits, first window the last 250 SCREEN dates. Default CONFIRM access raises `ConfirmLockedError`.

Core roster RW, EWMA, HAR-DRD, HARQ-DRD, LW-linear, LW-NL, DCC, DCC-NL, Ridge-DRD, XGBoost-DRD, LSTM-BEKK. EWMA, Ridge, XGBoost, and LSTM each have 20 frozen candidates. The other seven families have one frozen definition each. LSTM seeds $0,1,2,3,4$ live at the experiment layer. The candidate forecast is the equal-weight seed ensemble. Primary selection is mean VALIDATION reduced QLIKE on complete 250-target support. Exact ties use the lexicographically smaller configuration ID. Incomplete candidates are invalid rather than shortened.

Configuration artifact `configs/binance_open_data_core.yaml`, SHA-256 `8d63eacacf13a876651f9c4eb5399a8527d8458278974ff069dd0701dd5cbf07`. LSTM dropout constructor bound expanded from $[0.1,0.2]$ to $[0,0.2]$ so that frozen A1 (`dropout=0`) is constructible. The BEKK recursion is unchanged.

Focused tests on 2026-09-21, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_binance_protocol.py`.

```
19 passed in 3.99s
```

Full suite, same interpreter, `python -m pytest -q`.

```
625 passed, 3 warnings in 55.03s
```

No tests were skipped. The three warnings are the Block 3A non-finite HAC overflow, the Block 3C non-finite GW Omega overflow, and the `nonlinshrink` `numpy.matlib` pending deprecation. Warnings were not suppressed. No forecasting model was fit on market data.

Files created. `src/covharness/protocol/binance.py`, `tests/unit/test_binance_protocol.py`, `configs/binance_open_data_core.yaml`, and `docs/BINANCE_OPEN_DATA_PROTOCOL.md`. Files changed. `src/covharness/protocol/__init__.py`, `src/covharness/models/lstm_bekk.py`, `tests/unit/test_models_lstm_bekk.py`, `docs/PROJECT_STATE.md`, `docs/BINANCE_OPEN_DATA_PANEL.md`, and `README.md` (repository-structure entry for `configs/`). `AGENTS.md` was not modified. `PREREGISTRATION_DRAFT.md` was not modified. Final `PREREGISTRATION.md` was not created.

Binance open-data VALIDATION fitting on 2026-09-21. SCREEN was not requested. CONFIRM remained locked. No Diebold–Mariano, SPA, MCS, Giacomini–White, Mincer–Zarnowitz, Giacomini–Rossi, or portfolio evaluation was run. Frozen grids were not retuned. Model mathematics were not changed. `configs/binance_open_data_core.yaml` was not rewritten with results.

Preflight hashes matched the freeze. Production panel SHA-256 `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`. Configuration SHA-256 `8d63eacacf13a876651f9c4eb5399a8527d8458278974ff069dd0701dd5cbf07`. Asset order BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, ADAUSDT. VALIDATION schedule 250 targets, first origin 2022-07-16, last target 2023-03-23, halt 2023-03-24 absent. Maximum model-observable date 2023-03-22. Maximum scoring target 2023-03-23. SCREEN and CONFIRM rows in the source NPZ were omitted from the experiment view.

Orchestrator `src/covharness/protocol/binance_validation.py` with CLI `scripts/run_binance_validation.py`. Candidates are read from the frozen YAML. Checkpoints are written after each deterministic candidate and each LSTM seed. Partial checkpoints are not reused as complete. LSTM candidate scores use the equal-weight mean of five seed covariance forecasts, then reduced QLIKE. Seed QLIKE averages are not used for selection.

Focused tests on 2026-09-21, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_binance_validation.py`.

```
14 passed in 2.87s
```

Full suite before the market-data run, same interpreter, `python -m pytest -q`.

```
639 passed, 3 warnings in 57.58s
```

Empirical command from the repository root.

```
PYTHONUNBUFFERED=1 /local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -u scripts/run_binance_validation.py
```

Wall runtime 31736.848 s. Manifest `runtime_seconds_total` 31730.483 s. Family runtimes in seconds. RW 0.166. HAR-DRD 4.173. HARQ-DRD 4.198. LW-linear 0.095. LW-NL 0.101. DCC 5.626. DCC-NL 4.866. EWMA 3.519. Ridge-DRD 84.122. XGBoost-DRD 113.333. LSTM-BEKK 31502.872.

Completion. Seven fixed configurations, 20 EWMA, 20 Ridge-DRD, 20 XGBoost-DRD, 100 LSTM seeds, and 20 LSTM ensembles all have complete 250-target primary-loss support. Failures none. Repair and fallback counts zero on every candidate. QLIKE failures 0. Frobenius failures 0. Nonfinite forecasts 0. Fit failures 0.

Within-family VALIDATION selection, primary score mean reduced QLIKE on 250 targets, exact ties by lexicographically smaller ID. These are development-set results.

EWMA `EWMA01`, mean QLIKE $-32.73154867789014$, mean squared Frobenius $4.263040744795783\times 10^{-5}$.

Ridge-DRD `RIDGE01`, mean QLIKE $-32.84554386201314$, mean squared Frobenius $3.0331065954931998\times 10^{-5}$. `RIDGE01` is $\lambda=0$ and therefore nests HAR-DRD. The two scores agree to the recorded precision.

XGBoost-DRD `XGB08`, mean QLIKE $-32.78193966002438$, mean squared Frobenius $3.1641912477168996\times 10^{-5}$.

LSTM-BEKK `LSTM19` ensemble, mean QLIKE $-32.285398774051586$, mean squared Frobenius $4.612462237757348\times 10^{-5}$. Seed-level descriptive QLIKE on that configuration ranges from $-32.295177791360466$ (seed 4) to $-32.034444970669604$ (seed 0). The best seed was not selected.

Fixed families recorded without search. RW `RW01`. HAR-DRD `HARDRD01`. HARQ-DRD `HARQDRD01`. LW-linear `LWLIN01`. LW-NL `LWNL01`. DCC `DCC01`. DCC-NL `DCCNL01`.

Result artifacts.

`results/binance_validation_manifest.json` SHA-256 `6255533b64fefe371efb175f3dae20fb81b7a39cdb4f8f8a1441afeeb6733d3c`.

`results/binance_validation_selection.json` SHA-256 `f6d9830302b03afa2f3f45cd15576d70fbae4b96482af5c71d935822bc132f4b`.

`results/binance_validation_failures.json` SHA-256 `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`.

`results/binance_validation_candidate_summary.csv` SHA-256 `cf525c217379496a38ccc4e4f0f9a2e6dd8bbca02d6cf898ab3b1fbef3866cb0`.

`results/binance_validation_development_table.csv` SHA-256 `04822c1a1cbea425a80f5e555ca210bf3a365210d10d6d29b7bc636c3f398eab`.

`results/binance_validation_diagnostics.json` SHA-256 `cf21d4d11c03ca300517e35deea495f868255fd0fcc11c1a08871a37365398c6`.

`results/binance_validation_forecasts.npz` SHA-256 `564afb691b138af0dba44fc1009f1e30c8f1887670e02254773957d9a77a9773`.

Files created. `src/covharness/protocol/binance_validation.py`, `tests/unit/test_binance_validation.py`, `scripts/run_binance_validation.py`, and the result artifacts listed above. Files changed. `docs/PROJECT_STATE.md`, `docs/BINANCE_OPEN_DATA_PROTOCOL.md`, `docs/BINANCE_OPEN_DATA_PANEL.md`, and `README.md` (factual VALIDATION status and scripts layout only). `AGENTS.md` was not modified. `PREREGISTRATION_DRAFT.md` was not modified. Final `PREREGISTRATION.md` was not created. The frozen YAML was not rewritten.

Binance open-data SCREEN on 2026-09-21. VALIDATION grids were not rerun. Selected IDs were not replaced. CONFIRM remained locked. No Diebold–Mariano, Clark–West, Giacomini–White, Mincer–Zarnowitz, Giacomini–Rossi, or GMV significance test was run. Model mathematics were not changed. `configs/binance_open_data_core.yaml` was not rewritten. `configs/binance_open_data_screen.yaml` was not rewritten with SCREEN results.

Preflight hashes matched the freeze. Production panel SHA-256 `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`. Core configuration SHA-256 `8d63eacacf13a876651f9c4eb5399a8527d8458278974ff069dd0701dd5cbf07`. VALIDATION selection SHA-256 `f6d9830302b03afa2f3f45cd15576d70fbae4b96482af5c71d935822bc132f4b`. SCREEN configuration SHA-256 `b07011a798c36fc25e117b69d71f6c8eb5eb1ed756e97c39c19e29542af6203c`. Selected IDs EWMA01, RIDGE01, XGB08, and LSTM19, each with complete 250-target VALIDATION support. LSTM19 seeds 0, 1, 2, 3, and 4 completed on VALIDATION. Primary VALIDATION criterion was mean reduced QLIKE. Frobenius did not select or break ties. EWMA01 is the smallest-lambda grid point and was retained. RIDGE01 has $\lambda=0$ and nests HAR-DRD under the existing implementation and was retained.

SCREEN view is HISTORY_B plus SCREEN only. CONFIRM rows were omitted. Schedule 500 targets from 2023-11-30 through 2025-04-12, first origin 2023-11-29, first window 2023-03-25 through 2023-11-29, 24 refits at positions $0,21,\ldots,483$. Maximum model-observable date 2025-04-11. Maximum scoring target 2025-04-12. No observation dated 2025-04-13 or later entered a fit, update, feature, or SCREEN loss. Packed lags across 2023-03-24 remain rejected.

Orchestrator `src/covharness/protocol/binance_screen.py` with CLI `scripts/run_binance_screen.py`. Eleven frozen representatives. LSTM19 uses seeds $0,1,2,3,4$ with the equal-weight covariance ensemble as the LSTM column. The best seed is not selected. Checkpoints are written after each deterministic representative and each LSTM seed. Incomplete checkpoints are not reused as complete.

Focused tests on 2026-09-21, using `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_binance_screen.py`.

```
18 passed in 3.52s
```

Full suite after the incomplete-family orchestrator fix and before the empirical resume, same interpreter, `python -m pytest -q`.

```
657 passed, 3 warnings in 57.00s
```

Empirical command from the repository root.

```
PYTHONUNBUFFERED=1 /local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -u scripts/run_binance_screen.py
```

Family forecast runtimes from checkpoints, in seconds. RW 0.351. EWMA 0.374. HAR-DRD 8.593. HARQ-DRD 8.560. LW-linear 0.215. LW-NL 0.218. DCC 0.860. DCC-NL 0.526. Ridge-DRD 8.663. XGBoost-DRD 20.010. LSTM-BEKK 4919.511. Sequential family total about 4968 s. Artifact-writing resume after checkpoints, including SPA and MCS, had manifest `runtime_seconds_total` 2.977 s.

Completion. Nine of eleven representatives have complete 500-target support. DCC and DCC-NL failed at origin 2023-12-20, the second SCREEN refit, because ZeroMean GARCH(1,1) for asset 1 returned $\omega\approx 4.97\times 10^{-12}$, $a=0$, $b=1$, which violates the project IGARCH contract $a+b<1$. Those failures were preserved. No second VALIDATION configuration was substituted. SPA and MCS used the nine complete columns. Loss panels remain $500\times 11$ with NaN columns for the two failures.

Repair counts. HARQ-DRD recorded 2 estimation-window-mean fallbacks. All other complete representatives recorded 0. Evaluation-time repair was not used. QLIKE failures 0 on complete models. Nonfinite forecasts 0 on complete models.

LSTM19 seeds 0 through 4 all completed. Seed-level mean QLIKE ranged from $-31.351885218861142$ to $-31.211140652888883$. Ensemble mean QLIKE $-31.408752391639815$. The ensemble, not a seed, is the LSTM SCREEN column.

Primary SCREEN ranking channel is reduced QLIKE. Squared Frobenius is the complementary robustness channel. The two losses were not averaged.

Hansen SPA versus HAR-DRD, $B=5000$, seed $20260913$, $\ell=\max(2,\lfloor T^{1/3}\rfloor)=7$, consistent recentering primary. RIDGE01 is an exact zero benchmark differential on both losses and is labeled `exact_benchmark_tie`. It remains in descriptive tables and MCS. It is excluded only from SPA studentization. Nonzero constant differentials did not occur.

SPA reduced QLIKE. Universe 9. Statistic 3.0010295525346193. Consistent p-value 0.0016. Exact ties Ridge-DRD. Studentized HARQ-DRD 3.0010295525346193.

SPA squared Frobenius. Universe 9. Statistic 0.01017192672629011. Consistent p-value 0.8792. Exact ties Ridge-DRD.

Primary QLIKE MCS $(T_R,e_R)$, $\alpha=0.10$, $B=5000$, seed $20260913$. Membership at 0.10 is EWMA and HARQ-DRD. p-values. RW 0.0002. EWMA 0.3154. HAR-DRD 0.0124. HARQ-DRD 1.0. LW-linear 0.0002. LW-NL 0.0004. Ridge-DRD 0.0124. XGBoost-DRD 0.0124. LSTM-BEKK 0.001. Companion $(T_{\max},e_{\max})$ at 0.10 retains EWMA, HAR-DRD, HARQ-DRD, Ridge-DRD, and XGBoost-DRD.

Primary Frobenius MCS at $\alpha=0.10$. Membership is RW, EWMA, HAR-DRD, HARQ-DRD, Ridge-DRD, XGBoost-DRD, and LSTM-BEKK. p-values. RW 0.1084. EWMA 0.286. HAR-DRD 0.9932. HARQ-DRD 1.0. LW-linear 0.011. LW-NL 0.0112. Ridge-DRD 0.9932. XGBoost-DRD 0.46. LSTM-BEKK 0.1244.

Stage-2 paradigm membership was frozen before ranks. Econometric eligible set RW, EWMA, HAR-DRD, HARQ-DRD, LW-linear, LW-NL, DCC, DCC-NL. Shallow-ML Ridge-DRD and XGBoost-DRD participate in SPA, MCS, and tables and cannot occupy the econometric headline slot. Deep-learning eligible set is LSTM-BEKK only, so LSTM19 is the DL finalist by construction.

Econometric finalist rule uses primary QLIKE only. Four chronological 125-target blocks. Rank eight econometric models within each complete block, lower loss better, exact block-mean ties receive average ranks. Select lowest median of the four block ranks, then lowest mean of those ranks, then lexicographically smaller family ID. Full-SCREEN mean QLIKE is robustness only. Frobenius does not select. MCS membership does not replace a finalist.

DCC and DCC-NL have no complete SCREEN losses, so the econometric ranking used the six complete econometric families. Block ranks for HARQ-DRD are 2, 1, 2, 2. Median rank 2.0. Mean rank 1.75. EWMA also has median rank 2.0 and mean rank 2.0. HARQ-DRD is the econometric finalist by the mean-rank tie-break. Full-SCREEN mean-QLIKE robustness leader among complete econometric models is also HARQ-DRD. The headline was not replaced by that robustness check.

Finalists. DL LSTM-BEKK configuration LSTM19. Econometric HARQ-DRD configuration HARQDRD01. Primary QLIKE MCS at 0.10 contains HARQ-DRD (p-value 1.0) and does not contain LSTM-BEKK (p-value 0.001). Primary Frobenius MCS at 0.10 contains both (HARQ-DRD p-value 1.0, LSTM-BEKK p-value 0.1244). MCS did not replace either finalist.

These SCREEN numbers are not a confirmatory ranking and are not a superiority claim. CONFIRM remains locked.

Result artifacts.

`results/binance_screen_manifest.json` SHA-256 `dc54b52b245279f1b36a69d16ae1d3f88f8424dc6e1c4078124fe1755d12ea36`.

`results/binance_screen_model_summary.csv` SHA-256 `eed72b6ce306bcfbdb34d36cfd51e0ee035c01e4f077d17b1b4592c2ceef8354`.

`results/binance_screen_spa.json` SHA-256 `992dd16cb0d9cdf3d2511b39c093c1f1daf4d0062fc9852c74c6f0fc41e46420`.

`results/binance_screen_mcs_qlike.json` SHA-256 `234f195c5c74d3b1b42de921bd545895e9ac37cf7d39ede59c2f9a7ade59cefb`.

`results/binance_screen_mcs_frobenius.json` SHA-256 `ef1e2fd798b0d3ec82a23033100e9b23bbc03e2a8b618aecb8e1b792f820735d`.

`results/binance_screen_subblock_ranks.csv` SHA-256 `fb2decc2c158171ab22d8ac2df23c53096fd5d3c6e9afedf634cf96338578514`.

`results/binance_screen_finalists.json` SHA-256 `cefe5b389878552d3fc186a0137065dfbd1a71b4cd6cd131cde376c8ba3cc5ca`.

`results/binance_screen_failures.json` SHA-256 `ef62dfb8ffd16f010ded4ab63cef410a4de09cccb025da91acfa1f2ff8d9687a`.

`results/binance_screen_losses_qlike.npz` SHA-256 `3fa68921871277b52e60412bae9fb23f31e7b26e39bec446d1051f7f6e8af61d`.

`results/binance_screen_losses_frobenius.npz` SHA-256 `63a4de10db499b5527e5f1c276c17460481d85a7c3699a7a6226a97819729713`.

`results/binance_screen_forecasts.npz` SHA-256 `118f9eb4f5cfc8f2ffa1b1e73128358725aa179d91ebea8149c1f997a89e6c94`.

`results/binance_screen_diagnostics.json` SHA-256 `a77537bccc7a40a54b33ff41dbdba75f2cdedaab659874463aab6665bf2631e3`.

Files created. `src/covharness/protocol/binance_screen.py`, `tests/unit/test_binance_screen.py`, `scripts/run_binance_screen.py`, `configs/binance_open_data_screen.yaml`, and the result artifacts listed above. Files changed. `docs/PROJECT_STATE.md`, `docs/BINANCE_OPEN_DATA_PROTOCOL.md`, `docs/BINANCE_OPEN_DATA_PANEL.md`, and `README.md` (factual SCREEN status and layout comments only). `AGENTS.md` was not modified. `PREREGISTRATION_DRAFT.md` was not modified. Final `PREREGISTRATION.md` was not created. The SCREEN YAML was not rewritten with results.

Binance SCREEN descriptive interpretation on 2026-09-21. No refit. No SPA or MCS rerun. No CONFIRM access. Mean loss differentials were computed from the saved 500-by-11 loss panels. HAR-DRD and Ridge-DRD series are identical. HARQ-DRD repair flags in the existing SCREEN checkpoint mark targets 2024-12-10 and 2025-03-03. Artifacts `results/binance_screen_interpretation.md` and `results/binance_screen_effect_sizes.csv`. Cumulative-differential figures `results/binance_screen_cumulative_qlike_harq_minus_lstm.png` and `results/binance_screen_cumulative_frobenius_harq_minus_lstm.png`. These files are SCREEN descriptive analysis. They are not confirmatory inference and do not change the frozen finalists.

## Methodological decisions already in code

- Reduced QLIKE is the primary ranking loss. Full Stein is the SPD-proxy form.
- Squared Frobenius is the complementary robust loss. Ordinary unsquared Frobenius is a labeled non-robust contrast only.
- Localization diagnostics are descriptive. They are not ranking losses.
- Forecasts that fail PD are exposed, not repaired.
- `M/N` is a proxy-quality diagnostic, not a gate on reduced QLIKE.
- Protocol intervals are half-open on the trading-date index.
- VALIDATION 250 is a target. SCREEN and CONFIRM 500 are committed minima. Zero VALIDATION makes data-driven tuning unavailable.
- Rolling $m=250$ is the estimation window. The 21-origin cadence is parameter and estimator refit, not forecast cadence.
- Parameter refit, daily observable-state update, and forecast formation are distinct.
- CONFIRM is locked unless `unlock_confirm=True` is passed.
- The Binance branch uses a segmented protocol with a hard 2023-03-24 break. Generic `TemporalProtocol` remains the single-HISTORY allocator.
- Stochastic seeds and the configuration budget are protocol metadata, not later model-local choices.
- Statistical evaluation is open-to-close. Later economic GMV includes overnight.
- Pairwise tests use $d_t=L_{A,t}-L_{B,t}$ and Bartlett / Newey-West HAC. Harvey-Leybourne-Newbold is not applied.
- The current automatic lag remains $L=\lfloor 4(T/100)^{2/9}\rfloor$. The synthetic sensitivity study did not adopt a new default.
- A Monte Carlo rejection rate and Monte Carlo interval describe the implemented procedure. They are not empirical forecast-comparison $p$-values.
- A candidate recentered stationary-bootstrap pairwise mean test exists as a companion API. It is not the confirmatory default. It does not use an LRV.
- Clark-West is opt-in, nested, and scalar squared-error only.
- SPA and MCS are applied separately to each loss/proxy channel.
- SPA uses Hansen's benchmark-minus-alternative differential and the consistent p-value as the headline.
- SPA long-run variances use the stationary-bootstrap geometric kernel, not Bartlett HAC.
- Stationary-bootstrap block length is $\max(2,\lfloor T^{1/3}\rfloor)$ with $B=5000$ and seed $20260913$.
- Primary MCS is $(T_R,e_R)$ at SCREEN $\alpha=0.10$. $(T_{\max},e_{\max})$ is a companion.
- One-step GW uses undemeaned outer-product $\widehat{\Omega}=Z^{\top}Z/T$ and $\chi^2_q$. Constant nonzero $d_t$ is valid when $h_t$ has full rank.
- GW market-state instruments are origin-day and unstandardized.
- GW measurement/stress instruments are origin-day $\log\mathrm{RQ}_{\mathrm{agg}}$ and the 1 percent BNS equal-weight market jump indicator. They are not standardized.
- GW Bonferroni family size is exactly 2. The two specifications are not stacked.
- Primary MZ is Patton–Sheppard pooled vech with daily-score Bartlett HAC sandwich. Diagonal and off-diagonal subsets are descriptive.
- Exact-fit MZ returns `covariance=None` rather than a singular sandwich. Wald 0 / inf semantics are unchanged.
- Augmented MZ uses common project-specific gamma coefficients on origin-day market state. It is a separate Bonferroni family of the two finalists.
- Approximate PS21 uses $s_{ij,t}=\sqrt{H_{ii,t}H_{jj,t}}$ and the label `approximate_ps21`. It is not exact GLS.
- Giacomini–Rossi uses $\mu=0.30$, even $m=2\lfloor 0.30P/2\rfloor$, two-sided $k=3.012$, and a global uncentered LRV. The One-Time Reversal test is not implemented.
- The BNS jump state is a common-market portfolio test. It is not an any-constituent-jumped indicator.
- Random-walk realized covariance is $H_{t+1\mid t}=S_t$ with no repair.
- EWMA is the recursion $H_0=S_0$, $H_j=\lambda H_{j-1}+(1-\lambda)S_j$ on realized covariance. $\lambda$ is explicit. The 20-point VALIDATION grid is not frozen.
- Models consume a caller-supplied origin window. They do not inspect protocol block labels.
- Headline HAR-DRD uses Zhang-style non-overlapping HAR lags of widths $1$, $4$, and $17$ on the DRD split.
- Variance HAR has asset-specific intercepts and three shared scalar slopes. Correlation HAR has pair-specific intercepts and three shared scalar slopes.
- HAR-DRD is estimated by the within transformation. Dummy intercept columns are not constructed.
- Headline HAR-DRD is in levels. Log-variance, Fisher, ridge, graph, and HARQ terms are not used.
- Unique pairs follow the existing strict upper-triangle order $i<j$.
- The only headline HAR-DRD, HARQ-DRD, Ridge-DRD, and XGBoost-DRD repair is replacement by the arithmetic mean of the current origin's 250-day realized-covariance window. Between coefficient refits that fallback mean moves. Coefficients, Ridge scales, Ridge lambda, XGBoost group means, XGBoost scales, and XGBoost boosters do not.
- Standalone LW-linear and LW-NL follow the common 21-origin estimator-refit cadence. Between refits the stored covariance is held. Daily-moving-window LW is a later robustness, not the headline.
- Headline HARQ-DRD adds one daily per-asset quarticity interaction on variances and leaves the HAR-DRD correlation map unchanged.
- HARQ RQ is the per-asset series $\mathrm{RQ}_i=(M/3)\sum_l r_{i,l}^4$ supplied as a $(T,N)$ window. It is not the GW aggregate $\log\mathrm{RQ}_{\mathrm{agg}}$.
- HARQ imposes no sign constraint on $\phi_{Q,d}$. Zero RQ is admissible.
- HARQ uses the same BPQ estimation-window-mean insanity filter as HAR-DRD.
- Standalone LW models consume daily returns, not realized-covariance histories.
- Returns are demeaned in-window. $S=Y^{\top}Y/(T-1)$ is shared by LW-linear and LW-NL.
- Headline LW-linear is Ledoit-Wolf 2004b $\mu I$ shrinkage. Honey / equicorrelation is not used.
- Headline LW-NL is the pinned 2020 analytical estimator via `nonlinshrink==0.7`. QuEST and QIS are not used.
- Both LW estimators retain sample eigenvectors. Linear uses one affine eigenvalue map. Nonlinear uses eigenvalue-specific shrinkage.
- Standalone LW is distinct from DCC-NL targeting. DCC-NL uses the same analytical estimator on standardized residuals with $k=0$ and divisor $T$.
- The empirical daily-return convention remains unresolved until DATA GATE integration.
- Headline DCC identities are original Engle (2002) DCC, not cDCC.
- Stage-one GARCH is ZeroMean Gaussian GARCH(1,1) on in-window demeaned returns, fit by pinned `arch==8.0.0`.
- The GARCH backcast is the sample second moment of centered residuals with divisor $T$, passed as an explicit `arch` backcast.
- The fit-window mean is frozen between 21-origin parameter refits. Daily `update` does not re-estimate parameters.
- Plain DCC targeting is $S_{\mathrm{std}}^{\top}S_{\mathrm{std}}/T$ then diagonal renormalization. Strict PD is checked by Cholesky. $N>T$ is rejected as rank-deficient. $N=T$ is not categorically rejected.
- Headline second-stage estimation is all-pairs composite Gaussian QMLE over $i<j$, optimized by deterministic SLSQP from $(0.05,0.90)$. An accepted fit requires $\alpha+\beta<1$.
- DCC-NL changes only the intercept $C$. Final $H$ is not nonlinearly shrunk.
- Headline Ridge-DRD is the regularized HAR-DRD control. Same level variance, raw correlation, $1/4/17$ features, pooling, and repair.
- Ridge predictors are scaled only after within demeaning. Responses are not scaled. Intercepts are unpenalized.
- Ridge objective is SSE plus $\lambda\|\gamma\|_2^2$ on scaled slopes. One $\lambda\ge 0$ is shared by variance and correlation. $\lambda=0$ nests HAR-DRD.
- The Ridge 20-point empirical $\lambda$ grid is not frozen.
- Headline XGBoost-DRD is the nonlinear architecture-control step after Ridge. Same targets, $1/4/17$ features, dummy-free within transform, Ridge RMS-scaled predictors, WINDOW_STATE cadence, and repair. The linear ridge learner is replaced by two pooled squared-error boosters.
- XGBoost scaling matches Ridge so that the numerical predictor representation is identical. It is not imposed because trees require scaling. Responses are not standardized. Group IDs are not passed.
- Exactly two boosters are fit per refit. There is no $N$-model or $P$-model explosion and no five-seed ensemble. Headline configuration is deterministic CPU hist with `xgboost==3.2.0`.
- The XGBoost 20-configuration VALIDATION grid is not frozen. Constructor hyperparameters are explicit and untuned.
- Headline LSTM-BEKK is daily returns only. Hidden size equals $N$. Depth is a stacked LSTM in $\{3,4,5\}$.
- LSTM-BEKK public forecasts are native-unit matrices $H/10000$ after internal percent-scale training.
- LSTM $(a,b)$ live in the open simplex through softmax logits. Exact zeros are unattainable in production parameters.
- Static $C$ uses a softplus diagonal. Dynamic Swish diagonals may be negative. There is no covariance repair.
- LSTM training in Block 4A uses fixed epochs, full BPTT, CPU float64, and RMSprop settings recorded above. No VALIDATION search was run.
- Forecast generation and forecast evaluation remain separate. Models and the serial runner do not compute losses. The evaluation adapter consumes `RollingForecastRecord` objects after the fact.
- The evaluation target is the realized-covariance proxy on the record's target date $t+1$, looked up by calendar key. Origin-day $S_t$ is never the scoring target.
- Compared models must share identical evaluation target dates by default. Intersection support is explicit.
- Reduced QLIKE and squared Frobenius are the existing Block 2A implementations. Evaluation does not repair forecasts.
- Loss differentials $d_t=L_{A,t}-L_{B,t}$ reuse the existing Block 3A helper. Descriptive summaries do not rank models.
- The synthetic benchmark generator and `scripts/synthetic_benchmark_demo.py` are integration fixtures. Demonstration hyperparameters are not tuned. Synthetic losses are not empirical findings.

## Known problems or limitations

- Realized kernels are not implemented.
- The long historical U.S.-equity panel remains unresolved. The U.S.-equity DATA GATE remains closed. The Binance first-stage panel is committed and was used for VALIDATION and SCREEN. CONFIRM remains locked.
- Local equity ticks are a one-session cache. `data/cache/taq_five_stock_single_exchange_cleaned_20090213.pkl` holds 697588 cleaned midquote ticks for AAPL, IBM, JPM, MSFT, and XOM on 2009-02-13 only. No multi-day RCov cube, matched daily-return panel, or RQ panel is stored.
- Official M6 `assets_m6.csv` is a 100-asset daily price file from 2022-01-31 through 2023-02-17. Maximum coverage-only common intersection among 5 names is 265 dates. It cannot support a 1,250- or 1,500-day daily-return auxiliary under the confirmatory allocation. It is not a high-frequency RCov source.
- Official Binance monthly spot 1m archives for the first empirical universe BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, and ADAUSDT support a 1,750-date segmented production panel from 2021-11-09 through 2026-08-25. Measurement, hard-break calendar, production arrays, segmented schedule, eleven-model roster, and VALIDATION grids are frozen. Binance VALIDATION fitting completed on 2026-09-21 as a development-set exercise. Binance SCREEN completed on 2026-09-21 as selection data. CONFIRM remains locked. Klines remain distinct from TAQ midquotes. U.S.-equity DATA GATE remains closed.
- Production WRDS Daily TAQ (`taqmsec` / `taqm_YYYY`) remains catalog-visible without verified SELECT. The 2026-09-14 inventory recorded `SELECT 1` failures on three production CQM dates with permission denied for the underlying year schemas. That probe was not rerun.
- `crsp.dsf` and `crsp.stocknames` passed constant-only SELECT in that inventory. They are not local extracts and cannot replace quote-based RCov.
- Alpaca historical SIP was not probed. No `APCA-*` / `ALPACA-*` credentials exist in the environment or project configuration. Untested SIP access is not available.
- A 5–10 asset, approximately 500-session exploratory panel is not assembled. Confirmatory allocation still requires `T-m>=1000`. A separate pilot protocol amendment is required before exploratory fitting.
- The graph-neural deep-learning specification is unresolved. Final `PREREGISTRATION.md` cannot be written yet.
- Blocks 3A, 3B, and 3C are accepted as closed. The bounded synthetic DM size-sensitivity study has been run. The current automatic lag remains the baseline. Confirmatory DM use still awaits review of that evidence. No additional robustness procedure has been implemented.
- The BNS jump indicator can miss an idiosyncratic jump that is small in the equal-weight market average, and a common jump can be flagged even if some names did not jump.
- Random-walk and EWMA forecasts that remain singular PSD are not QLIKE-evaluable. That is a model-output limitation, not a license to repair $H$.
- HAR-DRD, HARQ-DRD, Ridge-DRD, and XGBoost-DRD recorded zero VALIDATION repair/fallback events on the 250 development targets. On SCREEN, HARQ-DRD recorded 2 estimation-window-mean fallbacks. Those counts are specific to this panel and block. They are not CONFIRM results.
- HARQ-DRD on Binance VALIDATION used the production per-asset RQ panel. TAQ-to-RQ assembly remains unimplemented.
- LW-NL requires $T\ge 13$ because the pinned `nonlinshrink` reference rejects $n_{\mathrm{eff}}<12$. Linear shrinkage has no such extra floor beyond $T\ge 2$.
- DCC-NL requires $T\ge 12$ because the same reference with $k=0$ uses $n_{\mathrm{eff}}=T$.
- Plain DCC sample targeting is unsupported when $N>T$. At $N\le T$, rank is not inferred from dimensions.
- Univariate GARCH QMLE can land on the IGARCH boundary $a+b=1$, which `arch` allows and the project rejects. That is a surfaced fit failure, not a silent repair. On this SCREEN panel DCC and DCC-NL both failed at origin 2023-12-20 with $a=0$, $b=1$ for asset 1. They were not replaced.
- Importing `nonlinshrink` emits NumPy's `PendingDeprecationWarning` for `numpy.matlib`. The warning is from the pinned reference, not from project code. It is not suppressed.
- Ridge-DRD $\lambda$ is an explicit hyperparameter. Binance VALIDATION selected `RIDGE01` ($\lambda=0$), which nests HAR-DRD. That selection is a development-set outcome.
- LSTM-BEKK-RC, GHAR, and the graph-neural slot are not implemented. Portfolio evaluation is not implemented. cDCC is not implemented.
- XGBoost-DRD constructor hyperparameters are explicit. Binance VALIDATION selected `XGB08`. That selection is a development-set outcome.
- Pooled XGBoost-DRD correlation training at $N=200$ has roughly $4.5$ million refit rows. Pair subsampling is not implemented. That remains a computing limitation.
- LSTM-BEKK rolling fits at $N=100$ or $N=200$ with $T=250$ remain a computing and overparameterization limitation. The source paper's longer panels are not this protocol.
- The LSTM 20-configuration Binance training grid was run on VALIDATION. The selected ensemble configuration is `LSTM19`. Dropout $0$ remains the no-dropout stacked-LSTM setting. Seed-level scores were retained and were not used for selection.
- The serial synthetic runner generates forecasts only. The Binance VALIDATION and SCREEN orchestrators reuse that private fit and advance dispatch on their respective calendars. CONFIRM is not requested by either orchestrator.
- The synthetic evaluation adapter and demonstration script have been used on synthetic panels only. They are not an empirical benchmark.
- A full production `TemporalProtocol` allocation still requires $T-m\ge 1000$. The demonstration therefore uses a short `build_schedule` rather than VALIDATION/SCREEN/CONFIRM lengths.
- Daily-moving-window Ledoit-Wolf is not implemented. The headline LW cadence holds the estimator between 21-origin refits.
- Final `PREREGISTRATION.md` is absent. The draft remains the only protocol record.
- The Newey-West 1994 lag is short relative to a highly persistent AR(1). Under the current automatic lag, two-sided 5 percent rejection at $T=250$ is $0.0558$ for $\rho=0$ and $0.1228$ for $\rho=0.6$, rising to $0.3818$ for $\rho=0.9$. Larger $T$ reduces but does not remove the high-persistence distortion. Mean $\hat\omega/\omega_{\mathrm{true}}$ falls with $\rho$. This is recorded finite-sample behavior of the current procedure, not a coding defect and not a selected new default.
- The candidate recentered stationary-bootstrap mean test remains oversized at high persistence. Stage 2 at $T=250$, $\rho=0.9$ rejects at $0.253$ versus stored automatic NW $0.3818$. Closer numerical size in some cells is not an adoption decision.
- MCS may retain a large set when forecasts are highly correlated. That is a feature of the procedure, not a code failure.
- Hansen SPA assumes positive differential variance. Exact-constant alternatives are rejected rather than studentized. The SCREEN wrapper labels an identically zero benchmark differential `exact_benchmark_tie`, keeps that model in the universe and MCS, and excludes only that column from SPA studentization. RIDGE01 versus HAR-DRD is that case on this SCREEN panel.
- Approximate PS21 weighting is a named approximation to Patton–Sheppard equation 21. It does not recover the unknown conditional proxy-error variance.
- Bartlett Newey-West long-run variance is positive semi-definite, so a non-roundoff negative Giacomini–Rossi LRV is not constructible without changing the estimator.

- SCREEN SPA, MCS, and median-rank finalists are selection evidence on 2023-11-30 through 2025-04-12. They are not confirmatory rankings. LSTM-BEKK remains the DL finalist by construction even though it is outside the primary QLIKE MCS at $\alpha=0.10$.
- The headline SCREEN comparison is HARQ-DRD versus LSTM-BEKK. That pair is frozen for a later CONFIRM authorization decision. CONFIRM has not been authorized and has not been run.
- Descriptive SCREEN ranks in `results/binance_screen_model_summary.csv` use `np.argsort(np.argsort(...))`. HAR-DRD and RIDGE01 have identical saved losses and still receive distinct ordinal ranks. That display artifact is not evidence of different performance. Finalist selection uses a separate average-rank helper. The frozen CSV was not rewritten.
- HARQ-DRD SCREEN diagnostics record `repair_count=2`. Checkpoint flags locate targets 2024-12-10 and 2025-03-03. The failed raw-forecast validity checks were not stored. No eigenvalue or optimization explanation was invented.
- Confirmatory pairwise inference remains unresolved. The Newey–West Diebold–Mariano calibration and the calibrated-but-unadopted bootstrap companion are not a settled CONFIRM default. Pairwise Diebold–Mariano was not run on Binance SCREEN.
- HARQ-DRD consumes realized covariance and per-asset RQ. LSTM-BEKK consumes daily returns only. Those information sets are not matched. SCREEN does not isolate architecture from information advantage.

## Next recommended task

Review the frozen SCREEN finalists HARQ-DRD (`HARQDRD01`) and LSTM-BEKK (`LSTM19` equal-weight seed ensemble) and decide whether to authorize a separate CONFIRM execution. Do not run CONFIRM in that review. Do not unlock CONFIRM unless that review explicitly authorizes it. Do not retune. Do not reopen VALIDATION or SCREEN. Do not open the U.S.-equity DATA GATE. Do not add LSTM-BEKK-RC, GHAR, or a graph-neural model.

The 2026-09-20 equity feasibility write-up remains `docs/EMPIRICAL_DATA_FEASIBILITY.md`. The open-data write-up is `docs/OPEN_DATA_FEASIBILITY.md`. The measurement specification is `docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md`. The production panel is `docs/BINANCE_OPEN_DATA_PANEL.md`. The protocol freeze is `docs/BINANCE_OPEN_DATA_PROTOCOL.md`. Model contracts are `docs/MODEL_IMPLEMENTATION.md`. Inference implementation is `docs/INFERENCE_METHODS.md`. The U.S.-equity quote path is `docs/EQUITY_MEASUREMENT.md`. VALIDATION and SCREEN artifacts are listed in the protocol document.
