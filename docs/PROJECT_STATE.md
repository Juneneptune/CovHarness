# Project state

Last updated 2026-09-15.

Project conda environment is `covharness` (Python 3.11). Recreate with `conda env create -f environment.yml` from the repository root.

## Current milestone

Block 4A-1 common model contract, random-walk realized covariance, and EWMA realized covariance. Both baselines are synthetic/unit validated. No empirical fitting occurred. The DATA GATE remains closed. Blocks 1, 2A, 2B, 3A, 3B, and 3C are accepted as closed. HAR-DRD and later roster members were not begun.

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

`BENCHMARK_IMPLEMENTATION_PLAN.md` no longer states that `M>N` is required for QLIKE evaluation. Kernels remain in the broader plan as alternative proxies.

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
- `tests/unit/test_epps.py`
- `notebooks/proxy_robust_losses.ipynb`
- `notebooks/dm_hac_size.ipynb`
- `results/proxy_robust_losses.png`
- `results/dm_hac_size.png`
- `PREREGISTRATION_DRAFT.md`
- `README.md`
- `docs/PROJECT_STATE.md`
- `BENCHMARK_IMPLEMENTATION_PLAN.md`
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

## Methodological decisions already in code

- Reduced QLIKE is the primary ranking loss. Full Stein is the SPD-proxy form.
- Squared Frobenius is the complementary robust loss. Ordinary unsquared Frobenius is a labeled non-robust contrast only.
- Localization diagnostics are descriptive. They are not ranking losses.
- Forecasts that fail PD are exposed, not repaired.
- `M/N` is a proxy-quality diagnostic, not a gate on reduced QLIKE.
- Protocol intervals are half-open on the trading-date index.
- VALIDATION 250 is a target. SCREEN and CONFIRM 500 are committed minima. Zero VALIDATION makes data-driven tuning unavailable.
- Rolling $m=250$ and 21-day refit cadence are part of the forecasting method.
- CONFIRM is locked unless `unlock_confirm=True` is passed.
- Stochastic seeds and the configuration budget are protocol metadata, not later model-local choices.
- Statistical evaluation is open-to-close. Later economic GMV includes overnight.
- Pairwise tests use $d_t=L_{A,t}-L_{B,t}$ and Bartlett / Newey-West HAC. Harvey-Leybourne-Newbold is not applied.
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

## Known problems or limitations

- Realized kernels are not implemented.
- The long historical empirical panel is unresolved. The DATA GATE remains closed. Empirical Block-4 fitting is blocked until that source is committed and verified.
- The graph-neural deep-learning specification is unresolved. Final `PREREGISTRATION.md` cannot be written yet.
- Blocks 3A, 3B, and 3C are accepted as closed. The pending bounded DM dependence/calibration review remains pending before confirmatory use.
- The BNS jump indicator can miss an idiosyncratic jump that is small in the equal-weight market average, and a common jump can be flagged even if some names did not jump.
- Random-walk and EWMA forecasts that remain singular PSD are not QLIKE-evaluable. That is a model-output limitation, not a license to repair $H$.
- HAR-DRD, HARQ-DRD, shrinkage, DCC, Ridge-DRD, LSTM-BEKK, GHAR, and the graph-neural slot are not implemented. Portfolio evaluation is not implemented.
- The Newey-West 1994 lag is short relative to a highly persistent AR(1). Under $\rho=0.6$ and $T=250$, HAC DM still over-rejects relative to 5 percent, while remaining far closer to nominal size than an IID $t$-test.
- MCS may retain a large set when forecasts are highly correlated. That is a feature of the procedure, not a code failure.
- Hansen SPA assumes positive differential variance. Exact-constant alternatives are rejected rather than studentized.
- Approximate PS21 weighting is a named approximation to Patton–Sheppard equation 21. It does not recover the unknown conditional proxy-error variance.
- Bartlett Newey-West long-run variance is positive semi-definite, so a non-roundoff negative Giacomini–Rossi LRV is not constructible without changing the estimator.

## Next recommended task

Implement HAR-DRD on the same realized-covariance model contract, still synthetic/unit only. Do not begin empirical fitting. The DATA GATE remains closed.
