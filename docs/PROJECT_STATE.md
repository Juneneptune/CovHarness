# Project state

Last updated 2026-09-13.

Project conda environment is `covharness` (Python 3.11). Recreate with `conda env create -f environment.yml` from the repository root.

## Current milestone

Block 3B. SPA and MCS multiple-model screening. Block 3A pairwise inference is closed. Giacomini-White, Mincer-Zarnowitz, Giacomini-Rossi, forecasting models, and portfolio evaluation were not added.

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

DATA GATE. Block 2 and Block 3 may use synthetic known-truth data. Block 4 must not begin until the empirical panel is committed and verified. The long historical dataset remains unresolved. No dataset was chosen in this block.

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
- `src/covharness/inference/__init__.py`
- `src/covharness/diagnostics/epps.py` (`m_over_n` added)
- `tests/unit/test_losses.py`
- `tests/unit/test_loss_robustness.py`
- `tests/unit/test_protocol.py`
- `tests/unit/test_inference.py`
- `tests/unit/test_inference_spa_mcs.py`
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

## Known problems or limitations

- Realized kernels are not implemented.
- The long historical empirical panel is unresolved. The DATA GATE blocks Block 4 until that source is committed and verified.
- The graph-neural deep-learning specification is unresolved. Final `PREREGISTRATION.md` cannot be written yet.
- Giacomini-White, Mincer-Zarnowitz, forecasting models, and portfolio evaluation are not implemented.
- The Newey-West 1994 lag is short relative to a highly persistent AR(1). Under $\rho=0.6$ and $T=250$, HAC DM still over-rejects relative to 5 percent, while remaining far closer to nominal size than an IID $t$-test.
- MCS may retain a large set when forecasts are highly correlated. That is a feature of the procedure, not a code failure.
- Hansen SPA assumes positive differential variance. Exact-constant alternatives are rejected rather than studentized.

## Next recommended task

Block 3C — Giacomini-White, Mincer-Zarnowitz, and Giacomini-Rossi
