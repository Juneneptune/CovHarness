# Model implementation

This document records the exact one-day-ahead covariance-model contracts. The public research narrative and SCREEN roster live in [`README.md`](../README.md). Models do not inspect VALIDATION, SCREEN, or CONFIRM labels. Evaluation remains a separate step in `covharness.evaluation`. Invalid matrices are not repaired at evaluation time.

---

## Common contract

The common forecast object is `covharness.models.CovarianceModel`. Realized-covariance models that consume only a covariance cube inherit `RealizedCovarianceModel`. HARQ-DRD uses the same forecast object and a `fit` that also takes a $(T,N)$ per-asset quarticity window. Ridge-DRD and XGBoost-DRD consume the same $(T,N,N)$ cube as HAR-DRD. Standalone Ledoit–Wolf, DCC, DCC-NL, and LSTM-BEKK models consume a $(T,N)$ daily-return window through `as_daily_return_history`.

Rolling behavior is declared by `ModelCapabilities` rather than class-name dispatch. The four cadences are origin-map (random walk), window-state (HAR, HARQ, Ridge, XGBoost), recursive-state (EWMA, DCC, DCC-NL, LSTM-BEKK), and refit-hold (standalone LW-linear and LW-NL). The one-step forecast is an independent $N\times N$ copy. Inputs are not mutated.

At a refit origin the runner calls `fit` on the current $m=250$ window through $t$ and then `forecast`. It does not ingest origin $t$ a second time. At a non-refit origin it advances daily state once, then forecasts. The evaluation adapter looks up the realized-covariance proxy by the record's target date, so $H_{t+1\mid t}$ is scored against $S_{t+1}$ and never against $S_t$. A non-PD forecast fails the existing QLIKE domain contract.

---

## Random walk

Identity `random_walk`. Source [`random_walk.py`](../src/covharness/models/random_walk.py).

```math
H_{t+1\mid t}=S_t.
```

Earlier matrices in the window do not enter. A singular PSD $S_t$ remains singular. Reduced QLIKE still requires a strictly PD forecast at evaluation time. That limitation is diagnosed, not repaired in the model.

---

## EWMA

Identity `ewma`. Source [`ewma.py`](../src/covharness/models/ewma.py).

EWMA is defined on the realized-covariance sequence $S_0,\ldots,S_{T-1}$ with decay $\lambda\in(0,1)$.

```math
H_0=S_0,\qquad
H_j=\lambda H_{j-1}+(1-\lambda)S_j,\quad j=1,\ldots,T-1.
```

The forecast is $H_{T\mid T-1}=H_{T-1}$. This is not an EWMA of daily-return outer products. $\lambda$ is an explicit constructor argument. The conventional RiskMetrics reference $0.94$ may be passed by a caller. It is not a tuned project choice. Binance VALIDATION selected EWMA01 with $\lambda=0.8000$. Between coefficient-free refits of the 250-day window, `update(S_t)` applies one recursion step with that same frozen $\lambda$. Lambda is never re-estimated by `update`. EWMA of PSD inputs is PSD. A shared null space can remain singular.

---

## HAR-DRD

Identity `har_drd`. Source [`har_drd.py`](../src/covharness/models/har_drd.py).

Headline HAR-DRD is a Zhang-style non-overlapping HAR on the Oh–Patton DRD split of each supplied realized covariance. The model is in levels. It does not use a log-variance transform, Fisher transform, ridge, graph term, or HARQ term.

For every supplied $S_t\in\mathbb{R}^{N\times N}$ we set $`v_t=\mathrm{diag}(S_t)`$, $`D_t=\mathrm{diag}(\sqrt{v_t})`$, and $`R_t=D_t^{-1}S_t D_t^{-1}`$. Every diagonal entry of every $S_t$ must be strictly positive. Positive semidefiniteness alone is not sufficient if a diagonal is zero, because $R_t$ is then undefined. Such a window is rejected with `InvalidModelInputError`. Input matrices are not altered.

Unique correlations use the repository's existing strict upper-triangle order $i<j$, matching `np.triu_indices(N, k=1)` in the Epps and Giacomini–White helpers. The pair count is $P=N(N-1)/2$.

At response date $k\ge 22$ the non-overlapping HAR features are one daily lag, four weekly observations, and seventeen monthly observations.

```math
z_d(k)=\text{value}[k-1],\qquad
z_w(k)=\mathrm{mean}(\text{value}[k-5:k-1]),\qquad
z_m(k)=\mathrm{mean}(\text{value}[k-22:k-5]).
```

The Python slices are exclusive on the right. The three blocks do not overlap. For a supplied window of length $T=250$ the response dates are $22,\ldots,249$, so there are $228$ regression dates, $228N$ variance observations, and $228P$ correlation observations. The construction never reads before local index $0$. Predictors for the target immediately after the window are $`\text{value}[T-1]`$, $`\mathrm{mean}(\text{value}[T-5:T-1])`$, and $`\mathrm{mean}(\text{value}[T-22:T-5])`$.

Variance equations have asset-specific intercepts $\alpha_D\in\mathbb{R}^N$ and three shared scalar slopes $\beta_D=(\beta_{D,d},\beta_{D,w},\beta_{D,m})$. Correlation equations have pair-specific intercepts $\alpha_R\in\mathbb{R}^P$ and three shared scalar slopes $\beta_R$. We estimate those maps by the exact fixed-effects within transformation. For each group we demean $y$ and the three-column design, stack only the demeaned three-column matrices, solve $\beta$ by least squares, and recover intercepts from the group means. We do not build an $N$-column or $P$-column intercept-dummy matrix.

The raw one-day forecast reconstructs $R_{\mathrm{raw}}$ with unit diagonal in the frozen pair order. When $v_{\mathrm{raw}}$ is finite and strictly positive, $H_{\mathrm{raw}}=D_{\mathrm{raw}}R_{\mathrm{raw}}D_{\mathrm{raw}}$ with $`D_{\mathrm{raw}}=\mathrm{diag}(\sqrt{v_{\mathrm{raw}}})`$. Raw components are stored even if the headline matrix is later replaced.

The headline forecast uses an explicit BPQ-style insanity filter. The raw forecast is valid only when every $v_{\mathrm{raw}}$ is finite and strictly positive, every $x_{\mathrm{raw}}$ is finite, every pairwise correlation lies in $[-1,1]$, $R_{\mathrm{raw}}$ is symmetric and strictly positive definite, and $H_{\mathrm{raw}}$ is finite, symmetric, and strictly positive definite. If all of those hold, $H_{\mathrm{final}}=H_{\mathrm{raw}}$, `repaired=False`, and `repair_method=None`. If any condition fails, $H_{\mathrm{final}}$ is the arithmetic mean of every realized covariance in the current origin's 250-day window, `repaired=True`, and `repair_method="estimation_window_mean"`. Between coefficient refits that fallback mean moves with the window. Coefficients do not. That fallback itself must be finite, symmetric, and strictly positive definite. Otherwise the model raises `InvalidModelForecastError`.

There is no second repair. The implementation does not clip correlations, floor eigenvalues, call `cov_nearest`, apply Higham projection, add jitter, diagonal-load, transform $v$ or $x$, or replace individual entries. Repair is never silent. Repair frequency is reported. A valid final matrix does not imply that the raw forecast was valid. Between coefficient refits `update_window` rebuilds origin daily, weekly, and monthly predictors from the current 250-day realized-covariance window and replaces that fallback mean. It does not re-estimate $\alpha_D$, $\beta_D$, $\alpha_R$, or $\beta_R$.

---

## HARQ-DRD

Identity `harq_drd`. Source [`harq_drd.py`](../src/covharness/models/harq_drd.py).

Headline HARQ-DRD is HAR-DRD plus one daily per-asset quarticity interaction on the variance equation. The correlation map is unchanged. There is no weekly or monthly quarticity term, no correlation attenuation, no log-variance transform, no Fisher transform, no ridge, and no graph term.

The documented per-asset realized quarticity is $\mathrm{RQ}_{i,t}=(M/3)\sum_l r_{i,t,l}^4$. HARQ consumes a precomputed window of shape $(T,N)$ aligned with the covariance history. It does not read raw intraday returns. That per-asset series is not the cross-sectional aggregate $\mathrm{RQ}_{\mathrm{agg}}=\mathrm{mean}_i\mathrm{RQ}_i$ used by the second Giacomini–White measurement state, and it is not logged. Zero RQ is allowed. Negative or non-finite RQ is rejected.

Variance intercepts $\alpha_Q$ are asset-specific. The four shared variance slopes are daily, the quarticity interaction $\phi_{Q,d}$ on $\sqrt{\mathrm{RQ}_{i,k-1}}\,v_{i,k-1}$, weekly, and monthly. No sign constraint is imposed on $\phi_{Q,d}$. Correlation intercepts and the three shared HAR slopes are the HAR-DRD objects. Both maps use the same within-transformation estimator. The variance design has four columns. The correlation design has three. Dummy intercept matrices are not built.

The headline forecast uses the same explicit BPQ-style estimation-window-mean insanity filter as HAR-DRD. That fallback is a model-internal, origin-window operation. It is distinct from prohibited evaluation-time repairs. Between coefficient refits `update_window` rebuilds HAR variance predictors, HAR correlation predictors, the origin-day per-asset RQ interaction, and the current-window fallback mean. It does not refit the four-slope HARQ regression.

---

## Ridge-DRD

Identity `ridge_drd`. Source [`ridge_drd.py`](../src/covharness/models/ridge_drd.py).

Headline Ridge-DRD is the regularized HAR-DRD control. Variance targets remain realized-variance levels. Correlation targets remain raw realized correlations. Predictors are the same non-overlapping daily, weekly (exactly four observations), and monthly (exactly seventeen observations) HAR features. There is no quarticity term, cross-sectional average, market-state variable, pair-summary feature, daily-return feature, macro series, log-variance transform, or Fisher transform.

Variance maps keep asset-specific intercepts and three shared slopes. Correlation maps keep pair-specific intercepts and three shared slopes. Dummy intercept matrices are not built. After the within transformation, each of the three predictor columns is divided by its estimation-window RMS $\sqrt{\mathrm{mean}(\widetilde X_j^2)}$. An exact zero column receives scale 1 and is retained. Responses are not standardized. Scales are fit only on the caller-supplied window at a scheduled refit. Between refits `update_window` rebuilds current origin features in raw units and replaces the fallback mean. It does not recompute RMS scales, slopes, intercepts, or $\lambda$.

The explicit penalty $\lambda\ge 0$ is required at construction and is shared by the variance and correlation problems. On the scaled within design the objective is the sum of squared errors plus $\lambda\|\gamma\|_2^2$. Intercepts are recovered after the slopes and are not penalized. Raw-space slopes are $\beta_j=\gamma_j/\mathrm{scale}_j$. The case $\lambda=0$ nests the implemented HAR-DRD OLS problem. Binance VALIDATION selected RIDGE01 with $\lambda=0$. sklearn is a test reference only.

The headline forecast uses the same explicit BPQ-style estimation-window-mean insanity filter as HAR-DRD.

---

## XGBoost-DRD

Identity `xgboost_drd`. Source [`xgboost_drd.py`](../src/covharness/models/xgboost_drd.py).

Headline XGBoost-DRD completes the controlled HAR-DRD to Ridge-DRD to XGBoost-DRD ladder. We hold fixed the variance-level and raw-correlation targets, the non-overlapping $1/4/17$ information set, the lag convention, the asset and pair group structure, the dummy-free within transformation, the Ridge RMS-scaled predictor representation, the WINDOW_STATE cadence, covariance reconstruction, and the current-origin 250-day mean repair. The linear ridge learner is replaced by a boosted-tree learner.

There is no quarticity term, cross-sectional average, pair summary, asset or pair identity feature, group-ID column, return, macro series, VIX, volume, OptionMetrics, log-variance transform, or Fisher transform. Unique pairs remain the strict upper triangle $i<j$. For $T=250$ the response dates remain $22,\ldots,249$, so there are $228$ regression dates.

Group-specific location remains entirely through the refit-window means $\bar y_D[g]$ and $\bar y_R[p]$. Dummy intercept columns are not built. Group identifiers are not passed to XGBoost. We do not claim nonlinear Frisch–Waugh–Lovell equivalence. After the same within demeaning, each of the three predictor columns is divided by the Ridge RMS $\sqrt{\mathrm{mean}(\widetilde X_j^2)}$. An exact zero column receives scale 1. Responses are not standardized. This scaling is imposed so that the numerical predictor representation matches Ridge exactly for the architecture-control experiment. It is not imposed because trees require feature scaling.

We train exactly two pooled boosters per refit. The variance map $f_D\colon\mathbb{R}^3\to\mathbb{R}$ and the correlation map $f_R\colon\mathbb{R}^3\to\mathbb{R}$ are shared across groups. We do not train $N$ variance models or $P$ pair models. Both boosters use `objective="reg:squarederror"` on the within residuals. There is no internal validation set and no early stopping.

Frozen package constants are `booster="gbtree"`, `tree_method="hist"`, `device="cpu"`, `n_jobs=1`, `subsample=1.0`, every `colsample` value $1$, `grow_policy="depthwise"`, `max_bin=256`, `base_score=0.0`, `random_state=0`, and `early_stopping=False`. Runtime is pinned at `xgboost==3.2.0`. Training is CPU only. Headline XGBoost-DRD is deterministic by that configuration. We do not apply the stochastic-model seed ensemble $(0,1,2,3,4)$. Binance VALIDATION selected XGB08.

At a forecast origin the raw HAR predictors are centered with the frozen refit-window group means and divided by the frozen RMS scales. Raw forecasts are $\bar y_g+f(Z_{\mathrm{origin},g})$. Between refits `update_window` rebuilds current-origin HAR predictors and the current 250-day fallback mean. It does not call `booster.fit`, recompute group means, or recompute scales. The headline repair remains the arithmetic mean of the current origin window.

SHAP, ALE, and feature-importance analysis are not implemented. Pair subsampling is not implemented. High-dimensional pooled-correlation training is a computing limitation, particularly near $N=200$, where the refit design has roughly $4.5$ million rows.

---

## Standalone Ledoit–Wolf

Identities `lw_linear` and `lw_nl`. Source [`ledoit_wolf.py`](../src/covharness/models/ledoit_wolf.py).

Standalone Ledoit–Wolf models consume a caller-supplied daily-return window of shape $(T,N)$. They do not consume realized-covariance histories. They do not shrink a DCC targeting matrix. DCC-NL is a separate model that uses the same analytical estimator only as its correlation intercept.

Returns are demeaned inside the current supplied window. For $T$ observations we set $Y=X-\mathrm{column\_mean}(X)$, $n_{\mathrm{eff}}=T-1$, and $S=Y^{\top}Y/(T-1)$. The same centered sample covariance is the starting object for both estimators. There is no annualization, scaling, winsorization, or silent missing-value deletion. On the Binance first-stage, the daily return is the matched 24-hour UTC interval used for RCov and RQ.

Headline LW-linear is Ledoit–Wolf 2004b rotation-equivariant shrinkage toward $\mu I$, with $`\mu=\mathrm{tr}(S)/N`$.

```math
\Sigma_L=(1-\rho)S+\rho\mu I.
```

$\rho$ is the Ledoit–Wolf estimated intensity. It is not a user-chosen coefficient. Equivalently, if $`S=U\mathrm{diag}(\lambda_i)U^{\top}`$, then $`\Sigma_L=U\mathrm{diag}((1-\rho)\lambda_i+\rho\mu)U^{\top}`$. Sample eigenvectors are retained. Every sample eigenvalue receives the same affine map. The Honey / equicorrelation-target estimator of Ledoit–Wolf 2004a is not the headline model.

Headline LW-NL is the Ledoit–Wolf 2020 analytical nonlinear estimator. We wrap the pinned PyPI package `nonlinshrink==0.7` (MIT license, https://github.com/matzhaugen/analytic_shrinkage), a transparent port of the 2018 working paper that became the 2020 Annals of Statistics method. It is not QuEST. It is not QIS 2022. We do not transcribe the kernel or Hilbert formulas. Centered returns $Y$ are passed with $k=1$ so that the reference uses $n_{\mathrm{eff}}=T-1$ and the same $S=Y^{\top}Y/(T-1)$. Sample eigenvectors are retained. Shrunk eigenvalues are eigenvalue-specific and are not a common affine map of the sample spectrum.

The reference requires $n_{\mathrm{eff}}\ge 12$, so $T\ge 13$. The $N\ge T$ supplement branch of the 2020 paper is exposed by that package and is included. A nonfinite, asymmetric, or non-strictly-PD reference matrix raises `InvalidModelForecastError`. There is no silent repair, jitter, or eigenvalue floor.

Both models treat the window estimate as the one-day-ahead forecast $H_{t+1\mid t}=\widehat{\Sigma}_t$. Standalone LW-linear and LW-NL follow the common 21-origin estimator-refit cadence as a project fairness choice. At `ForecastStep.refit=True` the entire estimator is recomputed from the current 250-day return window. At non-refit origins the stored covariance is returned unchanged. Sample covariance, $\rho$, and nonlinear eigenvalues are not refreshed daily. There is no recursive update and no partial daily LW rule such as a frozen $\rho$ with a refreshed $S$. A later robustness may evaluate daily-moving-window LW. That is not the headline cadence.

---

## Original DCC and DCC-NL

Identities `dcc` and `dcc_nl`. Source [`dcc.py`](../src/covharness/models/dcc.py).

Headline DCC models consume a caller-supplied daily-return window of shape $(T,N)$ through `as_daily_return_history`. They require $N\ge 2$. They do not consume realized-covariance histories.

Both identities use original Engle (2002) DCC, not Aielli cDCC. Public classes are `DCCCovariance` and `DCCNonlinearCovariance`. They share stage-one GARCH, the DCC recursion, the all-pairs composite likelihood, SLSQP, forecast construction, and daily update. They differ only in the intercept $C$. cDCC is deferred as a potential named robustness and is not implemented.

Raw returns are demeaned once per parameter-fit window. For each asset, $\mu_i$ is the mean of the supplied window and $\varepsilon_{i,t}=r_{i,t}-\mu_i$. A ZeroMean Gaussian GARCH(1,1) is then fit to $\varepsilon_i$ with the pinned runtime dependency `arch==8.0.0`. Between parameter refits, $\mu_i$ is frozen. Every newly observed return is centered with that same $\mu_i$. The mean is re-estimated only on the next common parameter refit.

The explicit backcast is the sample second moment with divisor $T$.

```math
h_{i,0}=\frac{1}{T}\sum_{t=0}^{T-1}\varepsilon_{i,t}^2.
```

That value is passed through `arch`'s explicit backcast argument. We do not initialize with $\omega_i/(1-a_i-b_i)$ and we do not inherit package mean, scaling, or backcast defaults. The production call is `mean="Zero"`, `vol="GARCH"`, $p=1$, $o=0$, $q=1$, `dist="normal"`, and `rescale=False` on the already-centered series. Conditional variance follows $h_{i,t}=\omega_i+a_i\varepsilon_{i,t-1}^2+b_i h_{i,t-1}$. The standardized residual is $s_{i,t}=\varepsilon_{i,t}/\sqrt{h_{i,t}}$, never $\varepsilon_{i,t}/h_{i,t}$. Accepted GARCH parameters must satisfy $\omega_i>0$, $a_i\ge 0$, $b_i\ge 0$, and $a_i+b_i<1$. A failed asset fails the DCC fit. Assets are not dropped. Parameters are not altered after the package fit. On the Binance SCREEN panel both DCC identities failed when a univariate fit reached $a=0$, $b=1$. That is a surfaced admissibility failure, not a general finding about DCC, and not a license to nudge coefficients into the interior.

Plain DCC targeting does not demean standardized residuals again. It forms $\widetilde{C}=S_{\mathrm{std}}^{\top}S_{\mathrm{std}}/T$ and renormalizes the diagonal to obtain $C$. The sample target is guaranteed rank-deficient when $N>T$. At $N=T$ it may still be full rank. Strict positive definiteness is checked by Cholesky after construction. There is no $N\ge T$ rejection rule, no jitter, no nearest-PD, and no eigenvalue flooring.

DCC-NL changes only that intercept. It calls `nonlinshrink.shrink_cov(Sstd, k=0)` under the already pinned `nonlinshrink==0.7` runtime, so the target covariance uses the DCC zero-mean convention and effective divisor $T$. The returned matrix is then diagonally renormalized to $C_{\mathrm{NL}}$. This is not QuEST, not QIS, and not the standalone LW-NL demeaned $T-1$ wrapper. Nonlinear shrinkage is not applied to the final $H$. If the nonlinear estimator fails or the normalized target is invalid, the model raises `InvalidModelForecastError`. It does not fall back to plain $C$. The reference requires $n_{\mathrm{eff}}\ge 12$, so DCC-NL requires $T\ge 12$.

The original DCC recursion uses $Q_0=C_{\star}$ with $C_{\star}=C$ or $C_{\mathrm{NL}}$. Window rows are $t=0,\ldots,T-1$. $Q_t$ scores the observed standardized residual $s_t$ after conversion to $`R_t=\mathrm{diag}(Q_t)^{-1/2}Q_t\mathrm{diag}(Q_t)^{-1/2}`$. After $s_t$ is observed,

```math
Q_{t+1}=(1-\alpha-\beta)C_{\star}+\alpha s_t s_t^{\top}+\beta Q_t.
```

No cDCC transformed shock enters. Stored `current_q` is $Q_{T-1}$, the state for the last observed return. `forecast()` forms $Q_{T\mid T-1}$ from that state and does not mutate it. The one-day forecast is $H_{t+1\mid t}=D_{t+1\mid t}R_{t+1\mid t}D_{t+1\mid t}$ with $`D_{t+1\mid t}=\mathrm{diag}(\sqrt{h_{t+1\mid t}})`$. An invalid final matrix raises rather than being repaired.

Headline second-stage estimation is the all-pairs bivariate composite Gaussian quasi-likelihood over every unique pair $i<j$ in the existing strict upper-triangle order. The pair count is $N(N-1)/2$. Contiguous $N-1$ 2MSCLE is not used. A random pair subset is not used. Parameters $(\alpha,\beta)$ are estimated by deterministic SciPy SLSQP starting at $(0.05,0.90)$, with bounds $[0,1]\times[0,1]$ and constraint $\alpha+\beta\le 1$. There are no random restarts and no alternative starts. An accepted fit must satisfy $\alpha\ge 0$, $\beta\ge 0$, and $\alpha+\beta<1$. The boundary $\alpha+\beta=1$ is rejected. Parameters are not moved inward by an epsilon.

Parameter refit cadence and state update are different. Parameters are re-estimated every 21 forecast origins. Between those refits, DCC and GARCH states update daily through `update(new_return)` and produce a new one-day-ahead covariance every day. `update` does not re-estimate $\mu$, GARCH parameters, $C_{\star}$, or $(\alpha,\beta)$.

---

## LSTM-BEKK

Identity `lstm_bekk`. Source [`lstm_bekk.py`](../src/covharness/models/lstm_bekk.py).

Headline LSTM-BEKK is the faithful daily-return competitor of Wang, Liu, Tran, and Wang (2025). It consumes a caller-supplied daily-return window of shape $(T,N)$ through `as_daily_return_history`. It requires $N\ge 2$. It does not consume realized covariance, realized quarticity, or any other intradaily feature. LSTM-BEKK-RC is deferred.

The paper recursion, in internal percent units $x_t=100(r_t-\mu)$, is scalar BEKK plus an LSTM intercept.

```math
H_t = CC^{\top} + C_t C_t^{\top} + a x_{t-1}x_{t-1}^{\top} + b H_{t-1}.
```

$C$ and $C_t$ are lower triangular. $a$ and $b$ are scalars. This is not full Engle–Kroner matrix $A,B$. Every term is positive semidefinite when $a,b\ge 0$. Strict positive definiteness is anchored by static $CC^{\top}$ with a strictly positive diagonal on $C$. Dynamic $C_t C_t^{\top}$ remains a Gram matrix even if a Swish-transformed diagonal entry is negative. There is no jitter, nearest-PD map, eigenvalue floor, or other silent repair.

Paper-specified architecture. The LSTM input is the lagged internal return. Hidden size equals the asset count $N$. Depth is three to five hidden layers. The constructor bound on dropout is $[0,0.2]$. The no-dropout stacked-LSTM setting is dropout $0.0$. The dynamic intercept is a lower triangle whose diagonal is passed through Swish $x\sigma(\beta x)$ with a learnable $\beta$. Training uses Gaussian negative log-likelihood, RMSprop, Cholesky evaluation of $\log\det H_t$ and the quadratic form, and gradient clipping. Empirical returns in the source paper are de-meaned and multiplied by 100.

The following completions are project choices. They fill omissions in the paper and are not attributed to Wang et al.

Stacked LSTM layers implement the paper's "3–5 hidden layers". A linear map $\mathbb{R}^N\to\mathbb{R}^{N(N+1)/2}$ with bias converts the hidden state to the packed triangle. Swish $\beta$ is one global scalar, initialized at $1$. Static $C$ is a packed lower triangle with a softplus diagonal. Stationarity uses three unconstrained logits whose softmax is $(w,a,b)$ on the interior of the simplex, initialized at $(0.05,0.05,0.90)$. $w$ is slack only. It does not enter the recursion. $H_0$ is the internal sample covariance $X^{\top}X/(T-1)$ when that matrix is strictly PD by Cholesky, otherwise the diagonal of that matrix when every variance is strictly positive. Static $C$ is initialized so $CC^{\top}=0.05 H_0$. Recurrent states start at zero on every `fit` and are not learned. Training is full-sequence BPTT with a fixed epoch count and no rolling-window early stopping. RMSprop uses $\alpha=0.99$, $\varepsilon=10^{-8}$, momentum $0$, uncentered updates, and no weight decay. The implementation is `torch.float64` on CPU under the pinned runtime dependency `torch==2.13.0`. The public forecast is $H/10000$ so caller-native covariance units are restored.

Fit indexing. Observed internal returns are $x_0,\ldots,x_{T-1}$. The first return is scored under $H_0$. No presample return is created. After $x_t$ is processed by the LSTM, $H_{t+1}$ scores $x_{t+1}$. All $T$ observations enter the training NLL. After the last scored return, $x_{T-1}$ is processed once more to form $H_T$, which is the one-step forecast and is not added to the fit-window likelihood. `forecast()` returns a copy of $H_T/10000$ and does not mutate state. `update(new_return)` centers with the frozen fit-window mean, scales by 100, advances hidden, cell, and $H$ once, and leaves all learned parameters unchanged. The serial runner calls `fit` only at scheduled refits. Non-refit origins call `update` once and do not reseed.

Constructor arguments `seed`, `num_layers`, `dropout`, `learning_rate`, `gradient_clip_norm`, and `max_epochs` are required. Binance VALIDATION selected LSTM19. Rolling SCREEN fits then used that frozen rule on the full 250-day window under seeds $(0,1,2,3,4)$. The reported LSTM column is the equal-weight mean of those five covariance matrices, then the loss. High-dimensional rolling estimation with $T=250$ remains a computing limitation. The model is not claimed to be practical at $N=100$ or $N=200$ under the project's window.
