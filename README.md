# covharness

`covharness` is a research implementation for the evaluation of one-day-ahead multivariate covariance forecasts. Its purpose is to provide a common measurement, estimation, and evaluation framework in which econometric, machine-learning, and deep-learning methods can be compared under the same empirical protocol.

The benchmark is designed so that a new forecasting method does not receive a different target, a richer information set, a larger tuning budget, or a more favorable evaluation criterion simply because it belongs to a different modeling tradition. Conventional models are therefore treated as serious competitors rather than default baselines. The planned comparison assigns common validation periods, tuning budgets, re-estimation schedules, and confirmatory procedures across model classes. Stochastic methods will be evaluated over pre-specified seed sets rather than selected ex post from favorable individual runs.

> **Current status.** The primary TAQ measurement path is exchange-level quotes, P1/P2, listing-venue P3, Q1–Q4, midquote, and previous-tick synchronization. Consolidated NBBO plus Q1–Q4 remains a labeled alternative. An Epps-effect frequency scan is implemented on the five-stock 13 February 2009 panel. Block 1 measurement work is closed. Covariance-space losses are implemented. Reduced QLIKE is the primary ranking loss. Squared Frobenius is the complementary robust criterion. Realized-kernel estimation is not implemented. The long historical extract is not solved.

Detailed implementation status is maintained in [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md).

---

## Why covariance forecast evaluation is difficult

The object of interest is the conditional covariance matrix

```math
\Sigma_{t+1\mid t} = \operatorname{Cov}(r_{t+1}\mid \mathcal{F}_t).
```

Unlike a standard supervised-learning target, this matrix is not observed directly, even after day $t+1$ has ended. Forecast evaluation must therefore rely on an ex-post covariance proxy constructed from higher-frequency observations. That proxy is informative but noisy, so a forecast comparison can be distorted by the choice of proxy, loss function, sampling scheme, or inferential procedure.

This consideration determines the structure of the repository. The measurement pipeline is fixed before forecasting models are introduced, the primary evaluation losses are chosen for their suitability with noisy volatility proxies, and confirmatory comparisons are separated from model development. The same covariance forecast is also economically relevant because portfolio variance is

```math
w^{\top}\Sigma_{t+1\mid t}w.
```

The project therefore maintains separate statistical and economic evaluation channels rather than treating one as a substitute for the other.

---

## Realized covariance proxy

Because $\Sigma_{t+1\mid t}$ is latent, forecasts are evaluated against realized covariance constructed from synchronized intraday returns. For a trading day with $M$ synchronized interval returns and $N$ assets, let

```math
R_t \in \mathbb{R}^{M\times N}.
```

The realized covariance estimator is

```math
\mathrm{RCov}_t
= R_t^{\top}R_t
= \sum_{j=1}^{M} r_{t,j}r_{t,j}^{\top}.
```

The implementation uses this unscaled Gram matrix. It does not divide by the number of intraday intervals, annualize the matrix, apply shrinkage, clip eigenvalues, or otherwise repair the estimator. Realized covariance is treated as an ex-post proxy for the latent covariance process, not as observed ground truth.

---

## Quote cleaning

Raw high-frequency quotes are not a clean observation of the latent price. Consolidated NBBO records include crossed markets, extremely wide transient spreads, and isolated midquotes that are inconsistent with neighboring quotes and with contemporaneous trades. Previous-tick synchronization cannot repair those states. It carries the most recent supplied price onto the sampling grid, so an invalid quote immediately before a grid time becomes the synchronized price.

That matters because realized covariance is a quadratic form. If $r_j$ is the synchronized return vector at interval $j$, then

```math
\mathrm{RCov}_t = \sum_{j=1}^{M} r_{t,j} r_{t,j}^{\top}.
```

An artificial return $r_{i,j}$ contributes $r_{i,j}^{2}$ on the variance diagonal and $r_{i,j} r_{k,j}$ to every covariance involving asset $i$. One contaminated interval can therefore distort an entire row and column of the daily proxy. The primary measurement pipeline is therefore

```math
\text{exchange-level quotes}
\to
\text{P1, P2}
\to
\text{P3 (listing venue)}
\to
\text{Q1–Q4}
\to
\text{midquote}
\to
\text{previous-tick synchronization}
\to
\text{log returns}
\to
\text{realized covariance}.
```

The implemented procedure is adapted from Section 5.1 of Barndorff-Nielsen, Hansen, Lunde, and Shephard (2011), who follow the more detailed trade-and-quote cleaning discussion in Barndorff-Nielsen, Hansen, Lunde, and Shephard (2009). The 2011 paper applies P3 before Q1–Q4. It retains quotes from one listing venue, NYSE for ordinary NYSE-listed names and NASDAQ for the NASDAQ-listed names in their sample.

Q1–Q4 are a single implementation, `covharness.data.quotes.clean_nbbo_quotes`. Two input adapters feed that cleaner.

**Paper-style single-exchange path.** Exchange-level quotes from `cqm_*` are restricted to the listing venue (P3) and to the requested TAQ suffix, then passed through P1, P2, and Q1–Q4. Listing venue is taken from CRSP `stocknames.exchcd` on the sample date (1 = NYSE, 3 = NASDAQ) and mapped to Daily TAQ Exchange-field codes (`N` for NYSE, `T`/`Q` for NASDAQ). Ordinary/common shares in the current five-stock extract use the blank or NULL suffix. The adapter is `covharness.data.exchange_quotes.clean_single_exchange_quotes`. It does not copy Q1–Q4.

**Consolidated NBBO alternative path.** The same Q1–Q4 function can be applied to `best_bid` / `best_ask` without P3. That path is retained and labeled. It is not deleted.

The quote rules themselves are unchanged.

P1. Delete observations outside regular exchange hours, 09:30–16:00 inclusive on the caller-supplied exchange-local clock.

P2. Delete observations with non-positive or non-finite bid or ask.

P3. On the single-exchange path only, retain quotes issued by the selected listing venue for the requested `sym_root` and `sym_suffix` identity.

Q1. If several quotes for the same security have the same timestamp, replace them by one observation using the median bid and the median ask.

Q2. Delete quotes with a negative spread, $\mathrm{ask}-\mathrm{bid}<0$. Locked quotes with zero spread are retained.

Q3. For each stock-day, compute the median spread after Q1 and Q2, and delete quotes whose spread exceeds ten times that median.

Q4. For each stock-day quote with a complete centered neighborhood, take the 25 observations immediately before and the 25 immediately after the quote, excluding the quote itself. Let $N_i$ be those 50 neighboring midquotes, $\widetilde{m}_i=\mathrm{median}(N_i)$, and $D_i$ the mean absolute deviation of $N_i$ from $\widetilde{m}_i$. Delete the quote if $|\mathrm{mid}_i-\widetilde{m}_i|>10 D_i$. If $D_i=0$, retain the center only when it equals $\widetilde{m}_i$. This $D_i$ is a mean absolute deviation from the local median, not the usual median-absolute-deviation statistic.

Q4 is a symmetric ex-post filter on historical measurement data. Neighborhoods stay inside a stock-day. After the filters, the midquote is $(\mathrm{bid}+\mathrm{ask})/2$.

Cleaning rules depend on the market-data representation to which they are applied. On 13 February 2009, the same Q1–Q4 rules applied to consolidated JPM NBBO left clustered five-minute dislocations near 22.6 at 09:45 and 10:40. Restoring the paper's P3 construction, NYSE-issued quotes for JPM, removed those five-minute artifacts. Trades around 09:45 remained near 24.93–24.99. On the single-exchange five-stock panel, no name has a five-minute return above 2 percent. JPM realized variance is 0.00138 versus 0.0477 on cleaned NBBO, and the 5×5 condition number is 39 versus about 931. The production Q1–Q4 code was not changed to obtain that result. The case study is in [`notebooks/data_cleaning_example.ipynb`](notebooks/data_cleaning_example.ipynb). The single-exchange five-stock panel is produced by `experiments/taq_five_stock_single_exchange_pilot.py`. The NBBO alternative remains in `experiments/taq_five_stock_pilot.py`.

## Synchronization and sampling

Intraday quotes do not arrive on a common clock across assets. After cleaning, covariance estimation constructs a synchronized midquote panel and only then forms returns.

**Previous-tick synchronization.** At a grid time $g$, each asset is assigned the most recent valid price observed at or before $g$. Future observations are never used for interpolation. If no prior observation is available, the synchronized value remains missing.

**Five-minute baseline.** The primary realized-covariance proxy is based on non-overlapping five-minute returns. This sampling frequency is retained as the baseline compromise between very fine sampling, where market-microstructure effects become more pronounced, and coarser sampling, where fewer intraday observations remain available for covariance estimation.

**Five-minute subsampling.** The second proxy reduces dependence on a single five-minute grid alignment. Starting from the synchronized one-minute price panel, the implementation constructs five shifted price grids with offsets $s\in\{0,1,2,3,4\}$. For each offset, every fifth price observation is selected before log returns are formed, and the existing realized-covariance function is then applied. The five covariance estimates are averaged as

```math
\mathrm{RCov}^{\mathrm{SS}}_t
= \frac{1}{5}\sum_{s=0}^{4}\mathrm{RCov}^{(s)}_t.
```

The averaging occurs across the five covariance estimates. Individual realized-covariance matrices are not divided by their numbers of intervals. The shifted estimates overlap and are not treated as independent measurements, so the construction does not imply a five-fold reduction in measurement-error variance. Missing observations are not handled through pairwise deletion, because doing so would allow different covariance entries to be estimated from different information sets.

**Epps-effect diagnostics.** Realized covariance and implied correlation are recomputed on previous-tick grids of 1, 2, 5, 10, 15, and 30 minutes from the same cleaned quotes. This is a diagnostic of the covariance proxy and synchronization procedure. It does not change the one-day-ahead forecast horizon, and no frequency is treated as the true covariance.

Opening, closing, early-close, and overnight conventions will be documented explicitly rather than embedded implicitly in the estimator.

---

## The Epps effect

A machine-learning pipeline can treat realized covariance as a label and treat a finer sampling grid as more data. That is not how the measurement object behaves.

Calendar-time realized covariance records co-movement only when two assets update inside the same sampling interval. Quotes do not arrive on a common clock. If two assets react to the same shock at slightly different observed times, a very fine grid can place those responses in different return intervals. The contemporaneous covariance contribution of that shock is then near zero. On a coarser grid, both responses can fall in the same interval, and the measured dependence can recover. Epps (1979) documented the resulting decline in measured correlation at short intervals. Barndorff-Nielsen, Hansen, Lunde, and Shephard (2011) treat nonsynchronous trading, together with microstructure noise, as a reason that simple calendar-time realized covariance is a noisy and potentially attenuated proxy, and they develop multivariate realized kernels as a later estimator. We keep unscaled calendar-time $\mathrm{RCov}$ as the baseline proxy and use the frequency scan as a diagnostic rather than as a search for a true matrix.

A one-shock sketch is enough for the evaluation implication. Asset A updates at 10:00:10 and asset B at 10:00:40. On a one-second grid those moves occupy different return bins, so their product does not enter that second's outer product. On a five-minute grid ending at 10:05 both moves occupy the same bin. Finer sampling therefore yields more return observations without automatically yielding a better covariance label. A forecasting model can appear to predict low correlations well because the realized-covariance proxy itself is attenuated.

The implemented diagnostic is `covharness.diagnostics.epps`. It reuses the production previous-tick synchronizer and the unscaled Gram-matrix realized covariance. For each frequency it reports the number of synchronized returns, the covariance and correlation matrices, off-diagonal correlation summaries, the individual pairwise correlations, eigenvalues, numerical rank, minimum eigenvalue, PSD status, and a condition number when the smallest eigenvalue is strictly positive. The average off-diagonal series is not forced to be monotonic.

On 13 February 2009, the five-stock single-exchange panel is a one-day demonstration, not an estimate of the magnitude of the Epps effect in U.S. equities. The later paper will rerun the diagnostic on the larger panel and report cross-day distributions. The figure below shows sampling interval against realized correlation. The heavy line is the average off-diagonal correlation. The light lines are the ten unique pairs.

![Realized correlations by sampling interval on 13 February 2009](results/taq_five_stock_epps_20090213.png)

The anomalous fine-frequency JPM results reported before the identity repair were not a Q1–Q4 failure on common stock. Root-only TAQ extraction mixed the blank/NULL-suffix JPM common share with preferred-share series that share `sym_root='JPM'` and NYSE `ex='N'`. Those preferred-share observations were valid quotes for different securities. They were not bad common-stock quotes. TAQ `sym_suffix` is therefore required when identifying the security. On this millisecond CQM table the ordinary share is stored as SQL NULL. After the extract was restricted to that identity, no nonblank JPM suffix remains in the common-stock stream.

The figure is the corrected one-day, five-name diagnostic. Average off-diagonal correlation is 0.629 at one minute, 0.587 at two minutes, 0.661 at five minutes, and 0.752 at thirty minutes. That pattern is not treated as a general estimate of the Epps effect in U.S. equities. Q1–Q4, previous-tick, sampling frequencies, and the universe were not changed. Derived tables are in [`results/taq_five_stock_epps_20090213.json`](results/taq_five_stock_epps_20090213.json), [`results/taq_jpm_one_minute_forensics_20090213.json`](results/taq_jpm_one_minute_forensics_20090213.json), and [`results/taq_jpm_provenance_closeout_20090213.json`](results/taq_jpm_provenance_closeout_20090213.json).

---

## Data access and provenance

The empirical implementation uses institutional market data obtained through WRDS. The 13 February 2009 five-stock sample (IBM, AAPL, MSFT, JPM, XOM) is a validation extract, not a multi-year dataset. The paper-style path uses `taqmsamp_all.cqm_20090213`. The labeled NBBO alternative uses `taqmsamp_all.nbbom_20090213`.

The intended architecture separates data acquisition from statistical estimation. WRDS retrieval lives under `experiments/`. Quote cleaning, P3 venue selection, synchronization, return construction, and realized-covariance estimation remain library stages. Persistent security identifiers will be preferred to ticker symbols where the available CRSP and TAQ linking resources permit this.

On 13 February 2009 the paper-style path uses CRSP listing venues mapped to TAQ `ex` codes. IBM, JPM, and XOM are NYSE (`N`). AAPL and MSFT are NASDAQ (`T`/`Q`). The production quote extract matches `sym_root`, listing-venue `ex`, and the ordinary-share `sym_suffix`. On this CQM table that suffix is SQL NULL. Blank strings are also accepted. `sym_suffix` is selected so identity remains auditable. The NBBO alternative extract remains `taqmsamp_all.nbbom_20090213`. Raw licensed market data are not committed. Derived diagnostics are in `results/taq_five_stock_single_exchange_20090213.json`, `results/taq_five_stock_pilot_20090213.json`, `results/taq_five_stock_epps_20090213.json`, `results/taq_five_stock_epps_20090213.png`, `results/taq_jpm_one_minute_forensics_20090213.json`, and `results/taq_jpm_provenance_closeout_20090213.json`.

---

## Statistical evaluation losses

The covariance target is latent. After the day has ended we still do not observe $\Sigma_{t+1\mid t}$. Evaluation uses a realized-covariance proxy $S$ in place of that matrix. With a conditionally unbiased proxy, $\mathrm{E}[S\mid\mathcal{F}]=\Sigma$, ranking forecasts by a robust loss agrees with ranking them against the latent target (Patton, 2011; Laurent, Rombouts, and Violante, 2013). The losses below are fixed before confirmatory evaluation.

**Squared Frobenius.** The adopted Frobenius ranking loss is the squared distance

```math
L_F(S,H)
=\|S-H\|_F^2
=\operatorname{tr}\bigl((S-H)^{\top}(S-H)\bigr)
=\sum_{ij}(S_{ij}-H_{ij})^2.
```

This is the matrix analogue of mean squared error. If $\mathrm{E}[S\mid\mathcal{F}]=\Sigma$, then

```math
\mathrm{E}\bigl[\|S-H\|_F^2\bigr]
=\|\Sigma-H\|_F^2
+\mathrm{E}\bigl[\|S-\Sigma\|_F^2\bigr].
```

The second term does not depend on the forecast. Taking the square root destroys that decomposition, so the ordinary Frobenius norm is not used for ranking. There is no scaling or annualization inside the loss. PSD is not required merely to compute it.

**Reduced multivariate QLIKE.** The primary ranking loss is

```math
L_Q(S,H)
=\log\det(H)+\operatorname{tr}(H^{-1}S).
```

$S$ must be square, finite, symmetric, and PSD. It may be singular. $H$ must be strictly PD. The implementation factorizes $H=LL^{\top}$ by Cholesky and uses that same factor for $\log\det(H)=2\sum_i\log L_{ii}$ and for the solves that compute $\operatorname{tr}(H^{-1}S)$. It does not form $H^{-1}$. A failed Cholesky is the positive-definiteness failure. The loss does not add jitter, clip eigenvalues, or otherwise repair a forecast.

**Full Stein.** When both $S$ and $H$ are SPD,

```math
L_S(S,H)
=\operatorname{tr}(H^{-1}S)-\log\det(H^{-1}S)-N
=L_Q(S,H)-\log\det(S)-N.
```

Reduced QLIKE and full Stein are not numerically equal. For a common SPD target they differ only by a target-only term, so model rankings and pairwise loss differentials agree. Full Stein rejects a singular $S$ because $\log\det(S)$ is undefined. A rank-deficient realized-covariance proxy does not by itself prevent evaluation with squared Frobenius or reduced QLIKE. Rank deficiency can still increase proxy noise and reduce inferential power.

**Variance-versus-correlation localization.** A separate diagnostic reports $\sum_i(S_{ii}-H_{ii})^2$ and $\|R(S)-R(H)\|_F^2$, where $R(A)=D(A)^{-1}AD(A)^{-1}$ and $D(A)=\mathrm{diag}(\sqrt{A_{ii}})$. These are descriptive. Correlation normalization is nonlinear, so an unbiased covariance proxy does not imply an unbiased correlation proxy. The diagnostics do not inherit the ranking-consistency guarantee of covariance-space Frobenius and QLIKE. A nonpositive diagonal is rejected rather than repaired.

The proxy-robustness construction $S=u\Sigma$ with $u\sim\mathrm{Exp}(1)$ is in `covharness.losses.robustness` and [`notebooks/proxy_robust_losses.ipynb`](notebooks/proxy_robust_losses.ipynb). $H_A=\Sigma$ ranks above the median-matched $H_B=\log(2)\Sigma$ in expected squared Frobenius, reduced QLIKE, and full Stein. Ordinary unsquared Frobenius ranks $H_B$ first. The figure is [`results/proxy_robust_losses.png`](results/proxy_robust_losses.png).

![Expected losses under a noisy unbiased proxy](results/proxy_robust_losses.png)

---

## Planned inference and confirmation

Differences in average out-of-sample loss are not treated as sufficient evidence of forecast superiority. Loss differentials can be serially dependent, and the comparison of many forecasting methods introduces a multiple-comparison problem.

The planned inferential layer includes

- Diebold-Mariano tests with HAC standard errors for pairwise forecast comparisons
- the Model Confidence Set for comparisons across the full model universe
- Giacomini-White tests for conditional and state-dependent differences in predictive ability
- Mincer-Zarnowitz diagnostics for forecast calibration

Synthetic experiments will be used before market-data inference to verify that the implementation behaves correctly in settings where the data-generating process is known. These experiments are intended to distinguish several evaluation failures that can otherwise be conflated, including serial dependence in loss differentials, multiple testing, and regime-dependent forecast performance.

Development and confirmation are separated chronologically. Model specifications, hyperparameters, information sets, and evaluation criteria are to be frozen before the locked confirmatory block is examined.

---

## Planned economic evaluation

A statistically more accurate covariance forecast need not produce a lower-risk portfolio. The benchmark therefore includes a separate global-minimum-variance evaluation. For a covariance forecast $\widehat{\Sigma}_t$, the unconstrained GMV weights are

```math
w_t
= \frac{\widehat{\Sigma}_t^{-1}\mathbf{1}}
{\mathbf{1}^{\top}\widehat{\Sigma}_t^{-1}\mathbf{1}}.
```

The planned analysis will compare realized portfolio variance and, in later stages, turnover and transaction costs. This channel is deliberately separate from covariance-space forecast loss because portfolio construction depends on the precision matrix $\widehat{\Sigma}_t^{-1}$. A method can improve a covariance estimate under one matrix loss while degrading the stability or economic usefulness of its inverse.

---

## Benchmark model universe

The initial benchmark is designed to span naive, realized-measure, shrinkage, dynamic-correlation, and shallow machine-learning approaches. The planned first-stage model set is

- random-walk realized covariance
- EWMA
- HAR-DRD
- HARQ-DRD
- Ledoit-Wolf linear shrinkage
- Ledoit-Wolf nonlinear shrinkage
- DCC
- DCC-NL
- Ridge-DRD

Later extensions include LSTM-BEKK, XGBoost-DRD, GHAR, and other covariance-specific or general multivariate forecasting architectures.

The comparison is intended to equalize opportunity rather than model complexity. Conventional models will receive the same validation period, configuration budget, and re-estimation cadence as newer methods. If a model is evaluated with a different information set, that distinction will be made explicit and analyzed rather than hidden within the implementation.

Each forecasting method is expected to conform to a common interface such as

```python
model.fit(train_data, ...)
forecast = model.forecast(...)
```

and to return an $N\times N$ one-step-ahead covariance forecast. Rolling estimation, evaluation losses, statistical inference, and portfolio analysis remain responsibilities of the common harness rather than individual model implementations.

---

## Repository structure

```text
covharness/
├── AGENTS.md
├── README.md
├── BENCHMARK_IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── environment.yml
├── docs/
│   ├── PROJECT_STATE.md
│   └── project_ledger.md
├── src/covharness/
│   ├── data/          # quote cleaning, P3 venue adapter, synchronization, returns
│   ├── realized/      # realized-covariance estimators
│   ├── simulation/
│   ├── models/
│   ├── losses/        # squared Frobenius, reduced QLIKE, full Stein
│   ├── inference/
│   ├── portfolio/
│   ├── protocol/
│   ├── diagnostics/
│   └── utils/
├── tests/unit/
├── experiments/
├── notebooks/
└── results/
```

Reusable implementation belongs under `src/covharness/`. Notebooks may be used for exploratory analysis and figures, but core statistical procedures should not exist only in notebooks.

The project conda environment is `covharness`. Recreate it with `conda env create -f environment.yml`, then `pip install -e ".[dev]"` from that environment.

---

## Current implementation

The following components are implemented and unit-tested

- Q1–Q4 quote cleaning adapted from Barndorff-Nielsen et al. (2011), Section 5.1
- paper-style single-exchange (P3) adapter that reuses that cleaner and retains the requested TAQ suffix identity
- consolidated NBBO as a labeled alternative input to the same cleaner
- previous-tick synchronization
- synchronized log returns
- standard daily realized covariance $\mathrm{RCov}_t = R_t^{\top}R_t$
- five-minute subsampled realized covariance
- Epps-effect frequency scan on previous-tick realized covariance and implied correlation
- squared Frobenius loss
- reduced multivariate QLIKE
- full Stein loss when the proxy is SPD
- descriptive variance-versus-correlation localization diagnostics
- a fixed-seed proxy-robustness demonstration

A five-stock panel on 13 February 2009 has been constructed on the single-exchange path. The same day remains available on the NBBO path. The Epps diagnostic has been run on the identity-corrected single-exchange panel. Block 1 measurement work is closed. Realized kernels are not implemented.

Forecasting models, the leak-proof temporal protocol, confirmatory inference, and portfolio evaluation remain planned work.

See [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for the exact implementation checkpoint and test history.

---

## Research integrity

The benchmark is governed by the following rules

- preserve chronological training, validation, and confirmation ordering
- fit preprocessing and scaling parameters only on information available at the time of estimation
- exclude future observations from feature construction
- apply comparable tuning budgets and re-estimation schedules across model classes
- use explicit random seeds and report seed distributions rather than selected favorable runs
- do not tune on the locked confirmatory block
- retain failed model configurations and report implementation failures rather than silently removing them
- do not repair invalid covariance matrices without recording the failure and the repair procedure
- fix evaluation proxies and losses before confirmation
- base confirmatory claims on the pre-specified inferential procedures rather than on rankings of average loss alone

---

## References

- Andersen, T. G., & Bollerslev, T. (1998). *Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts.* International Economic Review.
- Epps, T. W. (1979). *Comovements in stock prices in the very short run.* Journal of the American Statistical Association, 74(366), 291–298.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2009). *Realised kernels in practice: trades and quotes.* The Econometrics Journal, 12(3), C1–C32.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2011). *Multivariate realised kernels: consistent positive semi-definite estimators of the covariation of equity prices with noise and non-synchronous trading.* Journal of Econometrics, 162(2), 149–169.
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics.
- Engle, R. F., & Colacito, R. (2006). *Testing and Valuing Dynamic Correlations for Asset Allocation.* Journal of Business & Economic Statistics.
- Engle, R. F., Ledoit, O., & Wolf, M. (2019). *Large Dynamic Covariance Matrices.* Journal of Business & Economic Statistics.
- Giacomini, R., & White, H. (2006). *Tests of Conditional Predictive Ability.* Econometrica.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). *The Model Confidence Set.* Econometrica.
- Laurent, S., Rombouts, J. V. K., & Violante, F. (2013). *On Loss Functions and Ranking Forecasting Performances of Multivariate Volatility Models.* Journal of Applied Econometrics.
- Patton, A. J. (2011). *Volatility Forecast Comparison Using Imperfect Volatility Proxies.* Journal of Econometrics.

---

## License and data

MIT License. Raw market data may be subject to vendor licensing restrictions and are not committed to the repository.
