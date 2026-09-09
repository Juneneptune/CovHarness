# covharness

`covharness` is a research implementation for the evaluation of one-day-ahead multivariate covariance forecasts. Its purpose is to provide a common measurement, estimation, and evaluation framework in which econometric, machine-learning, and deep-learning methods can be compared under the same empirical protocol.

The benchmark is designed so that a new forecasting method does not receive a different target, a richer information set, a larger tuning budget, or a more favorable evaluation criterion simply because it belongs to a different modeling tradition. Conventional models are therefore treated as serious competitors rather than default baselines. The planned comparison assigns common validation periods, tuning budgets, re-estimation schedules, and confirmatory procedures across model classes. Stochastic methods will be evaluated over pre-specified seed sets rather than selected ex post from favorable individual runs.

> **Current status.** The measurement layer now includes high-frequency NBBO quote cleaning, previous-tick synchronization, synchronized log returns, standard daily realized covariance, and five-minute subsampled realized covariance. Cleaning is applied before synchronization. The next task is to rerun the five-asset WRDS/TAQ pilot through the production cleaning code and then compute the first cleaned real-data realized covariance. Realized-kernel estimation is not implemented. Forecasting models and confirmatory inference are not yet implemented, and the repository makes no claim that deep learning outperforms conventional econometric methods.

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

An artificial return $r_{i,j}$ contributes $r_{i,j}^{2}$ on the variance diagonal and $r_{i,j} r_{k,j}$ to every covariance involving asset $i$. One contaminated interval can therefore distort an entire row and column of the daily proxy. The measurement pipeline is therefore

```math
\text{raw NBBO}
\to
\text{quote cleaning}
\to
\text{midquote}
\to
\text{previous-tick synchronization}
\to
\text{log returns}
\to
\text{realized covariance}.
```

The implemented procedure is adapted from Section 5.1 of Barndorff-Nielsen, Hansen, Lunde, and Shephard (2011), who follow the more detailed trade-and-quote cleaning discussion in Barndorff-Nielsen, Hansen, Lunde, and Shephard (2009). We apply the following quote rules to consolidated NBBO fields `best_bid` and `best_ask`.

P1. Delete observations outside regular exchange hours, 09:30–16:00 inclusive on the caller-supplied exchange-local clock.

P2. Delete observations with non-positive or non-finite best bid or best ask.

P3 is not applied. The 2011 paper retains quotes from a selected single exchange. Our current pipeline uses consolidated NBBO, so the implementation is an adaptation of that procedure rather than an exact replication of its exchange-selection rule.

Q1. If several quotes for the same security have the same timestamp, replace them by one observation using the median bid and the median ask.

Q2. Delete quotes with a negative spread, $\mathrm{ask}-\mathrm{bid}<0$. Locked quotes with zero spread are retained.

Q3. For each stock-day, compute the median spread after Q1 and Q2, and delete quotes whose spread exceeds ten times that median.

Q4. For each stock-day quote with a complete centered neighborhood, take the 25 observations immediately before and the 25 immediately after the quote, excluding the quote itself. Let $N_i$ be those 50 neighboring midquotes, $\widetilde{m}_i=\mathrm{median}(N_i)$, and $D_i$ the mean absolute deviation of $N_i$ from $\widetilde{m}_i$. Delete the quote if $|\mathrm{mid}_i-\widetilde{m}_i|>10 D_i$. If $D_i=0$, retain the center only when it equals $\widetilde{m}_i$. This $D_i$ is a mean absolute deviation from the local median, not the usual median-absolute-deviation statistic.

Q4 is a symmetric ex-post filter on historical measurement data. It uses later same-day quotes by design and is not a forecasting feature. Neighborhoods are formed within a stock-day and never cross assets or calendar days. By default the first and last 25 observations of a stock-day are retained because a complete 50-neighbor window is unavailable. After the filters, the midquote is $(\mathrm{best\ bid}+\mathrm{best\ ask})/2$.

Q3 and Q4 address different failures. A wide quote is visible in the spread. A narrow but displaced quote, such as a bid and ask near 22.6 while neighboring midquotes remain near 25, can survive a spread filter and still contaminate previous-tick sampling. The empirical case is documented in [`notebooks/data_cleaning_example.ipynb`](notebooks/data_cleaning_example.ipynb), where raw JPM NBBO midquotes on 13 February 2009 produced implausible five-minute moves while the trade tape remained near 25. Synchronization was behaving correctly. The input quotes were not.

The production implementation is `covharness.data.quotes.clean_nbbo_quotes`. It returns the cleaned quotes and per-stage row counts, including input size, rows remaining after Q1–Q4, rows removed at each stage, and the overall removal fraction. Sale-condition filters for the trade tape are not implemented.

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

**Epps-effect diagnostics.** Realized correlations will be recomputed at several sampling frequencies to assess the sensitivity of measured dependence to intraday sampling. This is a diagnostic of the covariance proxy and synchronization procedure; it does not change the one-day-ahead forecast horizon.

Opening, closing, early-close, and overnight conventions will be documented explicitly rather than embedded implicitly in the estimator.

---

## Data access and provenance

The empirical implementation will use institutional market data obtained through WRDS. Access has been established, but the available products, tables, and field definitions are being verified before extraction begins.

The intended architecture separates data acquisition from statistical estimation. WRDS-specific code will retrieve and normalize intraday records into a canonical internal representation. Quote cleaning, synchronization, return construction, and realized-covariance estimation remain separate stages. Persistent security identifiers will be preferred to ticker symbols where the available CRSP and TAQ linking resources permit this.

The first real-data exercise is a small TAQ pilot used to validate the measurement pipeline before any multi-year extraction is attempted. Raw licensed market data remain private and are not committed to the repository. The JPM quote anomaly that motivated production cleaning is documented in [`notebooks/data_cleaning_example.ipynb`](notebooks/data_cleaning_example.ipynb).

---

## Planned statistical losses

Because realized covariance is a noisy proxy, the evaluation design does not rely on a single arbitrary matrix distance. The planned primary criterion is multivariate QLIKE / Stein loss, with squared Frobenius loss reported as a complementary measure. The losses will be fixed before confirmatory evaluation rather than selected after observing model rankings.

This choice follows the broader principle that forecast rankings should be robust to the use of an imperfect volatility proxy whenever the required assumptions hold. The project therefore distinguishes the forecasting target from the proxy used to evaluate it and avoids interpreting one convenient loss as universally sufficient.

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
├── pyproject.toml
├── environment.yml
├── docs/
│   ├── PROJECT_STATE.md
│   └── project_ledger.md
├── src/covharness/
│   ├── data/          # quote cleaning, synchronization, and return construction
│   ├── realized/      # realized-covariance estimators
│   ├── simulation/
│   ├── models/
│   ├── losses/
│   ├── inference/
│   ├── portfolio/
│   ├── protocol/
│   ├── diagnostics/
│   └── utils/
├── tests/unit/
├── experiments/
└── results/
```

Reusable implementation belongs under `src/covharness/`. Notebooks may be used for exploratory analysis and figures, but core statistical procedures should not exist only in notebooks.

The project conda environment is `covharness`. Recreate it with `conda env create -f environment.yml`, then `pip install -e ".[dev]"` from that environment.

---

## Current implementation

The following components are implemented and unit-tested

- high-frequency NBBO quote cleaning adapted from Barndorff-Nielsen et al. (2011), Section 5.1
- previous-tick synchronization
- synchronized log returns
- standard daily realized covariance $\mathrm{RCov}_t = R_t^{\top}R_t$
- five-minute subsampled realized covariance

Cleaning is a required stage before previous-tick sampling. The next measurement task is to rerun the five-asset real-data pilot through this production cleaner and then form the first cleaned realized-covariance matrix. Epps-effect and matrix-quality diagnostics follow that cleaned pilot. Realized kernels are not implemented.

Forecasting models, the statistical-loss layer, confirmatory inference, and portfolio evaluation remain planned work.

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
