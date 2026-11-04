# CovHarness

**A leak-free benchmark for forecasting covariance matrices of asset returns, built to test whether new methods, deep learning included, really beat strong econometric baselines.**

Covariance forecasts are used in portfolio construction and risk management, and a growing literature reports that neural networks produce better ones. Many of those claims rest on weak baselines, losses that break when the volatility measurement is noisy, and no formal test. CovHarness is an evaluation harness that makes the comparison fair and repeatable on any dataset. Every model is scored against the same target, with the same estimation windows, refit schedule, tuning budget and scoring rules, and the final comparison happens once, on a holdout that was locked before any model was scored on it.

## How an experiment works

1. **Measure.** Daily realized covariance is built from intraday prices. The harness has measurement paths for exchange bar data and for consolidated quote data, and the same downstream code serves both.
2. **Split time.** The sample is cut into an estimation history, a tuning period, a selection period and a holdout, in that order. Nothing is shuffled, and the code refuses to load holdout dates unless they are explicitly unlocked.
3. **Tune.** Each model family chooses its settings on the tuning period, with the same rolling window, the same refit schedule and a capped search budget for every family.
4. **Select.** Tuned models are compared on the selection period, and the finalists are chosen by a written selection rule.
5. **Preregister.** The holdout test, including the loss, the statistical test and its critical values, is committed to the repository before the holdout is opened.
6. **Confirm once.** The holdout is run a single time. Robustness checks and crash-robust views are reported next to the headline, whatever they show.

## Scoring a forecast that is never observed

Tomorrow's conditional covariance matrix $\Sigma_{t+1\mid t}$ is latent, even after the fact. Forecasts are scored against realized covariance, which is an unbiased but noisy stand-in. That rules out many common losses, because with a noisy target some of them rank a worse forecast first. The harness uses the losses that stay reliable under that noise (Patton, 2011; Laurent, Rombouts and Violante, 2013). The primary one is QLIKE, the multivariate Gaussian likelihood loss,

```math
L(S, H) = \log\det H + \mathrm{tr}(H^{-1} S),
```

where $H$ is the forecast and $S$ the realized covariance. Squared Frobenius distance is the secondary loss. Results are recomputed at more than one sampling frequency to check that rankings do not depend on how the target was measured, and subsampled and kernel estimators (Barndorff-Nielsen et al., 2011) are being added.

## Statistical inference

Daily loss differences are fat-tailed and autocorrelated, so standard tests often declare a winner when there is none. The harness provides four kinds of tests.

- **Pairwise comparison.** Diebold–Mariano tests (Diebold and Mariano, 1995) with HAC (Newey and West, 1987) and equal-weighted cosine (Lazarus et al., 2018) variance estimators, plus stationary-bootstrap calibration of critical values (Politis and Romano, 1994) so the false-positive rate is what it claims to be.
- **Many models at once.** The SPA test (Hansen, 2005) and the Model Confidence Set (Hansen, Lunde and Nason, 2011).
- **When and why a model wins.** Conditional predictive ability tests (Giacomini and White, 2006), the fluctuation test (Giacomini and Rossi, 2010) and Mincer–Zarnowitz calibration (Mincer and Zarnowitz, 1969; Patton and Sheppard, 2009).
- **Power.** Simulation of how likely a planned holdout test is to detect a given effect, run before the holdout is opened.

## Economic evaluation

Forecasts are also judged by the portfolios they produce. The harness builds global-minimum-variance portfolios from each forecast and reports realized risk, turnover, transaction costs and a utility-based performance fee (Fleming, Kirby and Ostdiek, 2001). Long-only, risk-parity, volatility-targeted and tail-risk (VaR and expected shortfall) evaluations are in development.

## Model library

| Family | Models | Input | References |
| --- | --- | --- | --- |
| Naive | random walk, exponential smoothing (EWMA) | realized covariance | |
| HAR | HAR-DRD, HARQ-DRD, SHAR-DRD, ridge HAR | realized covariance | Corsi (2009); Bollerslev, Patton and Quaedvlieg (2016, 2018); Patton and Sheppard (2015) |
| Machine learning | XGBoost on HAR features | realized covariance | Chen and Guestrin (2016) |
| Shrinkage | Ledoit–Wolf linear and nonlinear | daily returns | Ledoit and Wolf (2004, 2020) |
| GARCH family | DCC, DCC-NL, BEKK | daily returns | Engle (2002); Engle, Ledoit and Wolf (2019); Engle and Kroner (1995) |
| Deep learning | LSTM-BEKK, with variants that add realized-covariance inputs, a realized-covariance training target, or both, and an ablation without the LSTM | daily returns and realized covariance | Hochreiter and Schmidhuber (1997); Wang, Liu, Tran and Wang (2025) |

New models implement a common interface and declare which data they use. The harness handles estimation windows, refits, the holdout lock and scoring, so a model never sees its target or computes its own score. See [docs/MODEL_IMPLEMENTATION.md](docs/MODEL_IMPLEMENTATION.md).

## Datasets

- **Open crypto data.** One-minute spot prices from Binance's public archive. It is freely downloadable, so anyone can reproduce the example study below.
- **U.S. equities.** A quote-based measurement path (TAQ) with venue filtering and previous-tick synchronization, validated on a one-day extract. The full study is pending data access.
- **Larger universes.** An extension to 10 to 30 assets selected point-in-time by trading volume is in development, with its own locked holdout.

## Example study on open crypto data

The first complete run uses five liquid crypto assets (BTC, ETH, BNB, LTC and ADA) from 2021-11-09 to 2026-08-25, with a 500-day holdout from April 2025 to August 2026. The holdout test was fixed in advance in [PREREGISTRATION.md](PREREGISTRATION.md).

An LSTM-BEKK trained on realized covariance beat the preselected econometric benchmark, HARQ-DRD, on the primary loss. That benchmark did poorly on the holdout overall, ranking 15th of 21. Plain scalar BEKK and the LSTM-BEKK with its LSTM removed ranked first and second on the full sample, ahead of the realized-covariance LSTM-BEKK in third, but that ordering comes from the crash day discussed below. Without that day, and by median loss, the realized-covariance LSTM-BEKK ranks fourth and both BEKK variants fall to eighth or lower.

The test statistic is a Diebold–Mariano statistic on the daily QLIKE gaps, using the equal-weighted cosine variance estimator, and positive values favour the LSTM-BEKK. Under textbook theory it follows a t-distribution with 25 degrees of freedom, which gives a 5% critical value of 2.06. On realistic resampled data that cutoff produced false positives about 8% of the time, so before the holdout was opened we also calibrated stricter critical values on the selection period. We shifted its daily loss gaps to have a true mean of zero, resampled them in blocks that keep their streaks and fat tails, and took the value the statistic exceeds only 5% of the time, which is 2.47 with 7-day blocks and 2.42 with 29-day blocks. A result counts as significant only if it clears all three, so 2.47 is the bar that matters. All three values are fixed in [PREREGISTRATION.md](PREREGISTRATION.md).

| HARQ-DRD vs. LSTM-BEKK (realized-covariance target) | Test statistic | Clears the strict bar of 2.47? |
| --- | ---: | :---: |
| QLIKE on 5-minute realized covariance (primary, preregistered) | 2.65 | Yes |
| Same, excluding the 2025-10-10 liquidation day (exploratory) | 2.40 | No |
| Squared Frobenius loss (secondary) | 1.59 | No |
| QLIKE on 15-minute realized covariance (robustness) | 2.39 | No |

The result clears the bar but is fragile. One liquidation day carries much of the evidence. HARQ-DRD also fell back to its window-average forecast on 16 holdout days, 15 of them between 2025-10-11 and 2025-11-01, and excluding those days raises the statistic to 2.82. On typical days simple exponential smoothing ranked first of all 21 models, and model rankings shifted between the selection period and the holdout. Portfolios built from the two finalists had statistically indistinguishable realized risk, while the HARQ-DRD portfolio traded about twice as much ([turnover](results/exploratory/holdout_turnover/README.md)). The full report is in [results/confirm/REPORT.md](results/confirm/REPORT.md), and the crash-day analysis is in [results/exploratory/X3/REPORT.md](results/exploratory/X3/REPORT.md).

## Running it

```bash
conda env create -f environment.yml
conda activate covharness
python -m pytest -q
python scripts/synthetic_benchmark_demo.py
```

The demo runs the model roster on a small simulated panel and needs no market data. Reproducing the crypto study requires the Binance archive, described in [docs/BINANCE_OPEN_DATA_PANEL.md](docs/BINANCE_OPEN_DATA_PANEL.md). Large data files are not committed.

## Repository layout

```text
src/covharness/   measurement, models, losses, inference, portfolio and protocol code
tests/            unit tests, including simulation checks against known truth
scripts/          entry points for each experiment
results/          reports and summary tables
docs/             detailed protocol, measurement and inference documentation
```

## Further reading

- [PREREGISTRATION.md](PREREGISTRATION.md) is the locked holdout protocol for the crypto study.
- [docs/GLOSSARY.md](docs/GLOSSARY.md) maps internal IDs to plain names.

## References

- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2011). *Multivariate realised kernels: consistent positive semi-definite estimators of the covariation of equity prices with noise and non-synchronous trading.* Journal of Econometrics, 162(2), 149–169.
- Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). *Exploiting the errors: A simple approach for improved volatility forecasting.* Journal of Econometrics.
- Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2018). *Modeling and forecasting (un)reliable realized covariances for more reliable financial decisions.* Journal of Econometrics.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* Proceedings of the 22nd ACM SIGKDD Conference.
- Corsi, F. (2009). *A simple approximate long-memory model of realized volatility.* Journal of Financial Econometrics.
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy.* Journal of Business & Economic Statistics.
- Engle, R. F. (2002). *Dynamic conditional correlation.* Journal of Business & Economic Statistics.
- Engle, R. F., & Kroner, K. F. (1995). *Multivariate simultaneous generalized ARCH.* Econometric Theory.
- Engle, R. F., Ledoit, O., & Wolf, M. (2019). *Large dynamic covariance matrices.* Journal of Business & Economic Statistics.
- Fleming, J., Kirby, C., & Ostdiek, B. (2001). *The economic value of volatility timing.* Journal of Finance.
- Giacomini, R., & Rossi, B. (2010). *Forecast comparisons in unstable environments.* Journal of Applied Econometrics.
- Giacomini, R., & White, H. (2006). *Tests of conditional predictive ability.* Econometrica.
- Hansen, P. R. (2005). *A test for superior predictive ability.* Journal of Business & Economic Statistics.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). *The model confidence set.* Econometrica.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation.
- Laurent, S., Rombouts, J. V. K., & Violante, F. (2013). *On loss functions and ranking forecasting performances of multivariate volatility models.* Journal of Applied Econometrics.
- Lazarus, E., Lewis, D. J., Stock, J. H., & Watson, M. W. (2018). *HAR inference: Recommendations for practice.* Journal of Business & Economic Statistics.
- Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices.* Journal of Multivariate Analysis.
- Ledoit, O., & Wolf, M. (2020). *Analytical nonlinear shrinkage of large-dimensional covariance matrices.* Annals of Statistics.
- Mincer, J., & Zarnowitz, V. (1969). *The evaluation of economic forecasts.* In J. Mincer (Ed.), *Economic Forecasts and Expectations.* NBER.
- Newey, W. K., & West, K. D. (1987). *A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.* Econometrica.
- Patton, A. J. (2011). *Volatility forecast comparison using imperfect volatility proxies.* Journal of Econometrics.
- Patton, A. J., & Sheppard, K. (2009). *Evaluating volatility and correlation forecasts.* In T. G. Andersen et al. (Eds.), *Handbook of Financial Time Series.* Springer.
- Patton, A. J., & Sheppard, K. (2015). *Good volatility, bad volatility: Signed jumps and the persistence of volatility.* Review of Economics and Statistics.
- Politis, D. N., & Romano, J. P. (1994). *The stationary bootstrap.* Journal of the American Statistical Association.
- Wang, H., Liu, C., Tran, M.-N., & Wang, C. (2025). *Deep learning enhanced multivariate GARCH.* arXiv:2506.02796.

## License

MIT. Raw market data is not redistributed.
