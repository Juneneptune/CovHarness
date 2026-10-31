# Binance CONFIRM candidate protocol

This document is the candidate preregistration for one Binance confirmation run. It is not `PREREGISTRATION.md`. CONFIRM stays locked until a separate authorization. The user decides whether this candidate becomes the preregistration.

The register commit is `2186809e3e80bba7686b6bb2467dfff1d2b35a4b`. That commit records amendments R-2026-10-30b and R-2026-10-30c. Earlier register entries are unchanged. This protocol assembles those entries with the frozen Binance design. It does not reopen them.

The scope is the five-asset Binance panel. The equity path, the N=30 plan, and the graph-neural model are outside this run.

## Locked identifiers

Production panel `data/processed/binance_five_asset_panel.npz`. SHA-256 `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`. The file is not rewritten.

Configuration files that define roster models.

| File | SHA-256 | Roster members it defines |
| --- | --- | --- |
| `configs/binance_open_data_core.yaml` | `8d63eacacf13a876651f9c4eb5399a8527d8458278974ff069dd0701dd5cbf07` | RW01, EWMA01, HARDRD01, HARQDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, LSTM19, and the original DCC01 and DCCNL01 grids |
| `configs/binance_open_data_screen.yaml` | `b07011a798c36fc25e117b69d71f6c8eb5eb1ed756e97c39c19e29542af6203c` | SCREEN execution record of the core selections |
| `configs/binance_post_screen_d1.yaml` | `f6dc7b7118d8e2f89357e6cba32b6adf302d17e9028ae2b6abfa23f1c0af6e41` | EWMA21 and LSTM22 inside the D1 grid |

DCC02, DCCNL02, DCC03, DCCNL03, SHARDRD01, BEKKS01, LSTMBEKK-noC, V10, V01, and V11 are defined in the implementation and in the amendment register. They do not have a separate configuration file. DCC01 and DCCNL01 are not forecast.

B1b cutoffs are stored in `results/post_screen/B1b/b1b_cutoffs.json`. SHA-256 `26b68025d4186654759f2e2fc85d3f26eedcbda9868848fdbc7865c38a510715`. The ordinary cutoff is 2.0595385527532972. The size-adjusted cutoff at block length 7 is 2.4728093597276533. The size-adjusted cutoff at block length 29 is 2.4226501420943998. These values are not recomputed.

SCREEN selection under the frozen rank rule is `results/post_screen/G1/g1_selection.json`. SHA-256 `633f528a3f75e2f46096e66e135e0a963b8e2c019df226ee3c74c669d51be3ee`. Result commit `0d7f8beb0473078cc41c1c4b1ab20b640033bcda`.

## Calendar and information set

Assets, in order, are BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, and ADAUSDT. Dimension N is 5.

| Segment | Dates |
| --- | --- |
| HISTORY_A | 2021-11-09 through 2022-07-16 |
| VALIDATION | 2022-07-17 through 2023-03-23 |
| Halt | 2023-03-24, not a statistical day |
| HISTORY_B | 2023-03-25 through 2023-11-29 |
| SCREEN | 2023-11-30 through 2025-04-12 |
| CONFIRM | 2025-04-13 through 2026-08-25 |

CONFIRM has 500 targets. The first origin is 2025-04-12. The estimation window is the last 250 SCREEN dates. Every window has m equal to 250 admissible dates and excludes the target. The first CONFIRM target is a refit. Later refits occur every 21 origins. The counter resets at the start of CONFIRM. No fitted SCREEN object is carried into the block.

HISTORY_A and VALIDATION are not concatenated with the confirmatory calendar. No origin, window, state, or feature crosses the halt. Every origin precedes its target by one UTC day. Preprocessing scalers are fit on the estimation window only.

## What authorization builds

Authorization is a separate act. This candidate does not authorize it.

After authorization, and not before, the run builds two CONFIRM-dated objects with the frozen measurement code.

The robustness proxy is unscaled 15-minute realized covariance on each CONFIRM date. The interval is 15. The anchor rule is the existing Binance rule. There is no forward fill. The production panel is not rewritten.

The semivariance features are the positive and negative realized semivariances on each CONFIRM date. They are computed from the primary five-minute returns under the C4 rule. A zero return is in neither semivariance. These features are origin-dated. Target-day semivariances are not model inputs.

The primary evaluation proxy remains the unscaled five-minute realized covariance already stored in the production panel. It is not recomputed. It is not read before authorization.

## What authorization forecasts

Each roster model is forecast once. The identifiers are RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, DCC02, DCCNL02, DCC03, DCCNL03, BEKKS01, LSTMBEKK-noC, LSTM19, LSTM22, V10, V01, and V11.

Stochastic models use seeds 0, 1, 2, 3, and 4. The reported forecast is the equal-weight mean of the five seed covariance matrices. All five seeds are retained. The best seed is not selected.

The forecast is H at the target given the origin. It is scored against the target-day proxy. Evaluation does not repair a forecast. A numerical repair required by a model records the failure, the repair method, and the diagnostics.

## Frozen comparison

The headline pair is HARQDRD01 against V01. HARQDRD01 is the econometric finalist. V01 is the deep tournament winner. The outcome of amendment R-2026-10-30c is accepted.

The secondary contrasts are V01 minus V00 and V11 minus V00. V00 is LSTM22. Both contrasts are reported, including the contrast that contains the deep winner.

LSTM19 and EWMA01 stay in the forecast tables as original-grid comparisons. They do not occupy the headline slots. RIDGE01 and XGB08 stay machine-learning controls. DCC03 and DCCNL03 stay robustness fits. LSTMBEKK-noC stays an ablation.

## Losses and proxies

Reduced QLIKE is the primary loss. Squared Frobenius is the secondary loss. They are reported separately and are never pooled. The differential is the loss of the first named model minus the loss of the second. A negative mean favours the first model.

Five-minute realized covariance is the primary proxy. Fifteen-minute realized covariance is the robustness proxy. The two proxies are reported separately. A conclusion is called proxy-robust only when it holds under both.

## Pairwise claim

The confirmatory pairwise procedure is P5. The statistic is the Diebold–Mariano mean divided by the square root of the equal-weighted cosine long-run variance over T. The bandwidth nu is the floor of 0.4 times T to the power 2/3. At T equal to 500, nu is 25.

A confirmatory pairwise claim on the headline requires the absolute statistic to exceed all three cutoffs. Those cutoffs are 2.0595385527532972, 2.4728093597276533, and 2.4226501420943998. The win rate, the signs of the four chronological 125-day block means, and the cumulative-loss plot are always reported. They do not replace the test.

The same three-cutoff rule applies to each secondary contrast. Holm is the step-down correction on the two ordinary two-sided equal-weighted-cosine p-values at familywise level 0.05. The smaller p-value is compared with 0.025 and the larger with 0.05. A secondary claim also requires the absolute statistic to exceed both size-adjusted cutoffs. The size-adjusted cutoffs are not themselves Holm-adjusted.

## Multiple-model description

SPA and MCS on CONFIRM use mean block length 29, which is the floor of 1.3 times the square root of 500. Block length 7 is a sensitivity. Both use 5000 bootstrap draws and seed 20260913.

SPA uses benchmark HARDRD01 and consistent recentering. An exact-zero differential against that benchmark is an exact tie. It stays in the universe and is excluded from studentization. A constant nonzero differential is not jittered.

MCS uses the range statistic and the range elimination rule as the primary pair, and the max statistic with the max elimination rule as the companion, at alpha 0.10. The pairs are not crossed. The universe is every roster model with a finite raw loss on all 500 CONFIRM targets. An incomplete column is retained with its diagnostics and is not dropped from the record. SPA and MCS do not change the headline pair.

## Secondary diagnostics

Giacomini–White, pooled Mincer–Zarnowitz, and Giacomini–Rossi fluctuation are run for the headline pair only. They are secondary diagnostics. They cannot overturn or create the headline claim.

Giacomini–White uses the two frozen instrument sets separately. The market-state set is a constant, the log of the mean diagonal of origin-day realized covariance, and the mean pairwise correlation of that matrix. The measurement-stress set is a constant, the log of aggregate realized quarticity, and the Barndorff–Nielsen–Shephard market-jump indicator at level 0.01. The family size is 2. The adjusted p-value is the minimum of 1 and 2 times the raw p-value. The per-test cutoff is 0.025. The six columns are not stacked. Target-day matrices and full-CONFIRM quantiles are not instruments.

Pooled Mincer–Zarnowitz is the common-coefficient pooled-vech regression. Ordinary least squares and the approximate Patton–Sheppard weighting are both reported. The calibration null is intercept 0 and slope 1.

Giacomini–Rossi fluctuation uses window fraction mu equal to 0.30. The two-sided critical value is 3.012. The test rejects when the maximum absolute local statistic exceeds 3.012. The sign of the path is descriptive.

## Economic channel

The primary economic comparison is the E1 squared-return gap. Portfolio weights are the unconstrained global-minimum-variance weights of the raw covariance forecast. The gap is the squared simple return of HARQDRD01 minus the squared simple return of V01, after turnover costs. Costs of 5 and 10 basis points are both reported. The cost is subtracted from the simple return. The first turnover is zero because the book opens in the first weights. The same gap is reported for each secondary contrast. Long-only and equal-weight books are robustness comparisons. This risk comparison uses P5 and the three cutoffs.

The Fleming–Kirby–Ostdiek performance fee is additional. It does not replace the squared-return gap. It is not a substitute chosen because a SCREEN risk test failed to reject. Gamma is reported at 1, 5, and 10. The cost is reported at 5 and 10 basis points. All six cells are reported. Neither gamma nor the cost is selected later.

The fee uses the unconstrained global-minimum-variance simple returns after the stated turnover cost. No return forecast is added. For returns r_A and r_B, the daily fee Delta solves the average-utility equality

```math
\frac{1}{T}\sum_t\left[(r_{A,t}-\Delta)-\frac{\gamma}{2(1+\gamma)}(r_{A,t}-\Delta)^2\right]
=
\frac{1}{T}\sum_t\left[r_{B,t}-\frac{\gamma}{2(1+\gamma)}r_{B,t}^2\right].
```

On the headline, A is HARQDRD01 and B is V01. On a secondary contrast, A is the first named model and B is LSTM22. A positive Delta means the investor would pay that daily amount to hold A rather than B. If the quadratic has two real roots, the reported root is the one closer to the mean-variance expression (mean of A minus mean of B) minus gamma over 2 times (variance of A minus variance of B). The daily fee and the same fee multiplied by 365 are both stored. The year length matches the E1 crypto calendar. The scale is not selected.

## Other reporting rules

The headline uses raw forecasts. The C3 maps R1 and R2 are computed and reported as diagnostics only.

Carry-forward of the last admissible parameters is the primary estimation-failure policy. The interior parameter map is the robustness policy. DCC03 and DCCNL03 are the interior robustness columns on this roster.

If HARQDRD01 uses an estimation-window-mean fallback on any CONFIRM target, the headline pair is also reported with those targets removed. The primary table keeps the fallback targets.

## Holdout opening

The holdout is one authorized run. There is no tuning on CONFIRM. The roster, the losses, the proxies, the cutoffs, the pairs, and the refit schedule are not changed after the unlock. A later departure is recorded as a deviation. It is not edited into the preregistration.

## Deferred

D3b, D4, the graph-neural slot, GHAR, and F1 are outside this confirmatory roster. The Binance run does not wait on them. BEKKS01-INT has a complete SCREEN loss column and is not on the CONFIRM forecast roster.

## What the authorized run tests

The run forecasts the roster, scores both losses on both proxies, and then reports the following. It does not select a new finalist.

The headline P5 tests on raw reduced QLIKE and raw squared Frobenius, under the five-minute proxy, at all three cutoffs, with the win rate, the four block signs, and the cumulative-loss plot.

The same headline tests under the 15-minute proxy, reported separately.

The two secondary contrasts under the same pairwise rule, with the Holm correction on the ordinary p-values.

SPA and MCS at block lengths 29 and 7, on both losses, for the complete raw columns.

Giacomini–White, pooled Mincer–Zarnowitz, and Giacomini–Rossi fluctuation for the headline pair.

The E1 risk comparison and the six performance-fee cells for the headline pair and the two secondary contrasts.

The HARQ fallback sensitivity when a fallback target occurs.

The R1 and R2 diagnostics, labeled as diagnostics.
