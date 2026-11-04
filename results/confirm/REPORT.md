# Binance CONFIRM report

Model identifiers are defined in [docs/GLOSSARY.md](../../docs/GLOSSARY.md). Every number below is the saved confirmatory value. This is a reformatted version of the report produced by the run in commit `df44a11`, with one printed p-value corrected in `4c2a879`.

## 1 Headline

The preregistered rule rejects equal predictive accuracy, and the direction favours V01.

HARQDRD01 minus V01 on raw reduced QLIKE under the five-minute proxy has mean losses -31.738375 and -32.555815. The mean gap is 0.817439 and favours the second named model. The P5 statistic is 2.651948 with ordinary p-value 0.013695. The absolute statistic exceeds 2.059539, 2.472809, and 2.422650. The win rate is 0.432000. The four block mean signs are [-1, 1, 1, 1]. The sample has 500 targets and nu is 25.

## 2 Loss and proxy robustness

Each row is the same pair, HARQDRD01 minus V01. A positive gap favours the second named model.

| Proxy and loss | Mean losses | Gap | P5 statistic | Ordinary p-value | Cutoff result | Win rate | Block mean signs |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Five-minute squared Frobenius | 0.001717 and 0.001692 | 0.000024 | 1.585294 | 0.125469 | Does not reject at all three cutoffs | 0.398000 | [1, 1, 1, 1] |
| 15-minute reduced QLIKE | -33.224276 and -33.928786 | 0.704510 | 2.388403 | 0.024792 | Does not reject at all three cutoffs | 0.430000 | [-1, 1, 1, 1] |
| 15-minute squared Frobenius | 0.000316 and 0.000290 | 0.000026 | 1.499170 | 0.146352 | Does not reject at all three cutoffs | 0.408000 | [1, 1, 1, 1] |

The headline claim is not loss-robust.

The headline claim is not proxy-robust.

## 3 Secondary contrasts

Both rows use five-minute reduced QLIKE. A negative gap favours the first named model.

| Contrast | Mean losses | Gap | P5 statistic | Ordinary p-value | Cutoff result | Win rate | Block mean signs |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| V01 minus LSTM22 | -32.555815 and -32.061821 | -0.493994 | -1.022166 | 0.316492 | Does not reject at all three cutoffs | 0.504000 | [1, -1, -1, -1] |
| V11 minus LSTM22 | -32.225873 and -32.061821 | -0.164053 | -0.741516 | 0.465290 | Does not reject at all three cutoffs | 0.544000 | [-1, -1, -1, -1] |

Holm uses the two ordinary p-values. A secondary claim also requires both size-adjusted cutoffs.

The secondary claim for V01_minus_LSTM22 is not made. The Holm threshold is 0.025 and the ordinary comparison is not a rejection.

The secondary claim for V11_minus_LSTM22 is not made. The Holm threshold is 0.05 and the ordinary comparison is not a rejection.

## 4 SPA and MCS

The benchmark is HARDRD01 in every row. The only exact tie is RIDGE01.

| Loss | Block length | Consistent p-value |
| --- | ---: | ---: |
| qlike_5min | 29 | 0.083200 |
| qlike_5min | 7 | 0.005000 |
| frobenius_5min | 29 | 0.193400 |
| frobenius_5min | 7 | 0.224800 |
| qlike_15min | 29 | 0.002800 |
| qlike_15min | 7 | 0.004200 |
| frobenius_15min | 29 | 0.195800 |
| frobenius_15min | 7 | 0.222200 |

The second table is the range Model Confidence Set.

| Loss | Block length | Alpha | Models in the set |
| --- | ---: | ---: | --- |
| qlike_5min | 29 | 0.10 | RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, RIDGE01, XGB08, DCC02, DCCNL02, BEKKS01, LSTMBEKK-noC, LSTM19, LSTM22, V01, V11 |
| qlike_5min | 7 | 0.10 | RW01, EWMA01, EWMA21, XGB08, DCC02, DCCNL02, BEKKS01, LSTMBEKK-noC, LSTM19, LSTM22, V01, V11 |
| frobenius_5min | 29 | 0.10 | RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, DCC02, DCCNL02, DCC03, DCCNL03, LSTMBEKK-noC, LSTM19, LSTM22, V10, V01, V11 |
| frobenius_5min | 7 | 0.10 | RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, DCC02, DCCNL02, DCC03, DCCNL03, LSTMBEKK-noC, LSTM19, LSTM22, V10, V01, V11 |
| qlike_15min | 29 | 0.10 | RW01, EWMA01, EWMA21, HARQDRD01, SHARDRD01, XGB08, BEKKS01, LSTMBEKK-noC, LSTM19, LSTM22, V01, V11 |
| qlike_15min | 7 | 0.10 | RW01, EWMA01, EWMA21, XGB08, BEKKS01, LSTMBEKK-noC, LSTM19, LSTM22, V01, V11 |
| frobenius_15min | 29 | 0.10 | RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, DCC02, DCCNL02, DCC03, DCCNL03, LSTMBEKK-noC, LSTM19, LSTM22, V10, V01, V11 |
| frobenius_15min | 7 | 0.10 | RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, DCCNL02, DCCNL03, LSTMBEKK-noC, LSTM19, LSTM22, V10, V01, V11 |

## 5 Conditional and calibration diagnostics

Giacomini-White, pooled Mincer-Zarnowitz, and Giacomini-Rossi are headline-pair diagnostics. They do not change the headline decision.

The market-state Giacomini-White statistic is 109.750235 with Bonferroni p-value 2.484e-23. The measurement-stress statistic is 60.581537 with Bonferroni p-value 8.831e-13. The per-test cutoff is 0.025. Both adjusted p-values are below 0.025.

| Regression | Intercept | Slope | p-value |
| --- | ---: | ---: | ---: |
| mz_HARQDRD01_ols | 0.000792 | 0.227907 | 1.494e-27 |
| mz_V01_ols | 0.000480 | 0.613886 | 0.004613 |
| mz_HARQDRD01_approximate_ps21 | 0.000060 | 0.901943 | 0.120075 |
| mz_V01_approximate_ps21 | -0.000099 | 1.180570 | 0.059176 |

The Giacomini-Rossi maximum absolute statistic is 6.304415. The critical value is 3.012000. The test rejects.

## 6 Economic evaluation

The primary gap is the no-cost squared simple return of the unconstrained global-minimum-variance book. Costed gaps and the six fee cells are variants. No gamma or cost is selected.

### HARQDRD01 minus V01

| Comparison | Statistic | Mean gap | Cutoff result |
| --- | ---: | ---: | --- |
| Gross squared return | 1.173406 | | Does not reject at all three cutoffs |
| Cost 5 basis points | 1.165850 | | Does not reject at all three cutoffs |
| Cost 10 basis points | 1.160100 | | Does not reject at all three cutoffs |
| Long-only gross | 0.251050 | | Does not reject at all three cutoffs |
| HARQDRD01 minus equal weight | -5.740808 | -0.000416 | |
| V01 minus equal weight | -6.460438 | -0.000448 | |

| Gamma | Cost, basis points | Fee per day | Fee times 365 |
| --- | ---: | ---: | ---: |
| 1.0 | 5.0 | -0.000285 | -0.104047 |
| 1.0 | 10.0 | -0.000357 | -0.130199 |
| 5.0 | 5.0 | -0.000349 | -0.127449 |
| 5.0 | 10.0 | -0.000420 | -0.153383 |
| 10.0 | 5.0 | -0.000430 | -0.156847 |
| 10.0 | 10.0 | -0.000500 | -0.182487 |

### V01 minus LSTM22

| Comparison | Statistic | Mean gap | Cutoff result |
| --- | ---: | ---: | --- |
| Gross squared return | 0.080758 | | Does not reject at all three cutoffs |
| Cost 5 basis points | 0.082144 | | Does not reject at all three cutoffs |
| Cost 10 basis points | 0.082314 | | Does not reject at all three cutoffs |
| Long-only gross | -0.674178 | | Does not reject at all three cutoffs |
| V01 minus equal weight | -6.460438 | -0.000448 | |
| LSTM22 minus equal weight | -6.437662 | -0.000449 | |

| Gamma | Cost, basis points | Fee per day | Fee times 365 |
| --- | ---: | ---: | ---: |
| 1.0 | 5.0 | 0.000209 | 0.076161 |
| 1.0 | 10.0 | 0.000221 | 0.080616 |
| 5.0 | 5.0 | 0.000207 | 0.075652 |
| 5.0 | 10.0 | 0.000219 | 0.080090 |
| 10.0 | 5.0 | 0.000206 | 0.075015 |
| 10.0 | 10.0 | 0.000218 | 0.079431 |

### V11 minus LSTM22

| Comparison | Statistic | Mean gap | Cutoff result |
| --- | ---: | ---: | --- |
| Gross squared return | 0.105909 | | Does not reject at all three cutoffs |
| Cost 5 basis points | 0.105028 | | Does not reject at all three cutoffs |
| Cost 10 basis points | 0.103362 | | Does not reject at all three cutoffs |
| Long-only gross | -0.388229 | | Does not reject at all three cutoffs |
| V11 minus equal weight | -6.380711 | -0.000448 | |
| LSTM22 minus equal weight | -6.437662 | -0.000449 | |

| Gamma | Cost, basis points | Fee per day | Fee times 365 |
| --- | ---: | ---: | ---: |
| 1.0 | 5.0 | 0.000115 | 0.041977 |
| 1.0 | 10.0 | 0.000123 | 0.044855 |
| 5.0 | 5.0 | 0.000113 | 0.041126 |
| 5.0 | 10.0 | 0.000121 | 0.044009 |
| 10.0 | 5.0 | 0.000110 | 0.040061 |
| 10.0 | 10.0 | 0.000118 | 0.042950 |

## 7 HARQ fallback sensitivity

HARQDRD01 used the estimation-window-mean fallback on 16 CONFIRM targets. The dates are 2025-10-11, 2025-10-19 through 2025-11-01, and 2026-08-23. The primary table keeps those targets. The reduced pair has 484 targets and is a sensitivity.

The headline pair with fallback targets removed has mean losses -31.728498 and -32.615314. The mean gap is 0.886816 and favours the second named model. The P5 statistic is 2.822534 with ordinary p-value 0.009422. The absolute statistic exceeds the three preregistered cutoffs. The win rate is 0.417355. The four block mean signs are [-1, 1, 1, 1].

## 8 Recalibration diagnostics

R1 and R2 are diagnostics. The headline uses the raw forecast.

| Model | Map | Mean reduced QLIKE | Mean squared Frobenius |
| --- | --- | ---: | ---: |
| HARQDRD01 | r1 | -30.584721 | 0.001751 |
| HARQDRD01 | r2 | -31.088719 | 0.001710 |
| LSTM22 | r1 | -31.173012 | 0.001704 |
| LSTM22 | r2 | -31.712250 | 0.001695 |
| V01 | r1 | -31.524161 | 0.001713 |
| V01 | r2 | -31.742073 | 0.001696 |
| V11 | r1 | -31.166676 | 0.001726 |
| V11 | r2 | -31.678527 | 0.001697 |

## 9 Model table

Mean reduced QLIKE and squared Frobenius are reported under both proxies. Stochastic rows are equal-weight covariance ensembles.

| Model | Five-minute QLIKE | Five-minute Frobenius | 15-minute QLIKE | 15-minute Frobenius |
| --- | ---: | ---: | ---: | ---: |
| RW01 | -30.595609 | 0.003308 | -33.563382 | 0.001918 |
| EWMA01 | -31.805969 | 0.001863 | -33.988229 | 0.000463 |
| EWMA21 | -31.711754 | 0.002085 | -34.054414 | 0.000689 |
| HARDRD01 | -31.662358 | 0.001865 | -33.115961 | 0.000468 |
| HARQDRD01 | -31.738375 | 0.001717 | -33.224276 | 0.000316 |
| SHARDRD01 | -31.721639 | 0.002314 | -33.187574 | 0.000921 |
| LWLIN01 | -32.084323 | 0.001690 | -33.086819 | 0.000289 |
| LWNL01 | -31.780246 | 0.001689 | -33.090451 | 0.000289 |
| RIDGE01 | -31.662358 | 0.001865 | -33.115961 | 0.000468 |
| XGB08 | -31.691403 | 0.001781 | -33.148072 | 0.000380 |
| DCC02 | -32.339636 | 0.001686 | -33.677839 | 0.000285 |
| DCCNL02 | -32.387473 | 0.001686 | -33.701958 | 0.000285 |
| DCC03 | -32.104590 | 0.001687 | -33.427280 | 0.000286 |
| DCCNL03 | -32.138454 | 0.001687 | -33.434341 | 0.000286 |
| BEKKS01 | -32.719974 | 0.001687 | -33.855700 | 0.000286 |
| LSTMBEKK-noC | -32.676606 | 0.001687 | -33.878356 | 0.000286 |
| LSTM19 | -32.004755 | 0.001689 | -33.695216 | 0.000286 |
| LSTM22 | -32.061821 | 0.001689 | -33.766047 | 0.000286 |
| V10 | -31.933484 | 0.001689 | -33.641331 | 0.000286 |
| V01 | -32.555815 | 0.001692 | -33.928786 | 0.000290 |
| V11 | -32.225873 | 0.001694 | -33.778193 | 0.000291 |

Seed-level mean five-minute reduced QLIKE values are not the reported forecast. The reported stochastic forecast is the equal-weight covariance ensemble in the table above. Columns follow the saved seed order.

| Model | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTMBEKK-noC | -32.676606 | -32.676606 | -32.676606 | -32.676606 | -32.676606 |
| LSTM19 | -32.085752 | -31.872816 | -31.921901 | -32.048242 | -31.979598 |
| LSTM22 | -32.049942 | -31.995839 | -31.956426 | -32.002137 | -32.136617 |
| V10 | -31.974071 | -31.810240 | -31.840684 | -31.883121 | -31.971224 |
| V01 | -32.430201 | -32.381897 | -32.816699 | -32.572087 | -32.392766 |
| V11 | -32.301486 | -32.315830 | -32.067261 | -32.141943 | -32.237493 |

The five LSTMBEKK-noC forecast cubes are identical.

Estimation-window-mean fallback counts are HARDRD01 14, HARQDRD01 16, SHARDRD01 14, RIDGE01 14, and XGB08 41. DCC02 and DCCNL02 each carried parameters forward at origins 2025-07-05 and 2026-06-06. DCC03 and DCCNL03 each recorded six near-boundary interior-map events, at origins 2025-04-12, 2025-05-24, 2025-11-29, 2025-12-20, 2026-02-21, and 2026-06-06. BEKKS01 saved no policy-event list and its repair count is 0.

## Deviations

V01 uses the return-only input. The shared fit signature still requires a feature array, so the runner passed a column of zeros that V01 does not read. The forecast is unchanged by that array.

The five LSTMBEKK-noC seed cubes are identical. The ablation column is that common cube. No seed was selected.

BEKKS01 carry counts were not written to the job summary. The saved repair count is 0 and the saved policy-event list is empty. The model was not refit to recover a counter.
