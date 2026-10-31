# Binance CONFIRM report

## 1 Headline

The preregistered rule rejects equal predictive accuracy, and the direction favours V01.

HARQDRD01 minus V01 on raw reduced QLIKE under the five-minute proxy has mean losses -31.738375 and -32.555815. The mean gap is 0.817439 and favours the second named model. The P5 statistic is 2.651948 with ordinary p-value 0.013695. The absolute statistic exceeds 2.059539, 2.472809, and 2.422650. The win rate is 0.432000. The four block mean signs are [-1, 1, 1, 1]. The sample has 500 targets and nu is 25.

## 2 Loss and proxy robustness

The same pair on squared Frobenius under the five-minute proxy has mean losses 0.001717 and 0.001692. The mean gap is 0.000024 and favours the second named model. The P5 statistic is 1.585294 with ordinary p-value 0.125469. The absolute statistic does not reject at all three cutoffs. The win rate is 0.398000. The four block mean signs are [1, 1, 1, 1].
The same pair on reduced QLIKE under the 15-minute proxy has mean losses -33.224276 and -33.928786. The mean gap is 0.704510 and favours the second named model. The P5 statistic is 2.388403 with ordinary p-value 0.024792. The absolute statistic does not reject at all three cutoffs. The win rate is 0.430000. The four block mean signs are [-1, 1, 1, 1].
The same pair on squared Frobenius under the 15-minute proxy has mean losses 0.000316 and 0.000290. The mean gap is 0.000026 and favours the second named model. The P5 statistic is 1.499170 with ordinary p-value 0.146352. The absolute statistic does not reject at all three cutoffs. The win rate is 0.408000. The four block mean signs are [1, 1, 1, 1].
The headline claim is not loss-robust.
The headline claim is not proxy-robust.

## 3 Secondary contrasts

V01 minus LSTM22 on five-minute reduced QLIKE has mean losses -32.555815 and -32.061821. The mean gap is -0.493994 and favours the first named model. The P5 statistic is -1.022166 with ordinary p-value 0.316492. The absolute statistic does not reject at all three cutoffs. The win rate is 0.504000. The four block mean signs are [1, -1, -1, -1].
V11 minus LSTM22 on five-minute reduced QLIKE has mean losses -32.225873 and -32.061821. The mean gap is -0.164053 and favours the first named model. The P5 statistic is -0.741516 with ordinary p-value 0.465290. The absolute statistic does not reject at all three cutoffs. The win rate is 0.544000. The four block mean signs are [-1, -1, -1, -1].
Holm uses the two ordinary p-values. A secondary claim also requires both size-adjusted cutoffs.
The secondary claim for V01_minus_LSTM22 is not made. The Holm threshold is 0.025 and the ordinary comparison is not a rejection.
The secondary claim for V11_minus_LSTM22 is not made. The Holm threshold is 0.05 and the ordinary comparison is not a rejection.

## 4 SPA and MCS

SPA on qlike_5min at block length 29 has consistent p-value 0.083200. The benchmark is HARDRD01. Exact ties are ['RIDGE01'].
The range MCS on qlike_5min at block length 29 and alpha 0.10 contains RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, RIDGE01, XGB08, DCC02, DCCNL02, BEKKS01, LSTMBEKK-noC, LSTM19, LSTM22, V01, V11.
SPA on qlike_5min at block length 7 has consistent p-value 0.005000. The benchmark is HARDRD01. Exact ties are ['RIDGE01'].
The range MCS on qlike_5min at block length 7 and alpha 0.10 contains RW01, EWMA01, EWMA21, XGB08, DCC02, DCCNL02, BEKKS01, LSTMBEKK-noC, LSTM19, LSTM22, V01, V11.
SPA on frobenius_5min at block length 29 has consistent p-value 0.193400. The benchmark is HARDRD01. Exact ties are ['RIDGE01'].
The range MCS on frobenius_5min at block length 29 and alpha 0.10 contains RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, DCC02, DCCNL02, DCC03, DCCNL03, LSTMBEKK-noC, LSTM19, LSTM22, V10, V01, V11.
SPA on frobenius_5min at block length 7 has consistent p-value 0.224800. The benchmark is HARDRD01. Exact ties are ['RIDGE01'].
The range MCS on frobenius_5min at block length 7 and alpha 0.10 contains RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, DCC02, DCCNL02, DCC03, DCCNL03, LSTMBEKK-noC, LSTM19, LSTM22, V10, V01, V11.
SPA on qlike_15min at block length 29 has consistent p-value 0.002800. The benchmark is HARDRD01. Exact ties are ['RIDGE01'].
The range MCS on qlike_15min at block length 29 and alpha 0.10 contains RW01, EWMA01, EWMA21, HARQDRD01, SHARDRD01, XGB08, BEKKS01, LSTMBEKK-noC, LSTM19, LSTM22, V01, V11.
SPA on qlike_15min at block length 7 has consistent p-value 0.004200. The benchmark is HARDRD01. Exact ties are ['RIDGE01'].
The range MCS on qlike_15min at block length 7 and alpha 0.10 contains RW01, EWMA01, EWMA21, XGB08, BEKKS01, LSTMBEKK-noC, LSTM19, LSTM22, V01, V11.
SPA on frobenius_15min at block length 29 has consistent p-value 0.195800. The benchmark is HARDRD01. Exact ties are ['RIDGE01'].
The range MCS on frobenius_15min at block length 29 and alpha 0.10 contains RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, DCC02, DCCNL02, DCC03, DCCNL03, LSTMBEKK-noC, LSTM19, LSTM22, V10, V01, V11.
SPA on frobenius_15min at block length 7 has consistent p-value 0.222200. The benchmark is HARDRD01. Exact ties are ['RIDGE01'].
The range MCS on frobenius_15min at block length 7 and alpha 0.10 contains RW01, EWMA01, EWMA21, HARDRD01, HARQDRD01, SHARDRD01, LWLIN01, LWNL01, RIDGE01, XGB08, DCCNL02, DCCNL03, LSTMBEKK-noC, LSTM19, LSTM22, V10, V01, V11.

## 5 Conditional and calibration diagnostics

Giacomini-White, pooled Mincer-Zarnowitz, and Giacomini-Rossi are headline-pair diagnostics. They do not change the headline decision.
The market-state Giacomini-White statistic is 109.750235 with Bonferroni p-value 2.484e-23. The measurement-stress statistic is 60.581537 with Bonferroni p-value 8.831e-13. The per-test cutoff is 0.025. Both adjusted p-values are below 0.025.
Pooled Mincer-Zarnowitz mz_HARQDRD01_ols has intercept 0.000792, slope 0.227907, and p-value 0.000000.
Pooled Mincer-Zarnowitz mz_V01_ols has intercept 0.000480, slope 0.613886, and p-value 0.004613.
Pooled Mincer-Zarnowitz mz_HARQDRD01_approximate_ps21 has intercept 0.000060, slope 0.901943, and p-value 0.120075.
Pooled Mincer-Zarnowitz mz_V01_approximate_ps21 has intercept -0.000099, slope 1.180570, and p-value 0.059176.
The Giacomini-Rossi maximum absolute statistic is 6.304415. The critical value is 3.012000. The test rejects.

## 6 Economic evaluation

The primary gap is the no-cost squared simple return of the unconstrained global-minimum-variance book. Costed gaps and the six fee cells are variants. No gamma or cost is selected.
The gross squared-return gap HARQDRD01 minus V01 has statistic 1.173406 and does not reject at all three cutoffs.
The cost_5bps squared-return gap HARQDRD01 minus V01 has statistic 1.165850 and does not reject at all three cutoffs.
The cost_10bps squared-return gap HARQDRD01 minus V01 has statistic 1.160100 and does not reject at all three cutoffs.
The long_only_gross squared-return gap HARQDRD01 minus V01 has statistic 0.251050 and does not reject at all three cutoffs.
The unconstrained book HARQDRD01 minus equal weight has statistic -5.740808 and mean gap -0.000416.
The unconstrained book V01 minus equal weight has statistic -6.460438 and mean gap -0.000448.
The fee for HARQDRD01 minus V01 at gamma 1.0 and 5.0 basis points is -0.000285 per day and -0.104047 times 365.
The fee for HARQDRD01 minus V01 at gamma 1.0 and 10.0 basis points is -0.000357 per day and -0.130199 times 365.
The fee for HARQDRD01 minus V01 at gamma 5.0 and 5.0 basis points is -0.000349 per day and -0.127449 times 365.
The fee for HARQDRD01 minus V01 at gamma 5.0 and 10.0 basis points is -0.000420 per day and -0.153383 times 365.
The fee for HARQDRD01 minus V01 at gamma 10.0 and 5.0 basis points is -0.000430 per day and -0.156847 times 365.
The fee for HARQDRD01 minus V01 at gamma 10.0 and 10.0 basis points is -0.000500 per day and -0.182487 times 365.
The gross squared-return gap V01 minus LSTM22 has statistic 0.080758 and does not reject at all three cutoffs.
The cost_5bps squared-return gap V01 minus LSTM22 has statistic 0.082144 and does not reject at all three cutoffs.
The cost_10bps squared-return gap V01 minus LSTM22 has statistic 0.082314 and does not reject at all three cutoffs.
The long_only_gross squared-return gap V01 minus LSTM22 has statistic -0.674178 and does not reject at all three cutoffs.
The unconstrained book V01 minus equal weight has statistic -6.460438 and mean gap -0.000448.
The unconstrained book LSTM22 minus equal weight has statistic -6.437662 and mean gap -0.000449.
The fee for V01 minus LSTM22 at gamma 1.0 and 5.0 basis points is 0.000209 per day and 0.076161 times 365.
The fee for V01 minus LSTM22 at gamma 1.0 and 10.0 basis points is 0.000221 per day and 0.080616 times 365.
The fee for V01 minus LSTM22 at gamma 5.0 and 5.0 basis points is 0.000207 per day and 0.075652 times 365.
The fee for V01 minus LSTM22 at gamma 5.0 and 10.0 basis points is 0.000219 per day and 0.080090 times 365.
The fee for V01 minus LSTM22 at gamma 10.0 and 5.0 basis points is 0.000206 per day and 0.075015 times 365.
The fee for V01 minus LSTM22 at gamma 10.0 and 10.0 basis points is 0.000218 per day and 0.079431 times 365.
The gross squared-return gap V11 minus LSTM22 has statistic 0.105909 and does not reject at all three cutoffs.
The cost_5bps squared-return gap V11 minus LSTM22 has statistic 0.105028 and does not reject at all three cutoffs.
The cost_10bps squared-return gap V11 minus LSTM22 has statistic 0.103362 and does not reject at all three cutoffs.
The long_only_gross squared-return gap V11 minus LSTM22 has statistic -0.388229 and does not reject at all three cutoffs.
The unconstrained book V11 minus equal weight has statistic -6.380711 and mean gap -0.000448.
The unconstrained book LSTM22 minus equal weight has statistic -6.437662 and mean gap -0.000449.
The fee for V11 minus LSTM22 at gamma 1.0 and 5.0 basis points is 0.000115 per day and 0.041977 times 365.
The fee for V11 minus LSTM22 at gamma 1.0 and 10.0 basis points is 0.000123 per day and 0.044855 times 365.
The fee for V11 minus LSTM22 at gamma 5.0 and 5.0 basis points is 0.000113 per day and 0.041126 times 365.
The fee for V11 minus LSTM22 at gamma 5.0 and 10.0 basis points is 0.000121 per day and 0.044009 times 365.
The fee for V11 minus LSTM22 at gamma 10.0 and 5.0 basis points is 0.000110 per day and 0.040061 times 365.
The fee for V11 minus LSTM22 at gamma 10.0 and 10.0 basis points is 0.000118 per day and 0.042950 times 365.

## 7 HARQ fallback sensitivity

HARQDRD01 used the estimation-window-mean fallback on 16 CONFIRM targets. The dates are 2025-10-11, 2025-10-19 through 2025-11-01, and 2026-08-23. The primary table keeps those targets. The reduced pair has 484 targets and is a sensitivity.

The headline pair with fallback targets removed has mean losses -31.728498 and -32.615314. The mean gap is 0.886816 and favours the second named model. The P5 statistic is 2.822534 with ordinary p-value 0.009422. The absolute statistic exceeds the three preregistered cutoffs. The win rate is 0.417355. The four block mean signs are [-1, 1, 1, 1].

## 8 Recalibration diagnostics

R1 and R2 are diagnostics. The headline uses the raw forecast.
HARQDRD01 r1 mean reduced QLIKE is -30.584721 and mean squared Frobenius is 0.001751.
HARQDRD01 r2 mean reduced QLIKE is -31.088719 and mean squared Frobenius is 0.001710.
LSTM22 r1 mean reduced QLIKE is -31.173012 and mean squared Frobenius is 0.001704.
LSTM22 r2 mean reduced QLIKE is -31.712250 and mean squared Frobenius is 0.001695.
V01 r1 mean reduced QLIKE is -31.524161 and mean squared Frobenius is 0.001713.
V01 r2 mean reduced QLIKE is -31.742073 and mean squared Frobenius is 0.001696.
V11 r1 mean reduced QLIKE is -31.166676 and mean squared Frobenius is 0.001726.
V11 r2 mean reduced QLIKE is -31.678527 and mean squared Frobenius is 0.001697.

## 9 Model table

Mean reduced QLIKE and squared Frobenius are reported under both proxies. Stochastic rows are equal-weight covariance ensembles.

RW01 five-minute QLIKE -30.595609, five-minute Frobenius 0.003308, 15-minute QLIKE -33.563382, 15-minute Frobenius 0.001918.
EWMA01 five-minute QLIKE -31.805969, five-minute Frobenius 0.001863, 15-minute QLIKE -33.988229, 15-minute Frobenius 0.000463.
EWMA21 five-minute QLIKE -31.711754, five-minute Frobenius 0.002085, 15-minute QLIKE -34.054414, 15-minute Frobenius 0.000689.
HARDRD01 five-minute QLIKE -31.662358, five-minute Frobenius 0.001865, 15-minute QLIKE -33.115961, 15-minute Frobenius 0.000468.
HARQDRD01 five-minute QLIKE -31.738375, five-minute Frobenius 0.001717, 15-minute QLIKE -33.224276, 15-minute Frobenius 0.000316.
SHARDRD01 five-minute QLIKE -31.721639, five-minute Frobenius 0.002314, 15-minute QLIKE -33.187574, 15-minute Frobenius 0.000921.
LWLIN01 five-minute QLIKE -32.084323, five-minute Frobenius 0.001690, 15-minute QLIKE -33.086819, 15-minute Frobenius 0.000289.
LWNL01 five-minute QLIKE -31.780246, five-minute Frobenius 0.001689, 15-minute QLIKE -33.090451, 15-minute Frobenius 0.000289.
RIDGE01 five-minute QLIKE -31.662358, five-minute Frobenius 0.001865, 15-minute QLIKE -33.115961, 15-minute Frobenius 0.000468.
XGB08 five-minute QLIKE -31.691403, five-minute Frobenius 0.001781, 15-minute QLIKE -33.148072, 15-minute Frobenius 0.000380.
DCC02 five-minute QLIKE -32.339636, five-minute Frobenius 0.001686, 15-minute QLIKE -33.677839, 15-minute Frobenius 0.000285.
DCCNL02 five-minute QLIKE -32.387473, five-minute Frobenius 0.001686, 15-minute QLIKE -33.701958, 15-minute Frobenius 0.000285.
DCC03 five-minute QLIKE -32.104590, five-minute Frobenius 0.001687, 15-minute QLIKE -33.427280, 15-minute Frobenius 0.000286.
DCCNL03 five-minute QLIKE -32.138454, five-minute Frobenius 0.001687, 15-minute QLIKE -33.434341, 15-minute Frobenius 0.000286.
BEKKS01 five-minute QLIKE -32.719974, five-minute Frobenius 0.001687, 15-minute QLIKE -33.855700, 15-minute Frobenius 0.000286.
LSTMBEKK-noC five-minute QLIKE -32.676606, five-minute Frobenius 0.001687, 15-minute QLIKE -33.878356, 15-minute Frobenius 0.000286.
LSTM19 five-minute QLIKE -32.004755, five-minute Frobenius 0.001689, 15-minute QLIKE -33.695216, 15-minute Frobenius 0.000286.
LSTM22 five-minute QLIKE -32.061821, five-minute Frobenius 0.001689, 15-minute QLIKE -33.766047, 15-minute Frobenius 0.000286.
V10 five-minute QLIKE -31.933484, five-minute Frobenius 0.001689, 15-minute QLIKE -33.641331, 15-minute Frobenius 0.000286.
V01 five-minute QLIKE -32.555815, five-minute Frobenius 0.001692, 15-minute QLIKE -33.928786, 15-minute Frobenius 0.000290.
V11 five-minute QLIKE -32.225873, five-minute Frobenius 0.001694, 15-minute QLIKE -33.778193, 15-minute Frobenius 0.000291.

Seed-level mean five-minute reduced QLIKE values are not the reported forecast. The reported stochastic forecast is the equal-weight covariance ensemble in the table above.

LSTMBEKK-noC seed means are -32.676606, -32.676606, -32.676606, -32.676606, and -32.676606. The five forecast cubes are identical.
LSTM19 seed means are -32.085752, -31.872816, -31.921901, -32.048242, and -31.979598.
LSTM22 seed means are -32.049942, -31.995839, -31.956426, -32.002137, and -32.136617.
V10 seed means are -31.974071, -31.810240, -31.840684, -31.883121, and -31.971224.
V01 seed means are -32.430201, -32.381897, -32.816699, -32.572087, and -32.392766.
V11 seed means are -32.301486, -32.315830, -32.067261, -32.141943, and -32.237493.

Estimation-window-mean fallback counts are HARDRD01 14, HARQDRD01 16, SHARDRD01 14, RIDGE01 14, and XGB08 41. DCC02 and DCCNL02 each carried parameters forward at origins 2025-07-05 and 2026-06-06. DCC03 and DCCNL03 each recorded six near-boundary interior-map events, at origins 2025-04-12, 2025-05-24, 2025-11-29, 2025-12-20, 2026-02-21, and 2026-06-06. BEKKS01 saved no policy-event list and its repair count is 0.

## Deviations

V01 uses the return-only input. The shared fit signature still requires a feature array, so the runner passed a column of zeros that V01 does not read. The forecast is unchanged by that array.

The five LSTMBEKK-noC seed cubes are identical. The ablation column is that common cube. No seed was selected.

BEKKS01 carry counts were not written to the job summary. The saved repair count is 0 and the saved policy-event list is empty. The model was not refit to recover a counter.
