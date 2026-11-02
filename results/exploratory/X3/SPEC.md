# X3 specification

Exploratory, computed after CONFIRM. This file is not a confirmatory result and it does not change the preregistered rule.

## Question

Target 2025-10-10 is the October 2025 liquidation day. X3 records how that one target moves the saved CONFIRM reduced-QLIKE ranking. No model is refit. No day is removed from one model only.

## Inputs

The forecast cubes are `results/confirm/jobs/*/forecast.npz`. A stochastic identifier uses the equal-weight mean of its five seed cubes, the same ensemble the CONFIRM scorer forms. The 15-minute proxy, the target dates, and the origin dates are `results/confirm/inputs/confirm_measurements.npz`. The five-minute proxy is the production-panel realized covariance on those target dates, which is the proxy already stored inside each deterministic forecast file as the scored loss and is reused here only to rescore the ensemble. No other CONFIRM file is rewritten.

## Operations

Reduced QLIKE is the existing loss. A target enters a proxy view when that proxy and every model forecast on that date produce a finite loss. The five-minute view and the 15-minute view are separate.

For each of the 21 CONFIRM identifiers the table reports the mean loss on the finite-proxy targets, the mean without 2025-10-10, the median, and the rank in each of those three views. Rank 1 is the lowest loss. Ties use the average rank.

The headline gap is HARQDRD01 minus V01. The crash-day term, its share of the summed gap, and the P5 statistic on the finite-proxy sample and without 2025-10-10 are reported. P5 is the equal-weighted cosine statistic with nu fixed at 25. The cutoffs are 2.0595385527532972, 2.4226501420943998, and 2.4728093597276533.

Leave-one-out influence drops each headline target once, for both models together, and recomputes that same statistic. The minimum statistic, its date, and the five dates with the largest absolute change from the full-sample statistic are reported.

The same all-sample mean, mean without 2025-10-10, and median are reported for V01 minus LSTMBEKK-noC, LSTM22 minus LSTMBEKK-noC, V01 minus LSTM22, and BEKKS01 minus V01.

## Known-truth gate

HARQDRD01 per-target five-minute QLIKE must match the stored CONFIRM loss to 1e-10. The 15-minute headline gap must be 0.70451 at five decimals and its P5 statistic 2.388403 at six decimals. The 2025-10-10 headline gap must be 59.83 at two decimals. Means without that date must agree with EWMA21 -34.895, V01 -34.437, and HARQDRD01 -33.738 at the third decimal, allowing a discrepancy below 0.005 where the saved series is finer than the stated figure. The task text also states a 499-target five-minute P5 of 2.647 and a without-crash P5 of 2.400. Those two figures are checked against the saved series and are not imposed by dropping a day.

## Label

Every table and the report carry the label exploratory, computed after CONFIRM.
