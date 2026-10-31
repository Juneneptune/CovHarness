# Where the headline gap comes from

Exploratory, after CONFIRM; not a confirmatory claim.

The loss is raw reduced QLIKE on the five-minute proxy. The gap is the loss of the first named model minus the loss of the second. A positive gap favours the second model. For HARQDRD01 minus V01, a positive gap favours V01. No new p-value is computed. The joint Giacomini–White statistics already stored in `results/confirm/diagnostics.json` are unchanged.

## Giacomini–White slopes

The committed diagnostics file stores the joint statistics and not the slope coefficients. We recovered the coefficients by least squares of the headline gap on each frozen instrument set separately. The two sets are not stacked. The constant is included. Recomputing the joint statistics from those same series reproduces the stored market-state statistic 109.750235 and the stored measurement-stress statistic 60.581537. The stored Bonferroni p-values are 2.484e-23 and 8.831e-13.

Market-state set. The coefficient on the constant is -8.483613. The coefficient on origin-day log mean realized variance is -0.682862. The coefficient on origin-day mean realized correlation is 6.071119.

Measurement-stress set. The coefficient on the constant is -0.536823. The coefficient on origin-day log aggregate realized quarticity is -0.095012. The coefficient on the origin-day BNS jump indicator is 0.570277.

The negative variance slope and the negative quarticity slope predict a smaller V01 advantage when those origin-day series are higher, holding the other instrument in the same set fixed. The positive correlation slope and the positive jump slope predict a larger V01 advantage in those states. The intercept is the fitted gap when the other instruments are zero. Log variance and log quarticity are not zero on these origins, so the intercept is not itself a state.

## State cutoffs fixed on SCREEN origins

The cutoffs use the 500 SCREEN forecast origins from 2023-11-29 through 2025-04-11. They are not computed from CONFIRM. The one-third and two-thirds quantiles of origin-day log mean realized variance are -7.207464 and -6.490094. The same quantiles of origin-day mean realized correlation are 0.632427 and 0.738028. A CONFIRM origin at or below the lower quantile is low. A CONFIRM origin above the upper quantile is high. Every other CONFIRM origin is middle. The jump bucket is the BNS indicator on the CONFIRM origin day, with no tercile.

A lower-loss share is the share of days in the bucket on which that model has the strictly smaller loss. There are no exact ties. Pairs that do not contain V01 do not have a V01 win share.

## Variance buckets

Low variance has 235 days. HARQDRD01 minus V01 has mean gap 0.945392 and V01 lower-loss share 0.578723. HARQDRD01 minus BEKKS01 has mean gap 0.696506 and BEKKS01 lower-loss share 0.489362. V01 minus LSTMBEKK-noC has mean gap -0.128801 and V01 lower-loss share 0.578723. LSTM22 minus LSTMBEKK-noC has mean gap -0.190882 and LSTM22 lower-loss share 0.748936.

Middle variance has 181 days. HARQDRD01 minus V01 has mean gap 0.751430 and V01 lower-loss share 0.546961. HARQDRD01 minus BEKKS01 has mean gap 1.593598 and BEKKS01 lower-loss share 0.475138. V01 minus LSTMBEKK-noC has mean gap 0.592447 and V01 lower-loss share 0.513812. LSTM22 minus LSTMBEKK-noC has mean gap 1.928946 and LSTM22 lower-loss share 0.712707.

High variance has 84 days. HARQDRD01 minus V01 has mean gap 0.601713 and V01 lower-loss share 0.583333. HARQDRD01 minus BEKKS01 has mean gap 0.460468 and BEKKS01 lower-loss share 0.523810. V01 minus LSTMBEKK-noC has mean gap -0.197248 and V01 lower-loss share 0.678571. LSTM22 minus LSTMBEKK-noC has mean gap 0.037031 and LSTM22 lower-loss share 0.535714.

The unconditional mean HARQDRD01 minus V01 gap is largest in the low-variance bucket and smallest in the high-variance bucket. That lines up with the negative variance slope. It is not a new test.

## Correlation buckets

Low correlation has 80 days. HARQDRD01 minus V01 has mean gap 0.342401 and V01 lower-loss share 0.375000. HARQDRD01 minus BEKKS01 has mean gap 0.052199 and BEKKS01 lower-loss share 0.300000. V01 minus LSTMBEKK-noC has mean gap -0.206648 and V01 lower-loss share 0.687500. LSTM22 minus LSTMBEKK-noC has mean gap -0.149609 and LSTM22 lower-loss share 0.762500.

Middle correlation has 151 days. HARQDRD01 minus V01 has mean gap 0.563366 and V01 lower-loss share 0.483444. HARQDRD01 minus BEKKS01 has mean gap 1.509622 and BEKKS01 lower-loss share 0.364238. V01 minus LSTMBEKK-noC has mean gap 0.669610 and V01 lower-loss share 0.549669. LSTM22 minus LSTMBEKK-noC has mean gap 2.208299 and LSTM22 lower-loss share 0.728477.

High correlation has 269 days. HARQDRD01 minus V01 has mean gap 1.101336 and V01 lower-loss share 0.672862. HARQDRD01 minus BEKKS01 has mean gap 0.961601 and BEKKS01 lower-loss share 0.617100. V01 minus LSTMBEKK-noC has mean gap -0.089901 and V01 lower-loss share 0.550186. LSTM22 minus LSTMBEKK-noC has mean gap -0.052386 and LSTM22 lower-loss share 0.665428.

The unconditional mean HARQDRD01 minus V01 gap and the V01 lower-loss share are both largest in the high-correlation bucket. That lines up with the positive correlation slope. It is not a new test.

## Jump buckets

No jump has 400 days. HARQDRD01 minus V01 has mean gap 0.711584 and V01 lower-loss share 0.532500. HARQDRD01 minus BEKKS01 has mean gap 0.941935 and BEKKS01 lower-loss share 0.445000. V01 minus LSTMBEKK-noC has mean gap 0.158297 and V01 lower-loss share 0.592500. LSTM22 minus LSTMBEKK-noC has mean gap 0.768315 and LSTM22 lower-loss share 0.700000.

A jump has 100 days. HARQDRD01 minus V01 has mean gap 1.240861 and V01 lower-loss share 0.710000. HARQDRD01 minus BEKKS01 has mean gap 1.140258 and BEKKS01 lower-loss share 0.670000. V01 minus LSTMBEKK-noC has mean gap -0.029231 and V01 lower-loss share 0.490000. LSTM22 minus LSTMBEKK-noC has mean gap 0.000665 and LSTM22 lower-loss share 0.700000.

The unconditional mean HARQDRD01 minus V01 gap is larger on jump origins. That lines up with the positive jump slope. It is not a new test.

## Does the LSTM have the lower loss in any state

V01 has a negative mean gap against LSTMBEKK-noC in the low-variance, high-variance, low-correlation, high-correlation, and jump buckets. V01 has a positive mean gap against that ablation in the middle-variance, middle-correlation, and no-jump buckets. LSTM22 has a negative mean gap against the same ablation in the low-variance, low-correlation, and high-correlation buckets. LSTM22 has a positive mean gap in the middle-variance and middle-correlation buckets, and a mean gap near zero on jump origins. These signs describe the saved losses. They do not select a state.

## October 2025 sell-off

The shaded window is the UTC target dates 2025-10-10 and 2025-10-11. Those are the dates of the liquidation event reported by Reuters on 13 October 2025. The window was fixed from that report before the share below was computed. The cumulative HARQDRD01 minus V01 gap is in `results/confirm/exploratory/x1_cumulative_gap.png`. The sum of the gap on those two target dates is 59.986270. The sum over all 500 CONFIRM targets is 408.719650. The two dates account for 0.146766 of the summed gap.

Exploratory, after CONFIRM; not a confirmatory claim.
