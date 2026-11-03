# Glossary

Internal identifiers in the code, configs, and reports are listed here with the name used in the overview. Dates are UTC. A model identifier is one fitted configuration, not a whole family.

| Identifier | Plain name |
| --- | --- |
| HISTORY_A | First estimation history, 2021-11-09 to 2022-07-16 (250 days) |
| VALIDATION | Tuning period, 2022-07-17 to 2023-03-23 (250 days) |
| HISTORY_B | Post-halt estimation history, 2023-03-25 to 2023-11-29 (250 days). 2023-03-24 is the exchange halt and is not a forecast date |
| SCREEN | Selection period, 2023-11-30 to 2025-04-12 (500 days) |
| CONFIRM | Locked holdout, 2025-04-13 to 2026-08-25 (500 days) |
| RW01 | Random walk. Tomorrow's forecast is today's realized covariance |
| EWMA01 | Exponential smoothing with decay 0.80, the first point on the original grid |
| EWMA21 | Exponential smoothing with decay 0.60, the configuration selected after the edge search |
| HARDRD01 | HAR-DRD |
| HARQDRD01 | HARQ-DRD, the preselected econometric finalist |
| SHARDRD01 | SHAR-DRD, HAR-DRD with signed semivariances |
| RIDGE01 | Ridge-penalized HAR-DRD. The selected penalty is 0, so the forecast matches HAR-DRD |
| XGB08 | XGBoost on HAR features, the selected configuration |
| LWLIN01 | Ledoit–Wolf linear shrinkage |
| LWNL01 | Ledoit–Wolf nonlinear shrinkage |
| DCC01 | DCC with the original rejection policy. It did not finish the selection period |
| DCCNL01 | DCC with nonlinear shrinkage and the original rejection policy. It did not finish the selection period |
| DCC02 | DCC, carrying forward the last valid parameters after a failed refit |
| DCCNL02 | DCC-NL with the same carry-forward rule |
| DCC03 | DCC with an interior parameterization |
| DCCNL03 | DCC-NL with an interior parameterization |
| BEKKS01 | Scalar BEKK without covariance targeting. This is the selected BEKK configuration |
| BEKKS02 | Scalar BEKK with covariance targeting |
| BEKKD01 | Diagonal BEKK without covariance targeting |
| BEKKD02 | Diagonal BEKK with covariance targeting |
| BEKKF01 | Full BEKK without covariance targeting |
| BEKKF02 | Full BEKK with covariance targeting |
| BEKKS01-INT | Scalar BEKK with the interior parameterization. A robustness fit, not a headline model |
| LSTM19 | LSTM-BEKK from the original grid |
| LSTM22 | LSTM-BEKK selected after the depth search |
| LSTMBEKK-noC | LSTM-BEKK with the dynamic intercept removed, so the LSTM does not enter the recursion |
| V10 | LSTM-BEKK with realized covariance in the inputs |
| V01 | LSTM-BEKK trained on realized covariance. This is the deep-learning finalist |
| V11 | LSTM-BEKK with realized-covariance inputs and a realized-covariance training target |
| P5 | Equal-weighted cosine test. At 500 days it uses 25 cosine terms |
| SPA | Hansen's test of superior predictive ability |
| MCS | Model Confidence Set |
| GW | Giacomini–White conditional predictive-ability test |
| GR | Giacomini–Rossi fluctuation test |
| MZ | Pooled Mincer–Zarnowitz calibration regression |
| FKO | Fleming–Kirby–Ostdiek performance fee |
| Recalibration R1 | Map that multiplies a forecast by a scalar fit on past pairs only |
| Recalibration R2 | Map that shrinks a forecast toward the trailing realized-covariance mean, then rescales |
| QLIKE | Multivariate Gaussian likelihood loss, the primary score |
| A1 | Persistence of selection-period loss differentials, and the refit-cycle pattern |
| A1-addendum | Check of the refit-phase Wald result from A1 |
| A2 | Pooled Mincer–Zarnowitz calibration on the selection period |
| A3 | Signature plot and return-sign asymmetry before the holdout |
| A4 | Leverage asymmetry in selection-period variance and in LSTM-BEKK |
| B1 | Choice of the confirmatory pairwise test. The equal-weighted cosine test was selected |
| B1b | Size-adjusted critical values for that test, frozen before the holdout |
| B2 | Selection-period SPA and model confidence set at longer bootstrap blocks |
| C1 | Carry-forward rule when a DCC refit fails |
| C2 | BEKK family and the ablation that removes the LSTM intercept |
| C3 | Out-of-sample recalibration, the R1 and R2 maps |
| C4 | Semivariance panel and SHAR-DRD |
| D1 | One further step on the LSTM-BEKK and EWMA grids when the winner sat on the edge |
| D2 | LSTM-BEKK variants that add realized covariance as an input, a training target, or both |
| D3a | Shifted-day panels used only as extra training rows |
| E1 | Global-minimum-variance portfolio evaluation on the selection period |
| G1 | Rank rule that picks one econometric and one deep-learning finalist from selection-period scores |
| X3 | Exploratory influence of 2025-10-10 on the holdout ranking |
