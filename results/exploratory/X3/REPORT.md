# Crash-day influence on the holdout ranking

These numbers were computed after the holdout from the saved forecasts. No model was refit, and nothing here changes the preregistered result.

On 2025-10-10, a market-wide liquidation pushed realized volatility far above every model's forecast. QLIKE punishes underpredicted risk heavily, so that single day moves 500-day average losses by several points and reorders the ranking.

| HARQ-DRD vs. LSTM-BEKK (realized-covariance target) | All 500 days | Without 2025-10-10 |
| --- | ---: | ---: |
| Mean QLIKE gap | 0.82 | 0.70 |
| Test statistic (EWC) | 2.65 | 2.40 |

The crash day alone contributes about 15% of the summed gap.

Rankings of all 21 models by mean loss, by mean loss without the crash day, and by median loss are in ranking_5min.csv (5-minute realized covariance) and ranking_15min.csv (15-minute). pairs.csv has the same comparison for other model pairs, and loo_influence.csv has the test statistic after dropping each day in turn. The script is scripts/exploratory/x3_crash_day.py.
