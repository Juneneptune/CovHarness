# Task A3 report

## 1. Outcome

Task A3. Executed. Realized covariance was rebuilt at 1, 5, 15, and 30 minutes on HISTORY_A through SCREEN from the existing measurement path. Mean pairwise correlation rises from 0.654 at 1 minute to 0.708 at the frozen 5-minute grid and 0.725 at 30 minutes, while mean realized variance changes little. That is the Epps pattern on this panel. It does not change the proxy. Daily tail counts are compatible with equal up and down counts at 5 percent. Five-minute 95th-percentile counts have more down moves than up moves for four of the five names. Mean downside semivariance exceeds upside semivariance for every name, and the HAC interval for the gap excludes 0 for BNB and LTC.

## 2. Starting state

Branch `main`. Spec commit `c30b27a5739d803458006853c2d8bbc80c80d32b` was the HEAD before implementation. The working tree was clean. Documents and artifacts read were the Post-SCREEN register, `docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md`, `src/covharness/data/binance_klines.py`, `src/covharness/data/binance_calendar.py`, `src/covharness/realized/binance_panel.py`, and the production panel through 2025-04-12 only.

## 3. Spec commit

`c30b27a5739d803458006853c2d8bbc80c80d32b`, recorded before any A3 empirical number was computed.

## 4. Changes

`interval_end_minutes` and `dates_through_screen` were added in `covharness.data.binance_calendar`. `grid_log_returns` is the only log-difference implementation. `five_minute_log_returns` calls it. Interval 5 returns the frozen `FIVE_MINUTE_END_MINUTES` object. `measure_day_on_interval` and `build_preconfirm_interval_measurements` use that path. `realized_semivariances` sits next to `realized_covariance`. Tail counts and skewness are in `covharness.realized.sign_asymmetry`. The entry script is `scripts/post_screen_measurement.py`.

Scientific choices are the A3 register entry. Dates are 2021-11-09 through 2025-04-12. The halt is excluded as a row and kept as the 23:59 anchor for 2023-03-25. Grids are 1, 5, 15, and 30 minutes with the prior-day 23:59 anchor and no forward fill. Tail probabilities are 0.95 and 0.99, NumPy linear quantiles, strict exceedances, per asset, with a pooled table marked secondary. The binomial null is 1/2 with no multiplicity adjustment. Semivariance HAC uses the automatic Newey–West 1994 lag and critical value 1.96. The frozen proxy and the production panel file were not modified.

## 5. Problems

The first fast test run failed. One assertion indexed the synthetic cube one day late, so the five-minute wrapper comparison was not on the same day. A second assertion required bitwise equality between the partitioned sum of squares and the BLAS Gram diagonal. Those two reductions differ by at most about 6e-18 on the synthetic day. The day index was corrected. The sign partition is now checked elementwise, and the sum is compared with the Gram diagonal at relative tolerance 1e-12 and absolute tolerance 1e-15, the same order of tolerance already used for the Gram-versus-loop check in the measurement tests. The rerun passed. The production-panel comparison remained bitwise.

## 6. Verification

Fast tests.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_post_screen_a3.py tests/unit/test_binance_measurement.py -k "not matches_production"`

Exit code 0. 19 passed, 1 deselected.

Production parity.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_post_screen_a3.py::test_five_minute_preconfirm_matches_production_panel`

Exit code 0. 1 passed in 20.22 seconds. `np.array_equal` against production RCov on the 1250 pre-CONFIRM dates.

Empirical script.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python scripts/post_screen_measurement.py`

Exit code 0. Return counts are 1440, 288, 96, and 48. The elementwise semivariance partition matched the squared five-minute returns. No measurement failure and no repair. The loader end bound is 2025-04-12 23:59. CONFIRM rows copied are 0. The halt is not a measured row. `unlock_confirm` was not passed.

## 7. Artifacts

| Path | SHA-256 |
| --- | --- |
| `results/post_screen/A3/a3_summary.json` | `6c97200323cf7101b4f4a27f84bf0891ce633af2984283ed56cbc6b12b29c195` |
| `results/post_screen/A3/a3_signature.csv` | `766a553dbe7ec6ed13cf11382bc477058412ae77a9bc9c79b971d2dce438d6e7` |
| `results/post_screen/A3/a3_asymmetry.csv` | `3995cd26ca78d574b044572e3b8e2f466dd3129d717e3b9c768d327604ed469d` |
| `results/post_screen/A3/a3_semivariance.csv` | `c12ca4a29ebe220e2eba5b20b0a04395629711461a5a39409e5a88dd457e7066` |
| `results/post_screen/A3/signature_plot.png` | `7b3cba3e9e375d3ccf9366d2086053b16146f18383f5a7d3c6748ad9bdea0c9f` |

Spec commit `c30b27a5739d803458006853c2d8bbc80c80d32b`. Result commit `af3d14a2dca15cdf589586151bb8414e0ed13ee1`. Production panel SHA-256 remains `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`. The 2026-09-21 SCREEN artifacts, committed A1 files, and `configs/binance_open_data_core.yaml` were not modified.

Mean pairwise correlation and \(M/N\).

| Interval minutes | M | M/N | Mean pairwise correlation |
| --- | --- | --- | --- |
| 1 | 1440 | 288 | 0.654 |
| 5 | 288 | 57.6 | 0.708 |
| 15 | 96 | 19.2 | 0.722 |
| 30 | 48 | 9.6 | 0.725 |

Five-minute mean gap \(\mathrm{RS}^{+}-\mathrm{RS}^{-}\), HAC lag 7.

| Asset | Mean gap | 95 percent interval |
| --- | --- | --- |
| BTCUSDT | -6.55e-6 | [-2.57e-5, 1.26e-5] |
| ETHUSDT | -1.83e-5 | [-4.70e-5, 1.04e-5] |
| BNBUSDT | -2.82e-5 | [-4.86e-5, -7.76e-6] |
| LTCUSDT | -1.01e-4 | [-1.77e-4, -2.39e-5] |
| ADAUSDT | -5.14e-5 | [-1.23e-4, 2.03e-5] |

## 8. Docs

The register status for A3 is completed. The result commit cell holds `af3d14a2dca15cdf589586151bb8414e0ed13ee1`. `docs/PROJECT_STATE.md` records the hash, the correlation path, and this report. `README.md` names `scripts/post_screen_measurement.py` and states that the command does not change the frozen five-minute proxy.

## 9. Next task

A4, whether LSTM-BEKK reproduces a leverage asymmetry on SCREEN. Not started.
