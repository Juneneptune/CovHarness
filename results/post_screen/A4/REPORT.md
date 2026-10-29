# Task A4 report

## 1. Outcome

Task A4. Executed. On SCREEN, next-day log realized variance responds more to negative origin-day returns than to positive ones for every asset. The pooled data gap \(g_{\mathrm{pos}}+g_{\mathrm{neg}}\) is \(-4.24\), and its HAC interval excludes 0. LSTM19, the stored five-seed ensemble, has negative \(g_{\mathrm{neg}}\) point estimates, but every per-asset and the pooled gap interval includes 0. The slopes are much smaller than the data slopes. HARQ-DRD was the sign-symmetric control and is not uniformly zero. BTC's HARQ gap interval excludes 0. No model was refit.

## 2. Starting state

Branch `main` at spec commit `d0d3f70547038bee845b2984c1ffdb2349f61650`. The working tree was clean. Documents and artifacts read were the Post-SCREEN register, `src/covharness/models/lstm_bekk.py` for the signed-return input and the sign-symmetric \(a xx'\) term, `src/covharness/models/har_drd.py` for the 1/4/17 blocks, the SCREEN forecast NPZ, and production-panel rows from HISTORY_B through SCREEN.

## 3. Spec commit

`d0d3f70547038bee845b2984c1ffdb2349f61650`, recorded before any A4 empirical number was computed.

## 4. Changes

`covharness.inference.leverage` builds the HAR log-variance design and the OLS Bartlett sandwich. The sandwich reuses `hac_long_run_covariance` and the pooled MZ scaling. The entry script is `scripts/post_screen_leverage.py`. Outputs are under `results/post_screen/A4/`.

Scientific choices are the A4 register entry. Averages of the weekly and monthly blocks are taken on the variance scale and then logged. \(r^{+}=\max(r,0)\) and \(r^{-}=\min(r,0)\). The leverage label is \(g_{\mathrm{neg}}<0\) and \(|g_{\mathrm{neg}}|>g_{\mathrm{pos}}\). The gap interval uses automatic Newey–West lag 5 at \(T=500\) and critical value 1.96. The pooled fit sums asset scores within the day. HARQ fallback dates stay in the sample. CONFIRM was not copied.

## 5. Problems

The first test run failed one assertion. Scaling the variance on an earlier response date changed a later row's monthly lag, which is the correct HAR overlap of a past target with a later lag, not lookahead. The test was rewritten to scale only the last target, and to check that scaling an earlier target leaves the earlier row unchanged. The sign-recovery tests had already passed. The rerun passed 3 tests. The empirical script then exited 0.

## 6. Verification

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_post_screen_a4.py`

Exit code 0. 3 passed.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python scripts/post_screen_leverage.py`

Exit code 0. \(T=500\) SCREEN responses. Lag sample starts 2023-03-25. Halt absent. CONFIRM rows copied are 0. Forecast diagonals were strictly positive. No repair was applied in this task. `unlock_confirm` was not passed. The planted negative gap was recovered with a confidence interval below 0. A planted positive \(g_{\mathrm{neg}}\) stayed positive.

## 7. Artifacts

| Path | SHA-256 |
| --- | --- |
| `results/post_screen/A4/a4_summary.json` | `b20c559b4b1cf2dc5cad9c1e2bffbac55afce38bf6e31a6e25ba74d864a51643` |
| `results/post_screen/A4/a4_summary.csv` | `3455365c17a4aa979f4518c4b12d7b3223a38ceeb069b5bf0487d1f0448b4a39` |

Spec commit `d0d3f70547038bee845b2984c1ffdb2349f61650`. Result commit `71bdbc200c41224545116adfc0aef24ee0936690`. Panel SHA-256 remains `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`. Forecast SHA-256 remains `118f9eb4f5cfc8f2ffa1b1e73128358725aa179d91ebea8149c1f997a89e6c94`. The 2026-09-21 artifacts, committed A1 files, and `configs/binance_open_data_core.yaml` were not modified.

Gap \(g_{\mathrm{pos}}+g_{\mathrm{neg}}\) and HAC interval. The mechanical label is the spec boolean.

| Dependent | Asset | g_pos | g_neg | Gap | Interval | Label |
| --- | --- | --- | --- | --- | --- | --- |
| data | BTCUSDT | 7.01 | -14.81 | -7.80 | [-12.90, -2.69] | yes |
| data | ETHUSDT | 4.04 | -11.90 | -7.86 | [-10.88, -4.83] | yes |
| data | BNBUSDT | 6.22 | -12.98 | -6.75 | [-10.72, -2.79] | yes |
| data | LTCUSDT | 3.59 | -5.68 | -2.09 | [-4.94, 0.76] | yes |
| data | ADAUSDT | 3.81 | -6.28 | -2.47 | [-4.90, -0.04] | yes |
| data | pooled | 4.41 | -8.64 | -4.24 | [-6.87, -1.61] | yes |
| LSTM19 | BTCUSDT | 4.47 | -4.79 | -0.32 | [-1.31, 0.68] | yes |
| LSTM19 | ETHUSDT | 3.77 | -3.39 | 0.37 | [-0.79, 1.54] | no |
| LSTM19 | BNBUSDT | 4.58 | -5.53 | -0.95 | [-2.24, 0.33] | yes |
| LSTM19 | LTCUSDT | 2.64 | -2.74 | -0.09 | [-0.79, 0.60] | yes |
| LSTM19 | ADAUSDT | 3.30 | -3.80 | -0.49 | [-1.70, 0.72] | yes |
| LSTM19 | pooled | 3.51 | -3.77 | -0.26 | [-1.00, 0.48] | yes |
| HARQDRD01 | BTCUSDT | -0.49 | -1.24 | -1.73 | [-2.98, -0.49] | yes |
| HARQDRD01 | ETHUSDT | -0.23 | -0.68 | -0.91 | [-2.19, 0.36] | yes |
| HARQDRD01 | BNBUSDT | 0.28 | -1.15 | -0.87 | [-2.18, 0.43] | yes |
| HARQDRD01 | LTCUSDT | -0.15 | 0.37 | 0.22 | [-1.25, 1.69] | no |
| HARQDRD01 | ADAUSDT | -1.09 | 0.39 | -0.70 | [-2.19, 0.79] | no |
| HARQDRD01 | pooled | -0.70 | -0.002 | -0.70 | [-1.98, 0.58] | yes |

The pooled HARQ label is the mechanical boolean. \(g_{\mathrm{pos}}\) is negative there, so the label is not a large downside slope.

## 8. Docs

The register status for A4 is completed. The result commit cell is filled with `71bdbc200c41224545116adfc0aef24ee0936690` in the report commit. `docs/PROJECT_STATE.md` records the regression and the non-uniform HARQ control. `README.md` names `scripts/post_screen_leverage.py`.

## 9. Next task

B2, SPA and MCS block-length sensitivity on SCREEN. Not started.
