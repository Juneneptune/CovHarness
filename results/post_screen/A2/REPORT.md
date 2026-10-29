# Task A2 report

## 1. Outcome

Task A2. Executed. Descriptive pooled Mincer–Zarnowitz calibration was run on the nine complete SCREEN forecast columns. The exercise separates calibration from information and does not rank models. LSTM19, the stored five-seed covariance ensemble, rejects the null \((\alpha,\beta)=(0,1)\) under OLS with slope \(0.764\). Approximate PS21 also rejects at the automatic lag. The mean whitened ratio is \(1.066\) and its HAC interval includes \(1\). That disagreement is a calibration description. Forecasts were not recalibrated and models were not changed.

## 2. Starting state

Branch `main`. Spec was committed at `75240e13873322df8d93f4dfb3bb4a76626a3bb4` before the empirical run. The empirical run was already present at HEAD `80f03b0acc0231a660ae72e13ab1957133fef31f` when this report was written. The working tree was clean apart from this report and the register cell that records that result hash. Documents and artifacts read were the Post-SCREEN register, `docs/INFERENCE_METHODS.md` as cited by the committed spec, `src/covharness/inference/mz.py` through the existing estimator, `scripts/post_screen_mz.py`, `results/binance_screen_forecasts.npz`, and the production panel restricted to SCREEN target rows.

## 3. Spec commit

`75240e13873322df8d93f4dfb3bb4a76626a3bb4`, recorded before any A2 empirical number was computed.

## 4. Changes

Files. `src/covharness/losses/qlike.py` adds `whitened_trace_ratio` and `whitened_trace_ratio_panel`. `scripts/post_screen_mz.py` is the entry point. Outputs are under `results/post_screen/A2/`. Tests in `tests/unit/test_losses.py` cover the ratio. The estimator is the existing `mincer_zarnowitz_pooled_vech`. No new MZ estimator was written.

Scientific choices, from the A2 register entry. Nine complete columns, roster order, LSTM19 as the equal-weight five-seed ensemble. Seeds \(0,1,2,3,4\) are an appendix only. DCC01 and DCCNL01 are recorded as having no forecasts. OLS uses `weighting="none"`. Approximate PS21 uses `weighting="approximate_ps21"`. Unique entries are \(15\) at \(N=5\). Null \((\alpha,\beta)=(0,1)\). Bartlett HAC on daily-summed scores uses the automatic Newey–West 1994 lag, which is \(5\) at \(T=500\), and lag \(29\) as a sensitivity. Neither lag is adopted. Diagonal-only and off-diagonal-only coefficients are descriptive. The whitened ratio is \(r_t=\mathrm{tr}(H_t^{-1}S_t)/N\) from the QLIKE Cholesky solves. `inv(H)` is not formed. HARQ-DRD fallback targets 2024-12-10 and 2025-03-03 stay in the sample and are flagged. Log and square-root transforms were not used. These SCREEN ratios do not set any recalibration constant. Task C3, if later run, fits such a constant on past data only.

## 5. Problems

None observed.

## 6. Verification

Command, this report session.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_inference_mz.py tests/unit/test_losses.py`

Exit code 0. 48 passed.

The committed register entry records the same 48 passed before the SCREEN run. The ratio test draws a synthetic proxy with \(\mathrm{E}[S]=H\) and checks a mean near \(1\), and scaling \(H\) by \(2\) yields a mean near \(0.5\).

Known-truth checks are those unit tests. The production-panel realized covariance copied for SCREEN targets 2023-11-30 through 2025-04-12 matched the stored SCREEN `proxy_rcov` cube exactly. Dates after 2025-04-12 were not copied. CONFIRM remained locked. No `unlock_confirm`. HARQ repair count is 2, on the two flagged dates. Failure and repair counts for the MZ fit are zero. Diebold–Mariano, SPA, and MCS were not run. The empirical script was not rerun for this report. The committed outputs were read.

## 7. Artifacts

| Path | SHA-256 |
| --- | --- |
| `results/post_screen/A2/a2_summary.json` | `e6754bab573580edf9b9bd35bcc0897bc5b50f0ec24589dad1d392d4f0bc9517` |
| `results/post_screen/A2/a2_summary.csv` | `a4b5c2b586e36f03aae02440918d0ac06bc82b2fc1868ba344b54728acb28176` |
| `results/post_screen/A2/a2_sensitivity.csv` | `d24652a20d07ef8679f1e7ef27e0233e213c35ec759c503fd6c9939f6dfe46af` |
| `results/post_screen/A2/a2_appendix_lstm_seeds.csv` | `ba6f24a0125f9eac718f4f1c55e0e2facbda770f8d19225aa9668de60560793c` |
| `results/post_screen/A2/mean_whitened_ratio.png` | `55e440ea802127741d0c41600b63165ded5d310220f6341056f7ab6b2ab7653e` |

Spec commit `75240e13873322df8d93f4dfb3bb4a76626a3bb4`. Result commit `80f03b0acc0231a660ae72e13ab1957133fef31f`. Seeds in the appendix are \(0,1,2,3,4\). LSTM19 is the ensemble, not a selected seed. Untouched frozen artifacts include the 2026-09-21 SCREEN forecast NPZ `118f9eb4f5cfc8f2ffa1b1e73128358725aa179d91ebea8149c1f997a89e6c94`, the production panel `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`, the HARQ checkpoint `d2232e6e3e6aa013ff705ff4519312c3dd72cdfcd0e34eed9d50b714eaa0cc14`, committed A1 files, and `configs/binance_open_data_core.yaml`.

Headline table. \(T=500\), automatic HAC lag \(5\). Mean-ratio intervals are HAC 95 percent intervals at that lag.

| Model | Config | alpha | beta | Wald p OLS | Wald p PS21 | Mean ratio | Mean ratio interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW | RW01 | 0.000704 | 0.435 | 6.21e-24 | 2.55e-17 | 1.498 | [1.401, 1.595] |
| EWMA | EWMA01 | 0.000303 | 0.764 | 6.98e-4 | 1.84e-3 | 1.148 | [1.041, 1.255] |
| HAR-DRD | HARDRD01 | 0.000165 | 0.943 | 0.188 | 0.340 | 1.073 | [0.964, 1.182] |
| HARQ-DRD | HARQDRD01 | 0.000142 | 0.954 | 0.243 | 0.351 | 1.062 | [0.962, 1.161] |
| LW-linear | LWLIN01 | 0.000184 | 1.379 | 9.51e-6 | 3.99e-6 | 1.391 | [1.158, 1.624] |
| LW-NL | LWNL01 | 0.000183 | 1.339 | 2.76e-5 | 1.01e-5 | 1.473 | [1.229, 1.716] |
| Ridge-DRD | RIDGE01 | 0.000165 | 0.943 | 0.188 | 0.340 | 1.073 | [0.964, 1.182] |
| XGBoost-DRD | XGB08 | 0.000338 | 0.764 | 6.94e-3 | 0.209 | 1.098 | [0.998, 1.198] |
| LSTM-BEKK | LSTM19 | 0.000498 | 0.764 | 9.40e-5 | 0.0280 | 1.066 | [0.947, 1.185] |

Ridge-DRD matches HAR-DRD. HAR-DRD and HARQ-DRD do not reject \((0,1)\) at 5 percent under OLS or approximate PS21 at lag 5 or lag 29. Their mean-ratio intervals include 1. HARQ-DRD ratios on the fallback dates are 2.353 on 2024-12-10 and 2.899 on 2025-03-03. All five LSTM seeds reject OLS \((0,1)\). Seed 2 is the only seed whose mean-ratio interval excludes 1. XGBoost-DRD rejects OLS at lag 5 and does not reject approximate PS21. Lag 29 moves the XGBoost OLS \(p\)-value to 0.0587. RW, EWMA, LW-linear, and LW-NL reject both weightings at both lags, and their mean-ratio intervals lie above 1.

## 8. Docs

The register status for A2 is completed. The result commit cell now holds `80f03b0acc0231a660ae72e13ab1957133fef31f`. `docs/PROJECT_STATE.md` records that hash and this report path. `README.md` already names `scripts/post_screen_mz.py` and was not changed for this report.

## 9. Next task

A3, signature plot and return-sign asymmetry on pre-CONFIRM Binance data. Not started in the A2 result commit.
