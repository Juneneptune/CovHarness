# B1 report

## 1. Outcome

B1. Implemented and executed. The pre-registered size rule selected P5, the equal-weighted cosine test with \(\nu=\lfloor 0.4 T^{2/3}\rfloor\) and Student-\(t\) critical values. No candidate stayed inside \([0.03, 0.075]\) on every required cell. P5 had the smallest maximum absolute distortion, 0.0334, and that fact is a limitation. The choice was made on simulated series and on a resampled SCREEN gap. It was not made by which procedure favours a model. The frozen procedure was then applied to the four A1 pairs as a labeled selection-period demonstration. CONFIRM was not used.

## 2. Starting state

Branch `main` at spec commit `3ccffdbd84398a82879cd16d7da8d41c6ba8c5b6`, working tree clean apart from local C1 checkpoint files that are gitignored. Documents read were amendment R-2026-10-29, the B1 spec, `docs/INFERENCE_METHODS.md` through the existing calibration seed, Kiefer and Vogelsang (2005) Table I, Lazarus, Lewis, Stock, and Watson (2018) equation (10), and `src/covharness/inference/{hac.py,dm.py,size.py,bootstrap_mean.py}`.

## 3. Spec commit hash

`3ccffdbd84398a82879cd16d7da8d41c6ba8c5b6`, recorded before any B1 simulation. The freeze commit, written before the SCREEN demonstration, is `2aae89c62d7c5e5113e4c34e62fefcf8100c3c7f`.

## 4. Changes

`hac.py` adds the LLSW square-root lag, the Andrews (1991) AR(1) plug-in lag, linear interpolation of Kiefer and Vogelsang (2005) Table I, and the equal-weighted cosine variance. `dm.py` adds opt-in keywords. The default call remains Bartlett, Newey-West 1994, and \(N(0,1)\). `size.py` adds variance-1 Student-\(t(5)\) AR(1) paths. The calibration seed is the existing constant 20260916. The entry script is `scripts/dm_procedure_calibration.py`. P6 uses the existing inner bootstrap count 5000 and block length \(\lfloor 1.3\sqrt{T}\rfloor\).

## 5. Problems

The first calibration start failed in 1.418 seconds because the SCREEN loss file labels families, not configuration identifiers. The gap was rebuilt from `HARQ-DRD` and `LSTM-BEKK`, which are the columns for HARQDRD01 and LSTM19. The rerun exited 0. No second scientific choice was made.

## 6. Verification

`python -m pytest tests/unit/test_fixed_b_critical.py tests/unit/test_inference.py -q --tb=line` exited 0 with 25 passed and 1 warning, in 4.16 seconds. The table test matches Table I at \(b=0.02\) and at the analytical \(b=1\) values 3.764, 4.771, and 6.090. The statsmodels HAC reference test is included. Lag checks give \(\lfloor 1.3\sqrt{250}\rfloor=20\) and \(\lfloor 1.3\sqrt{500}\rfloor=29\). `python scripts/dm_procedure_calibration.py` exited 0 in 1358.805 seconds. `python scripts/dm_procedure_calibration.py --demonstrate` exited 0 after the freeze commit. SPA, MCS, and finalist selection were not run. CONFIRM was not requested.

Required two-sided 5 percent rejection rates at \(T=500\). The last two entries of each row are the D3 block lengths 7 and 29.

| Procedure | D1 0 | D1 0.3 | D1 0.6 | D2 0 | D2 0.3 | D2 0.6 | D3 7 | D3 29 | Qualifies | Max distortion | Average power |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| P1 | 0.0560 | 0.0750 | 0.1118 | 0.0498 | 0.0726 | 0.1020 | 0.1130 | 0.1110 | no | 0.0630 | 0.9685 |
| P2 | 0.0624 | 0.0628 | 0.0786 | 0.0586 | 0.0686 | 0.0758 | 0.1036 | 0.0934 | no | 0.0536 | 0.9581 |
| P3 | 0.0506 | 0.0492 | 0.0576 | 0.0510 | 0.0494 | 0.0632 | 0.0870 | 0.0818 | no | 0.0370 | 0.9374 |
| P4 | 0.0420 | 0.0604 | 0.0644 | 0.0424 | 0.0592 | 0.0724 | 0.1098 | 0.1068 | no | 0.0598 | 0.9444 |
| P5 | 0.0504 | 0.0522 | 0.0520 | 0.0466 | 0.0504 | 0.0556 | 0.0834 | 0.0774 | no | 0.0334 | 0.9316 |
| P6 | 0.0816 | 0.0846 | 0.0904 | 0.0804 | 0.0756 | 0.0942 | 0.1192 | 0.1172 | no | 0.0692 | 0.9631 |

P5 at \(T=500\) and \(\rho=0.8\) rejects 0.0676 under D1 and 0.0680 under D2. At \(\rho=0.9\) the rates are 0.1170 and 0.1124. Those cells are reported and do not set the requirement.

## 7. Artifacts

`results/post_screen/B1/b1_calibration.json` sha256 `1d695bd9fb66b4cd57bc40a8413b6be1438941862657163706f20aa82f3cbfa3`. `results/post_screen/B1/b1_demonstration.json` sha256 `4b4914259e4aa177cd6f0b6f3d78d7e58540de98d1eb2e5eedb1eea9457ccc1c`. Cell checkpoints and `progress.log` are local. The 2026-09-21 loss files were read and not modified. `configs/binance_open_data_core.yaml` was not modified.

SCREEN demonstration under P5. The rolling mean uses the frozen 21-origin cadence. A negative differential means the left model has the lower loss that day. These are selection-period descriptions.

| Loss | Pair | Statistic | Critical value | Reject at 5 percent | End cumulative mean | End 21-day mean | Win rate |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| reduced QLIKE | HARQ-DRD minus LSTM-BEKK | -3.625 | 2.060 | yes | -0.453310 | -0.617547 | 0.696 |
| reduced QLIKE | HARQ-DRD minus HAR-DRD | -2.692 | 2.060 | yes | -0.102632 | 0.017544 | 0.628 |
| reduced QLIKE | HAR-DRD minus XGBoost-DRD | -0.636 | 2.060 | no | -0.047899 | 0.000619 | 0.494 |
| reduced QLIKE | HARQ-DRD minus EWMA | -1.021 | 2.060 | no | -0.085259 | 0.327009 | 0.458 |
| squared Frobenius | HARQ-DRD minus LSTM-BEKK | -1.950 | 2.060 | no | -7.81478e-06 | -7.62389e-07 | 0.520 |
| squared Frobenius | HARQ-DRD minus HAR-DRD | -0.009 | 2.060 | no | -1.57080e-08 | 2.37953e-06 | 0.622 |
| squared Frobenius | HAR-DRD minus XGBoost-DRD | -0.872 | 2.060 | no | -3.66704e-06 | -6.80379e-06 | 0.444 |
| squared Frobenius | HARQ-DRD minus EWMA | -1.134 | 2.060 | no | -3.12488e-06 | 3.22895e-06 | 0.512 |

The end cumulative mean is the full-sample mean. The 21-day column is the mean of the last 21 differentials. \(\nu=25\) on every row.

## 8. Docs

The register row for B1 is completed. Spec commit `3ccffdbd84398a82879cd16d7da8d41c6ba8c5b6`. Freeze commit `2aae89c62d7c5e5113e4c34e62fefcf8100c3c7f`. Result commit `9169b92ad442ffa95d4f7a317aff55484d520463`. `docs/PROJECT_STATE.md` names D1 as the next task. The README pairwise section records P5 as the frozen confirmatory procedure and states that the default automatic-lag test is unchanged.

## 9. Next task

D1, named and not started.
