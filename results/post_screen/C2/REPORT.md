# C2 report

## 1. Outcome

C2. Implemented and executed. Six BEKK(1,1) configurations completed VALIDATION and SCREEN. VALIDATION mean QLIKE selected BEKKS01. The interior-map fit of that configuration is a robustness result and is not headline-eligible. The zero-intercept ablation completed five seeds on both periods. No SPA, MCS, or reselection was run.

## 2. Starting state

Branch `main`. Documents read were the register, including the C1 failure policy and amendment R-2026-10-29, `src/covharness/models/lstm_bekk.py`, and `src/covharness/models/stationarity.py`. `configs/binance_open_data_core.yaml` was not modified.

## 3. Spec commit hash

`23e728a5ebb8c2c3481a19637b9b8e0e8659b3c5`, recorded before any C2 fit.

## 4. Changes

`src/covharness/models/bekk.py` estimates scalar, diagonal, and full BEKK(1,1), with and without covariance targeting, by L-BFGS-B Gaussian QMLE. The scalar step and the Gaussian NLL are imported from the LSTM-BEKK module. Carry-forward is primary. The interior map uses the existing softmax persistence helper, once for a scalar pair and once per asset for the diagonal form. LSTMBEKK-noC forces the dynamic intercept to zero and keeps the LSTM22 training settings. The register reference is the task C2 spec.

## 5. Problems

The five ablation seeds produced the same forecast. On VALIDATION the seed QLIKE values match exactly. On SCREEN they differ by at most \(3.97\times 10^{-15}\). The cause is that a zero dynamic intercept removes the LSTM weights from the recursion, so the seed does not enter \(a\), \(b\), or static \(C\). The ensemble is the common matrix. It is not an average over a subset of the seeds.

On SCREEN, BEKKS02 mean QLIKE is −31.217194257539095, lower than the VALIDATION-selected BEKKS01 at −31.189796533774512. The family configuration remains BEKKS01.

## 6. Verification

`python -m pytest tests/unit/test_models_bekk.py tests/unit/test_models_lstm_bekk_variants.py::test_faithful_forecast_is_bitwise_unchanged -q --tb=line` exited 0 with 5 passed, in 6.31 seconds. The checks cover a hand-computed scalar step, positive definiteness of diagonal and full steps, recovery of simulated scalar coefficients within 0.05, and equality of the zero-dynamic-intercept path with the scalar recursion. The faithful forecast on the fixed input remained bitwise identical. `python scripts/post_screen_bekk.py` exited 0. Wall clock of that process was 9069 seconds. Every configuration had repair count 0 and carry-forward count 0. VALIDATION has 250 targets and SCREEN has 500. SPA and MCS were not run.

## 7. Artifacts

`results/post_screen/C2/c2_summary.json` sha256 `124ba27dc121010f4e693c7a6a2f01701f806d9fc3c3acea47546e7357b1d287`. Checkpoints and the progress log were not committed. The production panel and `configs/binance_open_data_core.yaml` were not modified.

VALIDATION mean QLIKE. BEKKS01 is the selected configuration.

| ID | Form | Targeting | QLIKE | Frobenius |
| --- | --- | --- | --- | --- |
| BEKKS01 | scalar | no | −32.151797 | 4.684567e-05 |
| BEKKS02 | scalar | yes | −32.140704 | 4.771425e-05 |
| BEKKD01 | diagonal | no | −31.868608 | 4.891594e-05 |
| BEKKD02 | diagonal | yes | −31.651674 | 5.958642e-05 |
| BEKKF01 | full | no | −31.225384 | 4.784685e-05 |
| BEKKF02 | full | yes | −31.436704 | 5.926363e-05 |

SCREEN mean QLIKE. Full BEKK at \(N=5\) has 65 parameters on a 250-day window when targeting is off, and 50 when targeting is on.

| ID | QLIKE | Frobenius |
| --- | --- | --- |
| BEKKS01 | −31.189797 | 8.174419e-05 |
| BEKKS02 | −31.217194 | 8.020629e-05 |
| BEKKD01 | −31.064068 | 8.888230e-05 |
| BEKKD02 | −31.091383 | 8.292163e-05 |
| BEKKF01 | −30.175845 | 9.245219e-05 |
| BEKKF02 | −30.472588 | 8.916158e-05 |

BEKKS01-INT SCREEN mean QLIKE is −31.190289. The ablation SCREEN ensemble QLIKE is −31.277044.

## 8. Docs

The register row for C2 is completed. Spec commit `23e728a5ebb8c2c3481a19637b9b8e0e8659b3c5`. Result commit `689fd0ccaa7d909fbf49d191df1bfd9a3da97d07`. `README.md` records that the BEKK family joins the econometric pool at G1 and that the ablation does not.

## 9. Next task

C3, E1, and G1, in the order already recorded. None was started.
