# C1 report

## 1. Outcome

C1. Implemented and executed. Carry-forward is the primary stationarity policy and the interior softmax map is the labeled robustness variant. VALIDATION and SCREEN were run for DCC02, DCCNL02, DCC03, and DCCNL03 only. All four produced complete trajectories. DCC01 and DCCNL01 remain the failed selection columns. No SPA, MCS, or finalist selection was run. These SCREEN losses are selection results, not a confirmatory ranking.

## 2. Starting state

Branch `main` at spec commit `ed7d2a4761fa5702e7c18e55c48eb2d7cc0c50b7`, working tree clean. Documents and code read before the fit were amendment R-2026-10-29, the C1 register entry, `src/covharness/models/dcc.py`, the VALIDATION and SCREEN rolling runners, and `results/binance_screen_failures.json`. The published failure is origin 2023-12-20, asset index 1, \(a=0\), \(b=1\).

## 3. Spec commit hash

`ed7d2a4761fa5702e7c18e55c48eb2d7cc0c50b7`, recorded before any C1 fit.

## 4. Changes

`src/covharness/models/stationarity.py` holds the shared softmax map with \(\varepsilon=10^{-6}\) and the near-boundary cut 0.999. `dcc.py` keeps the default reject path used by DCC01. Carry-forward keeps the last accepted parameters for every stage and rebuilds the filter state on the current window. A first refit with nothing to carry still raises. The interior map estimates stage-one GARCH and stage-two \((\alpha,\beta)\) on the softmax, so \(a+b<1\) holds by construction. Forecasts are not repaired. The entry point is `scripts/post_screen_dcc_failure_policy.py`. Outputs are under `results/post_screen/C1/`. The register reference is amendment R-2026-10-29 and the task C1 spec.

## 5. Problems

The first synthetic window did not hit \(a+b\ge 1\) after the two-asset stage-one demean, so the carry-forward tests did not raise. The cause was a knife-edge series, not the policy. The test window was replaced by seed 4, which `fit_stage_one_garch` rejects for asset 0. The corrected tests passed before the empirical run. The empirical script then exited 0. No second empirical attempt was made.

## 6. Verification

`python -m pytest tests/unit/test_dcc_failure_policy.py tests/unit/test_models_dcc.py -q --tb=line` exited 0 with 53 passed and 1 warning, in 6.02 seconds. The five new tests cover the boundary trigger, constant carried parameters with a moved state, a first-refit failure, an interior map with \(a+b<1\) and a near-boundary count, and the unchanged reject path. `python scripts/post_screen_dcc_failure_policy.py` exited 0 in 78.892 seconds. Repair counts are 0 on every new column. VALIDATION had 0 carry-forward events and 0 near-boundary events. SCREEN carry-forward counts are 9 for DCC02 and 9 for DCCNL02, each beginning at 2023-12-20 with the same asset-1 parameter vector as the published failure. SCREEN near-boundary counts are 14 for DCC03 and 14 for DCCNL03. SPA, MCS, and finalist selection were not run. CONFIRM was not requested.

## 7. Artifacts

`results/post_screen/C1/c1_summary.json` sha256 `2a4ad486944d062841f62405dd9f3a08ba4a8c1b98f4c6afc3c7e75a19a216cb`. `results/post_screen/C1/c1_loss_table.csv` sha256 `15fbc9f4fc7a0f2fe8d155aba2682bfb8c029fa163b7bef5a4e818be429ca118`. Forecast NPZ files and `progress.log` are gitignored. Their local hashes are stored in the summary. The 2026-09-21 diagnostics, the failure JSON, and `configs/binance_open_data_core.yaml` were not modified.

Policy counts.

| Period | ID | Policy | Carry-forward | Near-boundary | Repairs |
| --- | --- | --- | ---: | ---: | ---: |
| VALIDATION | DCC02 | CF | 0 | 0 | 0 |
| VALIDATION | DCCNL02 | CF | 0 | 0 | 0 |
| VALIDATION | DCC03 | IM | 0 | 0 | 0 |
| VALIDATION | DCCNL03 | IM | 0 | 0 | 0 |
| SCREEN | DCC02 | CF | 9 | 0 | 0 |
| SCREEN | DCCNL02 | CF | 9 | 0 | 0 |
| SCREEN | DCC03 | IM | 0 | 14 | 0 |
| SCREEN | DCCNL03 | IM | 0 | 14 | 0 |

SCREEN mean losses beside the existing columns. DCC01 and DCCNL01 stay empty. The new rows are labeled selection.

| ID | Mean QLIKE | Mean squared Frobenius |
| --- | ---: | ---: |
| RW01 | -31.270725513873614 | 0.00010781932629814412 |
| EWMA01 | -31.776804256929157 | 7.800251041053444e-05 |
| HARDRD01 | -31.759430985603913 | 7.489333409048754e-05 |
| HARQDRD01 | -31.862062800706465 | 7.48776260585042e-05 |
| LWLIN01 | -30.235151560818647 | 9.262341039526676e-05 |
| LWNL01 | -30.16220763900416 | 9.202238395666718e-05 |
| DCC01 |  |  |
| DCCNL01 |  |  |
| RIDGE01 | -31.759430985603913 | 7.489333409048754e-05 |
| XGB08 | -31.71153150409633 | 7.85603711558442e-05 |
| LSTM19 | -31.408752391639815 | 8.269241063292736e-05 |
| DCC02 | -30.87876975480985 | 8.292025836857435e-05 |
| DCCNL02 | -30.907994561233394 | 8.292460724261286e-05 |
| DCC03 | -30.862024738956112 | 8.653168394429344e-05 |
| DCCNL03 | -30.877973749306715 | 8.654361573709356e-05 |

## 8. Docs

The register row for C1 is completed. Spec commit `ed7d2a4761fa5702e7c18e55c48eb2d7cc0c50b7`. Result commit `7555cc91925967b531d6e3bba33c8e1bf47e73a2`. `docs/PROJECT_STATE.md` names B1 as the next task. The README DCC paragraph records the two new policies and states that DCC01 and DCCNL01 remain failed.

## 9. Next task

B1, named and not started.
