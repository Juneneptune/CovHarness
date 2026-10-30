# D1 report

## 1. Outcome

D1. Implemented and executed. One-factor VALIDATION search around LSTM19, plus EWMA decays below 0.80, selected EWMA21 (decay 0.60) and LSTM22 (depth 7, other LSTM19 settings held fixed). The declared outward steps, EWMA25 at decay 0.55 and LSTM28 at depth 8, were fit and lost. After those steps both winners have a searched neighbor on each side of the moved factor, so both are interior on the extended one-factor grids. The selected identifiers changed, so both were scored on SCREEN. LSTM19 remains the deep-learning representative. No SPA, MCS, or finalist reselection was run. These SCREEN losses are selection results.

## 2. Starting state

Branch `main` after the B1 report commit `218d9015defb2dea1434cbb44b27a851bd55851d`. Documents read before the fit were amendment R-2026-10-29, the D1 register entry, the LSTM-BEKK and EWMA grids in `configs/binance_open_data_core.yaml`, `results/binance_validation_candidate_summary.csv`, and `results/binance_validation_selection.json`. Stored VALIDATION QLIKE is \(-32.73154867789014\) for EWMA01 and \(-32.285398774051586\) for the LSTM19 ensemble. Those two configurations were not refit.

## 3. Spec commit hash

`e4f8328eacdf6088eb8765bae026f17e97c1464b`, recorded before any D1 fit. Commit `ef6cb44` only writes that hash into the entry script.

## 4. Changes

New identifiers are in `configs/binance_post_screen_d1.yaml`. The frozen core YAML was not edited. The entry script is `scripts/post_screen_d1_edge.py`. Selection minimizes ensemble mean QLIKE, with an exact tie broken by the lexicographically smaller identifier. LSTM ensembles average the five seed covariance matrices before scoring. Seeds are 0 through 4.

The seven planned LSTM configurations and the four planned EWMA decays were fit. EWMA21 won the first EWMA round, so the declared step EWMA25 was fit. LSTM22 won the first LSTM round, so the declared step LSTM28 was fit. LSTM27 did not win, so no further learning rate was invented. Ridge was not extended, because a penalty of 0 is ordinary least squares. XGB08 was not extended, because its depth and learning rate sit between other points of the frozen grid. The searched totals are EWMA 25 and LSTM-BEKK 28.

The paper-range constructor still rejects depth outside 3 to 5 and dropout outside \([0, 0.2]\). Cells that leave that range pass `architecture_range` `post_screen_d1`, which admits depth through 8 and dropout through 0.5. The default path is unchanged. The register reference is the task D1 spec.

The summary booleans `ewma_interior` and `lstm_interior` are false because they tag the first-round edge identifiers. After the one declared outward step, decay 0.60 lies between searched decays 0.55 and 0.65, and depth 7 lies between searched depths 6 and 8. Both outward steps lost. The winners are interior on those extended axes.

## 5. Problems

The first process rejected LSTM21 through LSTM24 before any forecast. The cause was the paper-range constructor. `num_layers` 6 and 7, and dropout 0.3 and 0.4, raised `InvalidModelConfigurationError`. The existing tests still require that default rejection. The fix, committed as `b1d5c70f47ecfe5e46b3759aa065f4bb7e1357e1` before the restart, adds the named opt-in and leaves those tests unchanged. EWMA21 through EWMA24 had already completed and were reloaded from checkpoints. The partial LSTM25 seed from the aborted process was discarded. The restarted process exited 0. No second scientific choice was made.

## 6. Verification

`python -m pytest tests/unit/test_models_lstm_bekk.py::test_num_layers_outside_range_is_rejected tests/unit/test_models_lstm_bekk.py::test_dropout_outside_interval_is_rejected tests/unit/test_models_lstm_bekk.py::test_post_screen_d1_range_admits_declared_edges tests/unit/test_models_lstm_bekk.py::test_identity_is_lstm_bekk -q --tb=short` exited 0 with 4 passed, in 3.83 seconds. The two existing range tests still reject depth 6 and dropout 0.21 on the default path. The new test accepts depth 8 and dropout 0.5 only under the named D1 range, and rejects depth 9, dropout 0.51, and an unknown range name.

`python scripts/post_screen_d1_edge.py` exited 0. The successful process started at 2026-10-30T02:17:55Z and ended at 2026-10-30T10:54:07Z, with elapsed time 30971.427 seconds. Every completed configuration has complete support and repair count 0. The aborted process, about six minutes, produced no retained LSTM forecast. SPA, MCS, finalist reselection, and CONFIRM were not run. The existing scheduled runners were used, so no new feature timing was added. The full package suite was not rerun.

## 7. Artifacts

`results/post_screen/D1/d1_summary.json` sha256 `80b0e3bcda0ba10949abc9940cbacc09b787bd8e52c984a3994fc251a9fd06c0`. `results/post_screen/D1/d1_validation_table.csv` sha256 `1df94e4742e9b37c436a66512fcb79ecfd44e894cc904282a239424d886a95d9`. `results/post_screen/D1/d1_artifact_hashes.json` sha256 `4540d24097078c14ace3befb46001734dc96ae0c041b85521befd91f2febf802`. Forecast NPZ files, checkpoint JSON files, and `progress.log` are gitignored. Their local hashes are in the artifact-hash file. The 2026-09-21 diagnostics, the stored validation summary, and `configs/binance_open_data_core.yaml` were not modified.

VALIDATION QLIKE. Lower is better. Seed SD is the sample standard deviation of the five seed means. The ensemble column is the loss of the equal-weight covariance mean. EWMA has no seeds. LSTM19 and EWMA01 are the stored baselines.

| ID | Setting | Seed mean | Seed SD | Ensemble or mean QLIKE | Runtime seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| LSTM19 | stored baseline | -32.158756 | 0.102656 | -32.285399 | 2480.422 |
| LSTM21 | depth 6 | -32.030364 | 0.096924 | -32.102652 | 2776.610 |
| LSTM22 | depth 7 | -32.200674 | 0.110091 | -32.322996 | 3096.589 |
| LSTM23 | dropout 0.3 | -32.174953 | 0.050193 | -32.276087 | 2451.685 |
| LSTM24 | dropout 0.4 | -32.166955 | 0.052919 | -32.260781 | 2437.143 |
| LSTM25 | 100 epochs | -32.145315 | 0.113394 | -32.279144 | 3260.041 |
| LSTM26 | 150 epochs | -32.102009 | 0.133250 | -32.278395 | 4865.645 |
| LSTM27 | learning rate 0.002 | -32.087814 | 0.191451 | -32.271197 | 2446.283 |
| LSTM28 | depth 8 | -32.159373 | 0.111297 | -32.259907 | 3436.868 |
| EWMA01 | stored decay 0.80 |  |  | -32.731549 | 0.175 |
| EWMA21 | decay 0.60 |  |  | -32.809792 | 0.002 |
| EWMA22 | decay 0.65 |  |  | -32.808820 | 0.002 |
| EWMA23 | decay 0.70 |  |  | -32.798170 | 0.002 |
| EWMA24 | decay 0.75 |  |  | -32.774342 | 0.002 |
| EWMA25 | decay 0.55 |  |  | -32.802975 | 0.235 |

EWMA21 through EWMA24 runtimes are checkpoint reloads. Those four decays were fit in the aborted process and reloaded with the same QLIKE. Exact seed means and Frobenius losses are in the CSV.

SCREEN, 500 targets, repair count 0. Stored 2026-09-21 scores are shown for comparison and were not recomputed.

| ID | QLIKE | Frobenius | Runtime seconds |
| --- | ---: | ---: | ---: |
| EWMA01 stored | -31.776804 | \(7.800251\times 10^{-5}\) | 0.374 |
| EWMA21 | -31.784158 | \(8.032475\times 10^{-5}\) | 0.376 |
| LSTM19 stored | -31.408752 | \(8.269241\times 10^{-5}\) | 4919.511 |
| LSTM22 | -31.297632 | \(8.369783\times 10^{-5}\) | 6194.704 |

On this selection period, EWMA21 has lower QLIKE than the stored EWMA01 score and higher Frobenius loss. LSTM22 has higher QLIKE and higher Frobenius loss than the stored LSTM19 score. That comparison does not change the representative.

## 8. Docs

The register row for D1 is completed. Spec commit `e4f8328eacdf6088eb8765bae026f17e97c1464b`. Result commit `afb951a005b6ad22e1b12097eb579ef01aef852f`. `docs/PROJECT_STATE.md` names C4 as the next planned task and records the D1 selection. `README.md` records EWMA21, LSTM22, and the named constructor range. `docs/MODEL_IMPLEMENTATION.md` records the same constructor range.

## 9. Next task

C4, named and not started.
