# G1 report

## 1. Outcome

G1. Executed. Amendment R-2026-10-30b was recorded before any block rank was computed. The frozen median-block-rank rule, applied to saved raw reduced QLIKE, selects HARQDRD01 from the econometric pool and V01 from the deep tournament. The headline pair is HARQDRD01 against V01. The descriptive SPA and MCS rerun does not change that pair. The freeze list is amendment R-2026-10-30c. The candidate protocol is `PREREGISTRATION_CANDIDATE.md`. CONFIRM was not unlocked, and no CONFIRM-dated panel was built.

Econometric pool. Lower reduced QLIKE receives rank 1. Ranks are the average of occupied positions when a block mean ties.

| Model | Block ranks | Median | Mean rank | Mean QLIKE |
| --- | --- | ---: | ---: | ---: |
| RW01 | 5, 10, 4, 5 | 5.0 | 6.00 | −31.270726 |
| EWMA21 | 1, 5, 1, 4 | 2.5 | 2.75 | −31.784158 |
| HARDRD01 | 4, 2, 3, 2 | 2.5 | 2.75 | −31.759431 |
| HARQDRD01 | 3, 1, 2, 3 | 2.5 | 2.25 | −31.862063 |
| SHARDRD01 | 2, 3, 5, 1 | 2.5 | 2.75 | −31.733461 |
| LWLIN01 | 9, 7, 9, 9 | 9.0 | 8.50 | −30.235152 |
| LWNL01 | 10, 6, 10, 10 | 10.0 | 9.00 | −30.162208 |
| DCC02 | 8, 9, 8, 7 | 8.0 | 8.00 | −30.878770 |
| DCCNL02 | 7, 8, 6, 6 | 6.5 | 6.75 | −30.907995 |
| BEKKS01 | 6, 4, 7, 8 | 6.5 | 6.25 | −31.189797 |

HARQDRD01 wins. EWMA21, HARDRD01, HARQDRD01, and SHARDRD01 share median rank 2.5. The mean-rank tie-break selects HARQDRD01 at 2.25 against 2.75 for the other three. The 2026-09-21 econometric finalist was also HARQDRD01, in a six-family pool whose median ranks are not the ranks in this ten-model pool.

Deep tournament.

| Model | Block ranks | Median | Mean rank | Mean QLIKE |
| --- | --- | ---: | ---: | ---: |
| LSTM22 | 4, 2, 4, 3 | 3.5 | 3.25 | −31.297632 |
| V10 | 3, 4, 3, 4 | 3.5 | 3.50 | −31.303579 |
| V01 | 1, 1, 2, 1 | 1.0 | 1.25 | −31.515557 |
| V11 | 2, 3, 1, 2 | 2.0 | 2.00 | −31.468871 |

V01 wins. LSTM19 is outside the tournament. Its four block means are −31.301884, −32.896408, −31.862895, and −29.573823, and its full-period mean is −31.408752. It has no tournament rank. The 2026-09-21 deep representative remains LSTM19 by construction.

Descriptive SCREEN SPA and MCS. Benchmark HARDRD01, consistent recentering, B equal to 5000, seed 20260913. MCS alpha is 0.10. The range pair is primary and the max pair is the companion. The universe has 27 complete raw columns. RIDGE01 is an exact benchmark tie on both losses. This table does not change the finalists.

| Loss | Block | Statistic | p lower | p consistent | p upper | Primary MCS | Companion MCS |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| QLIKE | 7 | 3.001030 | 0.0018 | 0.0018 | 0.0028 | EWMA01, EWMA21, HARQDRD01, SHARDRD01 | those four plus HARDRD01, RIDGE01, XGB08 |
| QLIKE | 29 | 3.012720 | 0.0018 | 0.0020 | 0.0038 | EWMA01, EWMA21, HARQDRD01, SHARDRD01 | those four plus HARDRD01, RIDGE01, XGB08 |
| Frobenius | 7 | 0.010172 | 0.6202 | 0.9680 | 0.9830 | 24 models, excluding BEKKF02, LWLIN01, LWNL01 | 26 models, excluding LWLIN01 |
| Frobenius | 29 | 0.012528 | 0.5946 | 0.9490 | 0.9570 | 24 models, excluding BEKKF02, LWLIN01, LWNL01 | all 27 models |

## 2. Starting state

Branch `main` at `ad14c6a`. Untracked `results/post_screen/B1/cells/` and the D2 seed pid files were left untracked. Branch `docs/editorial-pass` was left unmerged. Documents read were the post-SCREEN register, the reports under `results/post_screen/`, the README selection rule, `results/binance_screen_finalists.json`, and `docs/INFERENCE_METHODS.md`.

## 3. Spec commit hash

`cd68fd556add3eb03df9f933e736c11bce4580b8`, recorded before the selection script ran.

## 4. Changes

`PREREGISTRATION_DRAFT.md` gains amendments R-2026-10-30b and R-2026-10-30c. Earlier entries were not edited. `scripts/post_screen_g1_selection.py` ranks saved raw SCREEN losses and reruns SPA and MCS. `PREREGISTRATION_CANDIDATE.md` is the candidate protocol. `README.md`, `docs/PROJECT_STATE.md`, and `docs/INFERENCE_METHODS.md` record the adopted pair and the P5 rule.

The headline pair is HARQDRD01 against V01. The secondary contrasts are V01 minus LSTM22 and V11 minus LSTM22. A pairwise claim requires the ordinary cutoff 2.0595385527532972 and the B1b cutoffs 2.4728093597276533 and 2.4226501420943998. Reduced QLIKE is primary and squared Frobenius is secondary. The five-minute proxy is primary and the 15-minute proxy is robustness. Carry-forward is the primary failure policy and the interior map is robustness. The headline forecast is raw. The E1 squared-return gap remains the risk comparison, and the six Fleming–Kirby–Ostdiek cells are additional. The CONFIRM roster is the 21 identifiers in R-2026-10-30c. D3b, D4, the graph-neural slot, GHAR, and F1 stay out.

The fee equation in the candidate is the Fleming–Kirby–Ostdiek average-utility equality. A positive fee is the daily amount paid to hold the first named book. Two real roots are resolved by the root closer to the mean-variance expression. The daily fee and the fee times 365 are both stored. Gamma and the cost are not selected.

## 5. Problems

None observed.

## 6. Verification

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python scripts/post_screen_g1_selection.py` exited 0. The shell that committed the spec hash and ran the script elapsed 9.529 seconds. The script replayed `econ_median_rank_finalist` on the frozen QLIKE file and matched HARQ-DRD and the stored median ranks. The nine original family columns match the frozen loss file bitwise. C3 raw QLIKE means match `c3_summary.json` within \(10^{-8}\). Every scored target is inside 2023-11-30 through 2025-04-12.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_binance_protocol.py::test_default_confirm_access_raises tests/unit/test_protocol.py::test_confirm_locked_by_default tests/unit/test_protocol.py::test_no_accidental_helper_bypasses_confirm_lock tests/unit/test_binance_screen.py::test_default_confirm_remains_inaccessible tests/unit/test_binance_validation.py::test_default_confirm_remains_locked tests/unit/test_rolling_runner.py::test_confirm_cannot_be_requested_while_locked tests/unit/test_synthetic_benchmark.py::test_confirm_remains_locked` exited 0 with 7 passed in 3.22 seconds. Each test expects `ConfirmLockedError`.

No array under `results/` whose name or dtype marks it as a date contains a date on or after 2025-04-13. Text hits are listed in section 7. No CONFIRM forecast was built. The full package suite was not rerun.

## 7. Artifacts

`results/post_screen/G1/g1_selection.json`, sha256 `633f528a3f75e2f46096e66e135e0a963b8e2c019df226ee3c74c669d51be3ee`. Seed 20260913. Bootstrap size 5000. Block lengths 7 and 29. Benchmark HARDRD01.

The 2026-09-21 finalist, SPA, MCS, and loss files were not modified. `configs/binance_open_data_core.yaml` was not modified. The production panel was not rewritten. Its sha256 remains `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`.

Lock-check text hits on or after 2025-04-13, and inside the confirmation window through 2026-08-25, are prose or catalogs. `results/post_screen/E1/REPORT.md` states that no E1 target is on or after 2025-04-13. `results/binance_screen_interpretation.md` names the locked block. `results/binance_five_asset_panel_validation.json` records `date_end` 2026-08-25 for the production panel, which already stores the locked calendar. `results/data_gate_wrds_access_inventory_20260914.json` records WRDS and TAQ table dates, including a latest table date 2026-08-19. Those files were not modified and are not Binance forecast scores. Document dates in October 2026 appear in other reports and are not market rows.

## 8. Docs

R-2026-10-30b is completed. Spec `cd68fd556add3eb03df9f933e736c11bce4580b8`. Result `0d7f8beb0473078cc41c1c4b1ab20b640033bcda`. R-2026-10-30c is completed. Spec and result `2186809e3e80bba7686b6bb2467dfff1d2b35a4b`. The candidate commit is `804404065b330eb649ffcc1de8801da46a4fb2dd`. `docs/PROJECT_STATE.md` and the README selection, economic, and limitations sections now record the pair. `docs/INFERENCE_METHODS.md` records P5 as the candidate pairwise rule.

## 9. Next task

The decision on whether `PREREGISTRATION_CANDIDATE.md` becomes `PREREGISTRATION.md`. Not started. CONFIRM remains locked.
