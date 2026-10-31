# C3 report

## 1. Outcome

C3. Implemented and executed. Saved SCREEN forecasts were rescaled by a past-only scalar map and a past-only shrinkage map. No model was refit. The layer is diagnostic for every scored column, and the headline forecast remains raw. After shrinkage, the pooled-vech slope is 0.8332883426958895 for LSTM19, 0.7829200718545408 for LSTM22, and 0.8375116875703978 for V11. The stored A2 raw LSTM-BEKK slope is 0.764. The slopes move toward 1 and remain below 1.

## 2. Starting state

Branch `main` at `987a6f6`. Untracked `results/post_screen/B1/cells/` and the D2 seed pid files were left untracked. Documents and code read were the post-SCREEN register, including the R-2026-10-29 recalibration amendment, the A2 report numbers cited in the task, `src/covharness/evaluation/score.py`, `src/covharness/losses/qlike.py`, `src/covharness/inference/mz.py`, and the saved SCREEN and checkpoint forecast cubes. VALIDATION forecast dates were inspected and were not concatenated into the SCREEN fit.

## 3. Spec commit hash

`4b69ed387c2057abd7917dd77ae2a053109976dd`, recorded before the SCREEN run. The following commit `983b9b9f65ed22c61ad864a383a10feaeec6b97a` stores that hash in the entry script.

## 4. Changes

`src/covharness/evaluation/recalibration.py` owns the forecast-sequence transform. It is not a loss. The fitting set at an origin keeps pairs whose target date is on or before that origin and whose target segment matches the target being forecast, trailing at most 250 pairs. Coefficients update on the stored 21-origin refit cadence and are held in between. Fewer than 21 pairs leaves the identity. R1 uses the mean whitened trace ratio. R2 shrinks toward the mean of the fitting-set proxies, chooses the grid weight by fitting-set reduced QLIKE, and applies the same closed form. Lambda 1 nests R1. A proxy mean that fails Cholesky uses the identity at that origin and is logged. The register entry is the task C3 spec.

## 5. Problems

none observed

## 6. Verification

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest tests/unit/test_recalibration.py -q` exited 0 with 6 passed, in 3.02 seconds. The tests cover a planted scale, a planted shrinkage of 0.7, a one-day target shift that changes the fitting indices, exclusion of a VALIDATION segment from a SCREEN origin, identity on the first 21 rows, and equality of R2 and R1 when the grid is lambda 1. `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python scripts/post_screen_recalibration.py` exited 0. Runtime 493.4172896640375 seconds. Wall clock 496854 milliseconds. Every column has 500 targets, 21 identity origins, zero proxy-mean failures, and zero QLIKE and squared-Frobenius failures. All 500 target dates are in the SCREEN segment. VALIDATION pairs were not used. CONFIRM was not read. SPA, MCS, and finalist selection were not run.

## 7. Artifacts

`results/post_screen/C3/c3_summary.json` sha256 `ca4748e4fdb6ff885b6780f0a9566b44564538a7a06ee8e1f0aa5ff823019109`. `results/post_screen/C3/c3_paths.csv` sha256 `1b611bfde37163bea926664cbcbe0b87d77406325bc798db3f740d16ed687570`. Local forecast cube `results/post_screen/C3/recalibrated_forecasts.npz` sha256 `88503b6012129b6afd0a676f145b41fa7346ad34dc4c20638d420b37bd1b7878`. That NPZ is gitignored. Spec commit `4b69ed387c2057abd7917dd77ae2a053109976dd`. DCC01 and DCCNL01 are recorded as having no SCREEN forecasts. The frozen SCREEN forecast NPZ, the production panel, and `configs/binance_open_data_core.yaml` were not modified. Earlier post-SCREEN results were not modified.

HARQDRD01 raw mean QLIKE is −31.862062800706465, matching the stored loss-file mean. LSTM19 gaps against HARQDRD01 are 0.4533104090666491 raw, 0.4307809914046139 under R1, and 0.38435916112754853 under R2. LSTM22 gaps are 0.5644309223244797, 0.5597782629450947, and 0.4880797710027984. V11 gaps are 0.39319141242919287, 0.3754543812754392, and 0.3531969651631037. Means for every model and version are in the summary JSON.

## 8. Docs

The C3 register row is completed. The result commit is `cd7fa2630fe5488c3c4f59d3f14572bcf60aa67b`. `docs/PROJECT_STATE.md` records the checkpoint and names E1 next. `README.md` defines the two maps as a diagnostic layer and removes them from the unimplemented list.

## 9. Next task

E1, global-minimum-variance evaluation from the saved forecasts. Not started.
