# Task R2 report

## 1. Outcome

Task R2. Read-only with respect to data and models. Amendment R-2026-10-30 is in the Post-SCREEN register. It adopts LSTM22 and EWMA21 as the family configurations, records the depth-7 departure and the P5 size limitation, states the DCC carry-forward caveat, defers D3b and D4, and fixes D2's V00 as the already scored LSTM22. No model was fit.

## 2. Starting state

Branch `main` at `d8281d0`. The working tree was clean apart from untracked `results/post_screen/B1/cells/`, which was left untracked. Branch `docs/editorial-pass` was left unmerged. Documents read were the register, `docs/PROJECT_STATE.md`, `README.md`, and `results/post_screen/OVERNIGHT_SUMMARY_2.md`.

## 3. Spec commit

`9d50128d0a01a80396929381314aac9fc4b01125`. For this documentation task the spec and the result are that commit.

## 4. Changes

`PREREGISTRATION_DRAFT.md` gains amendment R-2026-10-30 and table rows R2 and B1b. D3b and D4 are marked deferred. R1, C1, B1, and D1 were already completed with the hashes in `OVERNIGHT_SUMMARY_2.md`. Earlier task narratives were not rewritten. `docs/PROJECT_STATE.md` sets the next planned tasks to C3, E1, and G1. `README.md` records the family configurations, the depth-7 departure, and the P5 size rates.

## 5. Problems

None observed.

## 6. Verification

The amendment was read back after the commit. No test suite was run. No CONFIRM row was written. `unlock_confirm` was not passed. D3b and D4 were not started.

## 7. Artifacts

`results/post_screen/R2/REPORT.md`. Commit `9d50128d0a01a80396929381314aac9fc4b01125`. No 2026-09-21 artifact was modified. `configs/binance_open_data_core.yaml` was not modified. Committed post-screen results from earlier tasks were not modified.

## 8. Docs

The register status for R2 is completed. Spec and result commit `9d50128d0a01a80396929381314aac9fc4b01125`. `docs/PROJECT_STATE.md` names C3, E1, and G1. `README.md` records LSTM22, EWMA21, and the P5 size limitation.

## 9. Next task

B1b, the size-adjusted P5 critical values. Not started in the R2 commit.
