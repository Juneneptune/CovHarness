# W1 confirmatory write-up review note

## 1. Outcome

Task W1. Implemented. The post-CONFIRM status is drafted on branch `docs/confirm-writeup` at commit `509be2ad3d04b2c443e8719a1fc5ac894b5928ce` and is not merged. This note lists the sections changed so the draft can be reviewed. The branch does not change any confirmatory artifact.

## 2. Starting state

Branch `main` at `a2e5e94` after the T3 rename commit. The working tree was clean. Documents read were `README.md`, `docs/PROJECT_STATE.md`, `results/confirm/REPORT.md`, `results/confirm/exploratory/X1_REPORT.md`, and `results/confirm/exploratory/X2_REPORT.md`.

## 3. Spec commit hash, recorded before any empirical run

`a2e5e94`. No empirical run. The numbers in the draft are copied from the committed CONFIRM report and the exploratory notes.

## 4. Changes

On `docs/confirm-writeup` only.

`README.md`. The opening sentence now points at `PREREGISTRATION.md`. Current research status is replaced by the post-CONFIRM status. The interim selection section is retitled Binance selection results superseded by CONFIRM. A new section, Confirmatory Binance results, follows that section. It states the preregistered HARQDRD01 versus V01 decision, the three cutoffs, the failure of loss robustness and proxy robustness, the descriptive ranking caveat for BEKKS01 and LSTMBEKK-noC, the Giacomini-Rossi and Giacomini-White rejections, the model confidence set at block length 29, the unrejected GMV risk gap, the six Fleming-Kirby-Ostdiek fees, the exploratory X1 and X2 notes, and the limitations. Sentences that still said the holdout had not been opened were updated in the statistical-evaluation, selection-rule, economic-evaluation, limitations, and reproduction sections.

`docs/PROJECT_STATE.md`. Three status bullets now record that the preregistration exists and that CONFIRM has been run once. A checkpoint dated 2026-10-31 records the manifest commit, the evaluation commit, the X1 and X2 commits, and the next recommended task. That task is a second holdout on later Binance data or the equity panel for any new model. No further claim will be taken from this holdout.

## 5. Problems

None observed.

## 6. Verification

`git diff` on the branch was inspected for colons in added lines. None were found. The branch was not merged. `git checkout main` restored `README.md` and `docs/PROJECT_STATE.md` on `main`. No tests were run. No forecast was refit.

## 7. Artifacts

Branch `docs/confirm-writeup`, commit `509be2ad3d04b2c443e8719a1fc5ac894b5928ce`. This review note is `results/post_screen/W1/REPORT.md` on `main`. `PREREGISTRATION.md`, `results/confirm/REPORT.md`, and the exploratory reports were not rewritten.

## 8. Docs

The register was not edited. The README and project-state edits are on the unmerged branch, not on `main`.

## 9. Next task

S0, the separate SignalHarness download, named and not started by this note.
