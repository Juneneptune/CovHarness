# T2 report

## 1. Outcome

T2. Executed. The three post-SCREEN batch summaries were moved with `git mv` into `results/post_screen/run_log/` and renamed by the date of the commit that first added each file. The first heading of each file now reads as a run summary. No protocol, score, or commit hash inside those summaries was changed. This task is documentation only and is exempt from the register-spec rule.

## 2. Starting state

Branch `main` after the E1 report commit `9a38c38fed5a1a959efc25aaf0d026262656207c`. Untracked `results/post_screen/B1/cells/` and the D2 seed pid files were left untracked. The files read were `results/post_screen/OVERNIGHT_SUMMARY.md`, `OVERNIGHT_SUMMARY_2.md`, and `OVERNIGHT_SUMMARY_3.md`, plus a repository search for those paths.

## 3. Spec commit hash

None. The task changes no protocol and is exempt from the register-spec rule. The tidy commit is `5d3657d6e1a1aeb99464cb78e83e8e3db1a0b2be`.

## 4. Changes

`OVERNIGHT_SUMMARY.md` moved to `results/post_screen/run_log/2026-10-28_queue1.md`. `OVERNIGHT_SUMMARY_2.md` moved to `results/post_screen/run_log/2026-10-30_queue2.md`. `OVERNIGHT_SUMMARY_3.md` moved to `results/post_screen/run_log/2026-10-30_queue3.md`. The dates are the `%cs` dates of the commits that added the files. Each first heading replaces Overnight with Run and names the queue. `results/post_screen/run_log/README.md` states what the files are, that each line links a task to its spec and result commits, and that the amendment register is the primary record. Path references in `PREREGISTRATION_DRAFT.md`, `results/post_screen/R1/REPORT.md`, and `results/post_screen/R2/REPORT.md` were updated and were not otherwise reworded. One sentence under the planned-entry table points to the run log.

## 5. Problems

none observed

## 6. Verification

`git log --diff-filter=A --format=%cs` returned 2026-10-28, 2026-10-30, and 2026-10-30 for the three original files. After the commit, `git log --follow --oneline` on each new path shows the original adding commit. Queue 1 follows `ee83e68`. Queue 2 follows `d8281d0`. Queue 3 follows `987a6f6`. `git grep OVERNIGHT_SUMMARY` returned no matches. `git show --stat -M 5d3657d6e1a1aeb99464cb78e83e8e3db1a0b2be` lists the three renames, the new README, and the path edits in the register and the two reports. No tests were run. No model was fit. CONFIRM was not read.

## 7. Artifacts

The moved summaries, `results/post_screen/run_log/README.md`, and the path edits are in commit `5d3657d6e1a1aeb99464cb78e83e8e3db1a0b2be`. Earlier result files were not deleted. Their history is the rename. No forecast, loss, or panel file was modified.

## 8. Docs

The amendment register gains one sentence under the planned-entry table and no rewritten earlier entry, apart from the path string in the R-2026-10-29 paragraph. `docs/PROJECT_STATE.md` and `README.md` contained no reference to the old summary paths and were not edited.

## 9. Next task

G1, the CONFIRM authorization gate. Not started.
