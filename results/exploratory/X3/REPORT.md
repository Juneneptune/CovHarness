# X3 report

Exploratory, computed after CONFIRM. Nothing in this report is a confirmatory claim.

## 1. Outcome

Task X3. Executed. Saved CONFIRM forecast cubes were rescored under reduced QLIKE. No model was refit. On the 500 targets with a finite five-minute proxy, HARQDRD01 minus V01 has mean gap 0.817439 and P5 statistic 2.651948. Without 2025-10-10 the mean gap is 0.699169 and the P5 statistic is 2.404565. That day contributes gap 59.834539, which is 0.146395 of the summed gap. The preregistered rule was not changed.

## 2. Starting state

Branch `main` at commit `6e419d1bccc999e64d2e5b133b4fd085eb4dcadf`. The working tree had untracked zip archives and an untracked `results/stage2` directory from another branch. Those files were not added. The files read were the CONFIRM forecast cubes, `results/confirm/inputs/confirm_measurements.npz`, the production panel realized-covariance array on the CONFIRM target dates, `results/confirm/headline.json` for the stored 15-minute comparison, and `results/confirm/DEVIATIONS.md`.

## 3. Spec commit hash

`9aebdef7f80e2bb24d6b7e5b302cf1ffe6943681`, recorded before the scoring command.

## 4. Changes

`scripts/exploratory/x3_crash_day.py` loads the 21 saved cubes. Stochastic identifiers use the five-seed equal-weight ensemble. Reduced QLIKE is the existing loss. P5 uses nu 25. A target stays in a view only when every model has a finite loss on that proxy. Leave-one-out drops one target from the headline gap and recomputes the statistic. Influence is the full-sample statistic minus the statistic without that date. `tests/unit/test_x3_leave_one_out.py` plants one outlier in a synthetic series. The register entry is Task X3 in `PREREGISTRATION_DRAFT.md`. `results/confirm/DEVIATIONS.md` gained an addendum and no existing line was edited.

## 5. Problems

The task text stated 499 five-minute targets and a P5 statistic of 2.647, and a without-crash statistic of 2.400. The saved series has 500 finite five-minute targets. The full-sample statistic is 2.651948, which matches the stored headline. The without-crash statistic is 2.404565. EWMA21 without that day is -34.892543, against the stated -34.895. V01 is -34.436784 and HARQDRD01 is -33.737615, which agree with -34.437 and -33.738 at the third decimal. No day was dropped to force the task-text count.

## 6. Verification

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_x3_leave_one_out.py`

Exit code 0. 3 passed in 2.72s. The planted outlier is the unique minimum and the unique largest absolute influence. Dropping an interior index matches the direct statistic on the remaining ordered series. A series of length 2 is rejected.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python scripts/exploratory/x3_crash_day.py`

Exit code 0. HARQDRD01 five-minute QLIKE matches the stored cube to 1e-10. The maximum absolute gap is 0. The 15-minute headline gap is 0.704510 and its P5 statistic is 2.388403. The crash-day gap is 59.834539. Every one of the 500 targets is finite under both proxies, so no target was removed for a missing proxy. Repair counts were not recomputed. The saved repair flags were not used as a sample filter. No lookahead check applies, because the cubes and the proxies are the stored CONFIRM objects. CONFIRM was not refit.

`git diff -- results/confirm/DEVIATIONS.md` shows additions only.

```
diff --git a/results/confirm/DEVIATIONS.md b/results/confirm/DEVIATIONS.md
index cd79b31..2c37eae 100644
--- a/results/confirm/DEVIATIONS.md
+++ b/results/confirm/DEVIATIONS.md
@@ -5,3 +5,7 @@ V01 uses the return-only input. The shared fit signature still requires a featur
 The five LSTMBEKK-noC seed cubes are identical. The ablation column is that common cube. No seed was selected.
 
 BEKKS01 carry counts were not written to the job summary. The saved repair count is 0 and the saved policy-event list is empty. The model was not refit to recover a counter.
+
+## Addendum after the run
+
+Commit 4c2a879 corrected a printed Mincer-Zarnowitz p-value in the report text. The stored statistics and the headline were unaffected.
```

## 7. Artifacts

`results/exploratory/X3/SPEC.md`, sha256 `ab14d4d76f0777cf9d915c61b1ac4f420e1ac5d93af110d1237d7fb364d5602b`.

`results/exploratory/X3/ranking_5min.csv`, sha256 `5920737bad66955164f2a1a551f797846c777adb6e29765b60808fe3aa799c25`.

`results/exploratory/X3/ranking_15min.csv`, sha256 `7515204898122d77998dcadba5f61134effd2437242271a87cb509168301bdb6`.

`results/exploratory/X3/pairs.csv`, sha256 `f38b1c99bc36056917bba5be7b05a7794610ad8587c25c45bce5b726760179df`.

`results/exploratory/X3/loo_influence.csv`, sha256 `eb65c861d5d67c054170d69019d9be6d121bb4b9420e12882d9f6d6d7f2a7179`.

Nu is 25. The cutoffs are 2.0595385527532972, 2.4226501420943998, and 2.4728093597276533. The label on every row is exploratory, computed after CONFIRM. `PREREGISTRATION.md`, `results/confirm/headline.json`, and the forecast cubes were not modified.

Five-minute mean rank 1 is BEKKS01. Without 2025-10-10 the mean rank 1 is EWMA21, then EWMA01, then RW01. Median rank 1 is EWMA21, then EWMA01, then RW01. HARDRD01, HARQDRD01, and SHARDRD01 have rank 15 or worse in the full mean, the mean without that day, and the median, under both proxies. HARDRD01 and RIDGE01 tie.

Leave-one-out minimum statistic is 2.404565 on 2025-10-10. The five largest absolute influences are 2025-10-10, 2026-08-19, 2026-08-22, 2026-08-21, and 2025-07-18.

V01 minus LSTMBEKK-noC has full mean 0.120792, mean without the crash day -0.119620, and median -0.074649. LSTM22 minus LSTMBEKK-noC has 0.614785, -0.106714, and -0.173218. V01 minus LSTM22 has -0.493994, -0.012906, and -0.013548. BEKKS01 minus V01 has -0.164160, 0.181659, and 0.158714.

The without-crash P5 statistic 2.404565 is above 2.059539 and below 2.422650 and 2.472809.

## 8. Docs

`PREREGISTRATION_DRAFT.md` gained Task X3 with status executed. `docs/PROJECT_STATE.md` gained one line pointing at this report. `README.md` was not changed. The result commit is the commit that adds this report.

## 9. Next task

W1b, the revision of the CONFIRM write-up on `docs/confirm-writeup`. Not started by this report.
