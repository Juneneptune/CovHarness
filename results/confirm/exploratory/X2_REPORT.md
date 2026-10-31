# X2 ablation gap concentration

Exploratory, after CONFIRM; not a confirmatory claim.

## 1. Outcome

Task X2. Executed. The question is whether the ablation's positive average gap against V01 and LSTM22 is a few days. On five-minute reduced QLIKE, V01 has the lower loss on 0.572 of the 500 CONFIRM targets. The median daily gap, V01 minus LSTMBEKK-noC, is -0.074649, and the mean is 0.120792. After the 5, 10, and 20 days with the largest absolute gap are removed, the mean is -0.115212, -0.133118, and -0.162366. LSTM22 has the lower loss on 0.700 of days. Its median gap against LSTMBEKK-noC is -0.173218, and its mean is 0.614785. After the same removals the mean is -0.102709, -0.118842, and -0.160409. The positive means are not the typical day. The largest absolute day for both pairs is 2025-10-10.

## 2. Starting state

Branch `main` at `fa2b1ac`. The working tree was clean. Documents read were `results/confirm/exploratory/X1_REPORT.md`, `results/confirm/exploratory/x1_numbers.json`, and the saved CONFIRM forecast cubes. No confirmatory file outside `results/confirm/exploratory/` was edited.

## 3. Spec commit hash, recorded before any empirical run

`fa2b1ac`. No model was fit. The evaluation uses saved forecast cubes.

## 4. Changes

`results/confirm/exploratory/x2_compute.py` evaluates the saved equal-weight covariance ensembles. `results/confirm/exploratory/x2_numbers.json` stores the shares, the trimmed means, and the twenty largest absolute days. Seed files store each seed's own loss. The ensemble loss is the loss of the saved mean covariance, not the mean of the seed losses. The gap is the first model's loss minus the second. A negative gap means the first model has the lower loss. X1 bucket labels use the SCREEN tercile cutoffs already stored in `x1_numbers.json`.

V01 minus LSTMBEKK-noC, twenty largest absolute days. Date, gap, variance bucket, correlation bucket, jump bucket.

2025-10-10, 120.086133, middle, middle, no.
2026-08-21, -4.916139, middle, middle, no.
2026-08-22, -4.047655, high, middle, no.
2025-10-25, 3.371938, low, middle, no.
2025-10-26, 2.931436, low, middle, no.
2025-07-18, 2.875262, high, middle, no.
2025-11-02, 2.702820, low, low, no.
2025-10-24, 2.699068, middle, high, no.
2026-02-05, -2.570745, high, high, no.
2025-11-01, 2.491253, middle, high, no.
2025-11-21, -2.479287, high, high, no.
2025-10-19, 2.472506, low, high, no.
2025-10-20, 2.336610, middle, high, no.
2025-10-29, 2.325737, middle, middle, no.
2025-10-27, 2.254577, low, high, yes.
2025-10-22, 2.141347, high, high, no.
2025-07-17, 1.955332, middle, middle, no.
2026-08-24, -1.937818, middle, high, yes.
2025-10-31, 1.893674, high, high, no.
2025-07-11, 1.745620, middle, middle, yes.

LSTM22 minus LSTMBEKK-noC, twenty largest absolute days.

2025-10-10, 360.642735, middle, middle, no.
2026-08-22, -4.941820, high, middle, no.
2026-05-30, -3.519457, low, high, no.
2026-02-06, 3.045631, high, high, no.
2025-08-14, 3.006350, middle, low, no.
2026-06-20, 2.679800, low, high, no.
2026-08-21, -2.633498, middle, middle, no.
2026-06-13, 2.499960, low, high, no.
2026-06-19, 2.470159, middle, high, no.
2026-02-05, 2.375343, high, high, no.
2026-06-21, 2.253036, low, high, yes.
2026-06-18, 2.217827, middle, high, yes.
2025-07-18, 2.087001, high, middle, no.
2026-06-22, 2.057095, low, middle, yes.
2026-06-12, 2.031370, middle, high, yes.
2026-06-17, 1.672491, middle, middle, yes.
2026-06-11, 1.670883, middle, high, no.
2026-06-16, 1.631569, middle, middle, no.
2026-06-14, 1.599985, low, middle, no.
2025-07-17, 1.542495, middle, middle, no.

## 5. Problems

None observed.

## 6. Verification

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python results/confirm/exploratory/x2_compute.py` exited 0. Both pairs have 500 finite gaps and a tie share of 0. No model was refit. No new p-value was computed. The SCREEN tercile counts were not recomputed. The stored X1 cutoffs were applied.

## 7. Artifacts

`results/confirm/exploratory/X2_REPORT.md`. `results/confirm/exploratory/x2_numbers.json` sha256 `036721fcbdb241b2ee870dc592b17e97436d94602b42576cb717f91b0ca635d5`. `results/confirm/exploratory/x2_compute.py` sha256 `4e5153a59d48607fb42e5540401685f0a86b63ec09d773aea3f5e31c67495281`. Saved forecast cubes, `results/confirm/REPORT.md`, and `results/confirm/exploratory/x1_numbers.json` were not rewritten.

## 8. Docs

No register entry. `docs/PROJECT_STATE.md` and `README.md` were not changed.

## 9. Next task

T3, named and started only after this report is committed.

Exploratory, after CONFIRM; not a confirmatory claim.
