# B1b report

## 1. Outcome

B1b. Implemented and executed. The size-adjusted two-sided critical values for P5 are \(c_7=2.4728093597276533\) and \(c_{29}=2.4226501420943998\). Both were computed on the resampled SCREEN headline-gap null and checked on a fresh draw. They are frozen and were not applied to any model comparison.

## 2. Starting state

Branch `main` at amendment commit `9d50128d0a01a80396929381314aac9fc4b01125`. Documents read were amendment R-2026-10-30, the B1 report, `scripts/dm_procedure_calibration.py`, and the equal-weighted cosine branch of `diebold_mariano`. The working tree was clean apart from untracked `results/post_screen/B1/cells/`.

## 3. Spec commit hash

`e4f01c0301320d5f92e0041e09069abc85ab2ba4`, recorded before the simulation.

## 4. Changes

`scripts/p5_size_adjusted_cutoffs.py` calls the existing D3 path builder and `diebold_mariano(..., reference="ewc")`. The null is 20000 stationary-bootstrap draws of the demeaned SCREEN HARQ-DRD minus LSTM-BEKK QLIKE gap at block lengths 7 and 29. The cutoff is NumPy's linear 0.95 quantile of the absolute statistic. The ordinary cutoff is the existing Student-\(t\) value at \(\nu=25\), 2.0595385527532972. The register rule from R-2026-10-30 is unchanged. A confirmatory claim needs the ordinary cutoff and both size-adjusted values.

## 5. Problems

None observed.

## 6. Verification

`python scripts/p5_size_adjusted_cutoffs.py` exited 0 in 30.120 seconds. Fresh 5000-replication rejection rates are 0.0516 at block 7 and 0.0472 at block 29. Two Monte Carlo standard errors at the nominal 0.05 rate are 0.006164. Both rates lie inside that band. The cutoffs were not applied to a comparison. SPA, MCS, and CONFIRM were not run.

## 7. Artifacts

`results/post_screen/B1b/b1b_cutoffs.json` sha256 `26b68025d4186654759f2e2fc85d3f26eedcbda9868848fdbc7865c38a510715`. Calibration seed 20260916. Null seeds 1018646849 and 925551463. Check seeds 192526076 and 1684267205. Quantile-standard-error seeds 310480529 and 698742525. Bootstrap standard errors are 0.024786494686054643 and 0.02011665228045664. The 2026-09-21 loss file was read and not modified. `configs/binance_open_data_core.yaml` was not modified.

## 8. Docs

The register row for B1b is completed. Spec commit `e4f01c0301320d5f92e0041e09069abc85ab2ba4`. Result commit `7145b67ab24c0ed36956ad82535cba188f242765`. `docs/PROJECT_STATE.md` and `README.md` record the frozen values.

## 9. Next task

C4, the semivariance panel and SHAR-DRD. Not started in the B1b commit.
