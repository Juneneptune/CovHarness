# D2 report

## 1. Outcome

D2. Implemented and executed. Fifteen information-matched fits completed, five seeds for each of V10, V01, and V11. Every ensemble uses all five seeds. V00 was not refit. The SCREEN gaps below are descriptive. No reselection, SPA, or MCS was run.

## 2. Starting state

Branch `main` after the C4 report. Documents read were amendments R-2026-10-29 and R-2026-10-30, `src/covharness/models/lstm_bekk.py`, the D1 summary, and the C4 semivariance array. The production panel was not modified.

## 3. Spec commit hash

`60b240bae32577e0159df8cc494019b3db4935d4`, recorded before any D2 fit.

## 4. Changes

`src/covharness/models/lstm_bekk_variants.py` adds V10, V01, and V11 at the LSTM22 settings, including the named opt-in `architecture_range=post_screen_d1`. V10 appends standardized origin-day semivariances, realized correlations, and log quarticity. V01 trains on reduced QLIKE. V11 does both. The faithful class forecast on the fixed test input stayed bitwise identical. V01 and V11 kept the LSTM22 learning rate and epoch count. The register reference is the task D2 spec and amendment R-2026-10-30.

## 5. Problems

None observed. The SCREEN seed distribution for V00 is not in the retained D1 summary, so that seed mean and seed SD are blank in the SCREEN table. The V00 ensemble scores are the stored D1 scores.

## 6. Verification

`python -m pytest tests/unit/test_models_lstm_bekk_variants.py tests/unit/test_models_lstm_bekk.py -q --tb=line` exited 0 with 65 passed. The new checks cover the bitwise faithful forecast, equality of the torch reduced QLIKE with `covharness.losses` on a fixed pair, a scaler fit on the supplied window only, a feature row that ignores the next day, an origin window that excludes the target, and rejection of a zero semivariance. `nproc` reported 48. The launch cap was 44, and all 15 jobs ran at once. Each job set one thread. No job failed, so none was restarted. Wall clock from launch to the last summary was 32 minutes 3 seconds. SPA and MCS were not run.

## 7. Artifacts

`results/post_screen/D2/d2_summary.json` sha256 `610d8a739233fab89b6277dbaf58dd45337940fbee6dd361d40756bf6d66f746`. Seed summaries are under `results/post_screen/D2/<variant>/seed_<s>/summary.json`. Checkpoints, logs, and the semivariance NPZ were not committed. The production panel and `configs/binance_open_data_core.yaml` were not modified.

VALIDATION ensemble QLIKE and Frobenius. V00 is the stored LSTM22 score.

| Variant | Seed mean QLIKE | Seed SD | Ensemble QLIKE | Ensemble Frobenius |
| --- | --- | --- | --- | --- |
| V00 | −32.200674 | 0.110091 | −32.322996 | 4.628149e-05 |
| V10 | −31.996422 | 0.053303 | −32.089654 | 4.623369e-05 |
| V01 | −32.354198 | 0.086761 | −32.391761 | 4.724258e-05 |
| V11 | −32.400706 | 0.083073 | −32.465960 | 4.700102e-05 |

SCREEN. V00 seed mean and seed SD were not retained.

| Variant | Seed mean QLIKE | Seed SD | Ensemble QLIKE | Ensemble Frobenius |
| --- | --- | --- | --- | --- |
| V00 |  |  | −31.297632 | 8.369783e-05 |
| V10 | −31.196361 | 0.038788 | −31.303579 | 8.405916e-05 |
| V01 | −31.465473 | 0.050343 | −31.515557 | 8.008975e-05 |
| V11 | −31.410991 | 0.043838 | −31.468871 | 8.037765e-05 |

Descriptive SCREEN gaps. V11 minus V00 ensemble QLIKE is −0.171240. HARQ-DRD minus V11 is −0.393191, using the stored HARQ-DRD mean −31.862062800706465.

## 8. Docs

The register row for D2 is completed. Spec commit `60b240bae32577e0159df8cc494019b3db4935d4`. Result commit `453174f0eee09275533c149da89e010ff9d8d98d`. `README.md` records the diagnostics.

## 9. Next task

C2, the BEKK family and the zero-intercept ablation. It was specified separately and is not part of this result.
