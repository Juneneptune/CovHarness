# Binance open-data first-stage protocol

Protocol date 2026-09-21. This document freezes the first-stage Binance open-data core benchmark schedule, roster, and configuration grids. VALIDATION fitting is recorded in Section 9. SCREEN execution and the frozen Stage-2 finalist rule are recorded in Sections 10 and 11. CONFIRM was not run. Generic `TemporalProtocol` is unchanged. CONFIRM remains locked. Final `PREREGISTRATION.md` remains absent.

Blocks 1–3 and 4A-1 through 4A-10 remain closed. LSTM-BEKK-RC, GHAR, and the graph-neural slot remain outside this first-stage core.

---

## 1. First-stage roster

The core roster is frozen as RW, EWMA, HAR-DRD, HARQ-DRD, LW-linear, LW-NL, DCC, DCC-NL, Ridge-DRD, XGBoost-DRD, and LSTM-BEKK.

No model is added or removed after VALIDATION performance is seen. This roster is not the complete research-paper finish line.

---

## 2. Segmented rolling schedule

Implementation `covharness.protocol.binance.BinanceSegmentedProtocol`. Public constructor `build_binance_segmented_protocol()`.

VALIDATION uses the development calendar HISTORY_A plus VALIDATION only. SCREEN and CONFIRM use the confirmatory calendar HISTORY_B plus SCREEN plus CONFIRM only. Those calendars are never concatenated. No origin, target, estimation window, recursive state, or feature may include 2023-03-24.

Packed lags were rejected because 2023-03-23 and 2023-03-25 are separated by 48 hours. HAR lags, EWMA state, DCC state, and LSTM-BEKK recurrent state must not treat that gap as one statistical step.

First-origin refit. The first target in each evaluation block has `refit=True`. Later refits occur every 21 forecast origins inside that block. The counter resets at the start of VALIDATION, SCREEN, and CONFIRM.

| Block | Targets | First origin | First window | Refits |
| --- | --- | --- | --- | --- |
| VALIDATION | 250, 2022-07-17 through 2023-03-23 | 2022-07-16 | HISTORY_A, 2021-11-09 through 2022-07-16 | 12 |
| SCREEN | 500, 2023-11-30 through 2025-04-12 | 2023-11-29 | HISTORY_B, 2023-03-25 through 2023-11-29 | 24 |
| CONFIRM | 500, 2025-04-13 through 2026-08-25 | 2025-04-12 | the last 250 SCREEN dates | 24 |

Every origin precedes its target by exactly one UTC calendar day. Every window has $m=250$ admissible statistical dates and excludes the target. CONFIRM's first refit uses SCREEN-period dates only. No fitted SCREEN object needs to be carried forward.

Schedule construction reads dates only. It does not load the production NPZ.

---

## 3. CONFIRM lock

Public CONFIRM target and schedule requests raise `ConfirmLockedError` unless `unlock_confirm=True` is passed. The object is constructed locked. Frozen CONFIRM start and end dates may appear in configuration as calendar metadata. They are not forecast-result access.

No CONFIRM execution occurred in this freeze.

---

## 4. Validation-selection rule

VALIDATION is the only block used to choose hyperparameters. The primary score is mean reduced multivariate QLIKE over all 250 VALIDATION targets. Squared Frobenius is a required descriptive secondary channel. It does not select the configuration, break ties, or change the search space.

SCREEN, CONFIRM, crisis subsets, cumulative-loss plots, and individual seeds do not select a configuration.

Tie rule. If two admissible finite primary scores are exactly equal, choose the lexicographically smaller frozen configuration ID. There is no epsilon tolerance.

---

## 5. Seed and ensemble rule

Stochastic methods use seeds $0,1,2,3,4$. All five are reported. Best-seed selection is forbidden.

For LSTM-BEKK, each of the 20 architecture/optimizer candidates is run under all five seeds. The candidate's primary forecast at each target is the arithmetic mean of the five seed covariance matrices. The candidate score is mean reduced QLIKE of that equal-weight ensemble. Individual seed forecasts and losses are retained for later reporting. Seed is not part of the 20-configuration budget.

---

## 6. Candidate failure rule

A candidate is valid only if it supplies the complete required VALIDATION target support.

Deterministic methods must produce all 250 forecasts, each evaluable under reduced QLIKE.

LSTM-BEKK must produce that support for all five seeds. A failed seed is not discarded.

An invalid candidate is preserved with diagnostics and is not scored by silently shortening support. No unregistered substitute is introduced. If every candidate of a tunable family fails, the family is reported as failed rather than removed.

Already-frozen model-specific forecast repair rules are unchanged. Evaluation itself still performs no repair.

---

## 7. Fixed versus searched families

One frozen configuration, the already implemented mathematical definition, for RW, HAR-DRD, HARQ-DRD, LW-linear, LW-NL, DCC, and DCC-NL. These are not described as untuned defaults.

EWMA has 20 decay values EWMA01–EWMA20, $0.8000$ through $0.9975$.

Ridge-DRD has 20 shared lambdas RIDGE01–RIDGE20, $0$ through $100000$. RIDGE01 is the zero-penalty nested HAR solution. Predictor scaling, response scaling, and intercept treatment are unchanged.

XGBoost-DRD has the Cartesian product of four capacity settings and five regularization settings, XGB01 $=$ C1/R1 through XGB20 $=$ C4/R5. Package constants remain CPU hist, `reg:squarederror`, no early stopping, and `random_state=0`.

LSTM-BEKK has the Cartesian product of five architecture/training settings and four optimizer settings, LSTM01 $=$ A1/O1 through LSTM20 $=$ A5/O4. A1 uses dropout $0.0$, the no-dropout stacked-LSTM setting. Hidden size remains $N$. Training remains float64 CPU, full BPTT, fixed epochs, and the existing RMSprop constants. There is no inner validation split, early stopping, or learning-rate scheduler.

The constructor bound on LSTM dropout was expanded from $[0.1,0.2]$ to $[0,0.2]$ so that A1 is constructible. The BEKK recursion is unchanged.

---

## 8. Configuration artifact

Path `configs/binance_open_data_core.yaml`.

SHA-256 `8d63eacacf13a876651f9c4eb5399a8527d8458278974ff069dd0701dd5cbf07`.

The file records branch identity, production-panel SHA-256, assets, segments, $m=250$, cadence 21, first-origin refit, block-local reset, roster, losses, seeds, ensemble rule, failure rule, tie rule, CONFIRM lock, and every candidate. Results must not be written back into this file.

Selection helper `select_validation_configuration` consumes already supplied configuration IDs, complete-support flags, and primary scores. It does not fit a model or read market data.

---

## 9. VALIDATION execution status

VALIDATION fitting completed on 2026-09-21 as a development-set exercise. SCREEN execution is recorded in Section 11. CONFIRM remained locked.

Artifacts.

`results/binance_validation_manifest.json`

`results/binance_validation_selection.json`

`results/binance_validation_candidate_summary.csv`

`results/binance_validation_development_table.csv`

`results/binance_validation_diagnostics.json`

`results/binance_validation_failures.json`

`results/binance_validation_forecasts.npz`

The selection file records the four searched configuration IDs and the seven fixed family IDs. Those IDs have not been copied back into `configs/binance_open_data_core.yaml`. They were copied into `configs/binance_open_data_screen.yaml` after VALIDATION completed. The development table is not a confirmatory ranking.

Maximum model-observable date 2023-03-22. Maximum scoring target 2023-03-23.

The U.S.-equity DATA GATE remains closed. LSTM-BEKK-RC, GHAR, and the graph-neural slot remain unimplemented.

---

## 10. Frozen Stage-2 finalist rule

This rule was recorded in `configs/binance_open_data_screen.yaml` before SCREEN losses were used.

The headline research comparison remains one deep-learning finalist versus one econometric finalist.

Paradigm membership is frozen before ranks. Econometric eligible set. RW, EWMA, HAR-DRD, HARQ-DRD, LW-linear, LW-NL, DCC, DCC-NL. Shallow-ML controls Ridge-DRD and XGBoost-DRD participate in SPA, MCS, loss tables, and robustness reporting. They cannot occupy the econometric headline slot. Deep-learning eligible set. LSTM-BEKK. Because this first-stage roster contains only one deep model, LSTM19 is the DL finalist by construction.

Primary reduced QLIKE is the only headline finalist channel. Partition the 500 SCREEN targets into four chronological 125-target blocks at positions 0-124, 125-249, 250-374, and 375-499. Rank complete econometric models within each block. Lower mean QLIKE receives a better rank. Exact block-mean ties receive average ranks. Select the lowest median of the four block ranks, then the lowest mean of those ranks, then the lexicographically smaller family ID. Full-SCREEN mean QLIKE is a robustness leader only. Squared Frobenius does not select. MCS membership is reported for each finalist and does not replace a finalist.

Hansen SPA uses HAR-DRD as benchmark, $B=5000$, seed 20260913, and consistent recentering. An identically zero benchmark differential is labeled `exact_benchmark_tie`, retained in tables and MCS, and excluded only from SPA studentization.

---

## 11. SCREEN execution status

SCREEN fitting and selection completed on 2026-09-21. CONFIRM remained locked. These SCREEN results are selection data, not confirmation.

SCREEN configuration `configs/binance_open_data_screen.yaml`. SHA-256 `b07011a798c36fc25e117b69d71f6c8eb5eb1ed756e97c39c19e29542af6203c`. Frozen representatives EWMA01, RIDGE01, XGB08, LSTM19 with seeds 0-4, and the seven fixed family IDs. Results were not written back into this file.

Complete 500-target support. RW, EWMA, HAR-DRD, HARQ-DRD, LW-linear, LW-NL, Ridge-DRD, XGBoost-DRD, LSTM-BEKK. DCC and DCC-NL failed at origin 2023-12-20 under the existing IGARCH contract and were not replaced.

Econometric finalist HARQ-DRD (`HARQDRD01`). DL finalist LSTM-BEKK (`LSTM19` equal-weight ensemble). Full-SCREEN mean-QLIKE robustness leader among complete econometric models is also HARQ-DRD. RIDGE01 is an exact HAR-DRD SPA benchmark tie. HARQ-DRD recorded 2 SCREEN repairs. LSTM-BEKK is outside the primary QLIKE MCS at $\alpha=0.10$ and remains the DL finalist by construction.

Artifacts.

`results/binance_screen_manifest.json`

`results/binance_screen_model_summary.csv`

`results/binance_screen_spa.json`

`results/binance_screen_mcs_qlike.json`

`results/binance_screen_mcs_frobenius.json`

`results/binance_screen_subblock_ranks.csv`

`results/binance_screen_finalists.json`

`results/binance_screen_failures.json`

`results/binance_screen_losses_qlike.npz`

`results/binance_screen_losses_frobenius.npz`

`results/binance_screen_forecasts.npz`

`results/binance_screen_diagnostics.json`

Hashes are recorded in `docs/PROJECT_STATE.md`. CONFIRM remains locked. No confirmatory pairwise test was run.
