# C4 report

## 1. Outcome

C4. Implemented and executed. The pre-CONFIRM semivariance array was built from the production five-minute returns, and SHARDRD01 completed VALIDATION and SCREEN with no recorded repairs. It is eligible for the econometric pool at G1. No SPA, MCS, or reselection was run.

## 2. Starting state

Branch `main` after the B1b report commit `5d7591a`. Documents read were amendments R-2026-10-29 and R-2026-10-30, `src/covharness/models/har_drd.py`, `harq_drd.py`, and `src/covharness/realized/binance_panel.py`. The production panel was not modified.

## 3. Spec commit hash

`16144a6497600c1f8e427c8cdf23752bf23abc1a`, recorded before the semivariance build and the fit.

## 4. Changes

`src/covharness/models/shar_drd.py` imports the HAR-DRD within estimator, lag convention, and estimation-window-mean fallback. The variance design uses lagged RS+, lagged RS−, and the unchanged weekly and monthly variance lags. The runner accepts optional semivariance panels only when the model declares that input. The register reference is the task C4 spec and amendment R-2026-10-29.

## 5. Problems

RS+ plus RS− is bitwise equal to realized variance on the exactly representable test case. On the pre-CONFIRM panel the same sum differs from the stored five-minute variance diagonal by at most \(2.7755575615628914\times 10^{-17}\). The cause is float64 summation order. The semivariances were not altered to absorb that residual.

## 6. Verification

`python -m pytest tests/unit/test_models_shar_drd.py -q --tb=line` exited 0 with 3 passed, in 2.62 seconds. The tests cover the dyadic bitwise sum, the equal-slope reduction to HAR-DRD within \(10^{-15}\), and the index check that the response row is not a feature. `python scripts/post_screen_shar.py` exited 0. VALIDATION runtime 4.394 seconds, 250 targets, repair count 0. SCREEN runtime 8.797 seconds, 500 targets, repair count 0. CONFIRM dates were not built. SPA and MCS were not run.

## 7. Artifacts

`results/post_screen/C4/c4_summary.json` sha256 `b34dc67e76265908182170a9497050c20da9a547e1dff26233898b37a8e6cc3e`. Local semivariance NPZ sha256 `7366a2b60673d95bf0b0ca58d507ee2bbb86f1c2488ab06c31773d1fe53299cb`. That NPZ is gitignored. VALIDATION mean QLIKE is −32.81890778474039 and SCREEN mean QLIKE is −31.733461479979443. The production panel and `configs/binance_open_data_core.yaml` were not modified.

## 8. Docs

The register row for C4 is completed. Spec commit `16144a6497600c1f8e427c8cdf23752bf23abc1a`. Result commit `c6de3e5cbc98d161be3f57afdbb01de56d48e6db`. `README.md` records SHARDRD01.

## 9. Next task

D2, the information-matched LSTM-BEKK variants. Not started in the C4 commit.
