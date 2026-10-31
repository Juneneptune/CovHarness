# E1 report

## 1. Outcome

E1. Implemented and executed. Saved SCREEN forecasts were mapped to daily global-minimum-variance weights. No model was refit. The primary risk figures are the annualized sample standard deviation of the simple portfolio return and the mean of its square. On the unconstrained raw book, the frozen P5 statistics for HARQDRD01 minus LSTM22, V11, EWMA21, and HARDRD01 are −1.6837853064044193, −1.3583087950363657, −1.1144755058725615, and 0.19103258229680717. None exceeds the ordinary critical value 2.0595385527532972 or either size-adjusted cutoff. The corresponding win rates are 0.546, 0.51, 0.532, and 0.486. These tests are descriptive. Global minimum variance is a risk target, not a return strategy.

## 2. Starting state

Branch `main` after the C3 report commit `b8df4da`. Untracked `results/post_screen/B1/cells/` and the D2 seed pid files were left untracked. Documents and code read were the post-SCREEN register, including P5 and the B1b cutoffs, the README planned economic evaluation, the empty portfolio package, the C3 forecast cube, and the production panel's daily log returns. CONFIRM dates were not scored.

## 3. Spec commit hash

`56bb3bf2fc84764b5f39687cbe647c5827531710`, recorded before the SCREEN run. The following commit `0903126` stores that hash in the entry script.

## 4. Changes

`src/covharness/portfolio/gmv.py` holds the Cholesky weight, the SciPy SLSQP long-only quadratic program, drifted turnover, and the basis-point cost. The book opens in the first origin's weights, so the first turnover is zero. Costs are subtracted from returns. The register entry is the task E1 spec. A positive scale cancels in the unconstrained weight, so R1 reproduces the raw unconstrained book. The largest raw-versus-R1 gap in annualized standard deviation, including the long-only solver, is \(6.630208460034481\times 10^{-8}\).

## 5. Problems

none observed

## 6. Verification

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest tests/unit/test_portfolio_gmv.py -q` exited 0 with 4 passed, in 1.28 seconds. The tests cover a unit sum, equal weights under the identity covariance, zero turnover for constant weights and zero returns, equality of long-only and unconstrained weights when the unconstrained weights are positive, and a net return that falls by exactly the basis-point rate times turnover. `/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python scripts/post_screen_gmv.py` exited 0. Runtime 351.9109487289097 seconds. Solver failures are 0. Every book uses 500 targets. The forecast proxy matches the panel five-minute realized covariance with maximum absolute gap 0. No target is on or after 2025-04-13. The same four P5 pairs on long-only raw weights and on unconstrained R1 and R2 weights also stay inside the ordinary cutoff. SPA, MCS, and finalist selection were not run.

## 7. Artifacts

`results/post_screen/E1/e1_summary.json` sha256 `f12d5b4d1389638717b63611d8fd0a7f61ba75e5fe4c508d9da65022c4bcccb8`. `results/post_screen/E1/e1_cumulative_squared_return_gap.png` sha256 `ea3e81d5eab7c37aaddd36d728abfb3a8edf0c6925d5ecc2fe07d3568674cae5`. Spec commit `56bb3bf2fc84764b5f39687cbe647c5827531710`. Annualization uses 365 and the sample standard deviation uses one degree of freedom. Sharpe ratios use a zero risk-free rate and are stored in the summary. They are not a performance claim. DCC01 and DCCNL01 have no forecasts. The five other BEKK configurations are in the summary and are not in the tables below. The C3 cube, the production panel, and earlier post-SCREEN results were not modified.

### Raw forecasts

| Model | Book | Gross SD | Net 5 bps | Net 10 bps | Mean turnover | Proxy variance |
|---|---|---:|---:|---:|---:|---:|
| RW01 | unconstrained | 0.471297 | 0.471345 | 0.471439 | 1.014416 | 6.551737e-04 |
| RW01 | long_only | 0.489184 | 0.488961 | 0.488771 | 0.595083 | 7.216466e-04 |
| EWMA01 | unconstrained | 0.484275 | 0.484159 | 0.484049 | 0.205736 | 6.498420e-04 |
| EWMA01 | long_only | 0.488752 | 0.488656 | 0.488565 | 0.128207 | 7.309518e-04 |
| HARDRD01 | unconstrained | 0.460630 | 0.460700 | 0.460776 | 0.318982 | 6.376926e-04 |
| HARDRD01 | long_only | 0.485700 | 0.485679 | 0.485662 | 0.188324 | 7.207373e-04 |
| HARQDRD01 | unconstrained | 0.460940 | 0.460898 | 0.460868 | 0.451172 | 6.385085e-04 |
| HARQDRD01 | long_only | 0.485581 | 0.485375 | 0.485180 | 0.295552 | 7.186501e-04 |
| LWLIN01 | unconstrained | 0.480395 | 0.480395 | 0.480394 | 0.024076 | 7.458503e-04 |
| LWLIN01 | long_only | 0.494506 | 0.494503 | 0.494500 | 0.014731 | 7.946704e-04 |
| LWNL01 | unconstrained | 0.476692 | 0.476692 | 0.476693 | 0.027366 | 7.323425e-04 |
| LWNL01 | long_only | 0.493789 | 0.493789 | 0.493789 | 0.014878 | 7.875620e-04 |
| RIDGE01 | unconstrained | 0.460630 | 0.460700 | 0.460776 | 0.318982 | 6.376926e-04 |
| RIDGE01 | long_only | 0.485700 | 0.485679 | 0.485662 | 0.188324 | 7.207373e-04 |
| XGB08 | unconstrained | 0.479234 | 0.479040 | 0.478887 | 0.787226 | 6.714505e-04 |
| XGB08 | long_only | 0.487099 | 0.486907 | 0.486744 | 0.521507 | 7.334634e-04 |
| LSTM19 | unconstrained | 0.474900 | 0.474912 | 0.474932 | 0.189338 | 6.972345e-04 |
| LSTM19 | long_only | 0.489827 | 0.489818 | 0.489813 | 0.117239 | 7.525347e-04 |
| EWMA21 | unconstrained | 0.485344 | 0.485276 | 0.485221 | 0.378054 | 6.462487e-04 |
| EWMA21 | long_only | 0.486855 | 0.486733 | 0.486620 | 0.231146 | 7.225490e-04 |
| LSTM22 | unconstrained | 0.473778 | 0.473766 | 0.473761 | 0.193129 | 6.987814e-04 |
| LSTM22 | long_only | 0.490956 | 0.490930 | 0.490907 | 0.121757 | 7.573789e-04 |
| DCC02 | unconstrained | 0.511667 | 0.511433 | 0.511248 | 0.520663 | 7.345347e-04 |
| DCC02 | long_only | 0.501114 | 0.500835 | 0.500587 | 0.358913 | 7.852222e-04 |
| DCCNL02 | unconstrained | 0.511190 | 0.510961 | 0.510778 | 0.511721 | 7.332288e-04 |
| DCCNL02 | long_only | 0.501120 | 0.500835 | 0.500580 | 0.355369 | 7.847430e-04 |
| DCC03 | unconstrained | 0.510676 | 0.510496 | 0.510348 | 0.423680 | 7.387706e-04 |
| DCC03 | long_only | 0.501949 | 0.501715 | 0.501496 | 0.257848 | 7.911628e-04 |
| DCCNL03 | unconstrained | 0.510786 | 0.510597 | 0.510441 | 0.420063 | 7.381772e-04 |
| DCCNL03 | long_only | 0.501933 | 0.501695 | 0.501474 | 0.257291 | 7.909153e-04 |
| SHARDRD01 | unconstrained | 0.460084 | 0.460027 | 0.459986 | 0.445629 | 6.433149e-04 |
| SHARDRD01 | long_only | 0.482177 | 0.482039 | 0.481914 | 0.282368 | 7.243391e-04 |
| BEKKS01 | unconstrained | 0.484625 | 0.484758 | 0.484902 | 0.254488 | 6.948960e-04 |
| BEKKS01 | long_only | 0.491998 | 0.492031 | 0.492070 | 0.148225 | 7.666679e-04 |
| LSTMBEKK-noC | unconstrained | 0.482022 | 0.482090 | 0.482166 | 0.194472 | 6.902686e-04 |
| LSTMBEKK-noC | long_only | 0.491240 | 0.491244 | 0.491252 | 0.116208 | 7.649457e-04 |
| V10 | unconstrained | 0.471752 | 0.471727 | 0.471709 | 0.195360 | 6.977791e-04 |
| V10 | long_only | 0.489621 | 0.489602 | 0.489587 | 0.127024 | 7.537033e-04 |
| V01 | unconstrained | 0.470687 | 0.470734 | 0.470788 | 0.181485 | 6.778601e-04 |
| V01 | long_only | 0.487386 | 0.487384 | 0.487385 | 0.109338 | 7.488399e-04 |
| V11 | unconstrained | 0.468748 | 0.468764 | 0.468785 | 0.183200 | 6.737699e-04 |
| V11 | long_only | 0.487183 | 0.487188 | 0.487197 | 0.113686 | 7.430221e-04 |
| EQUAL | equal_weight | 0.602626 | 0.602631 | 0.602635 | 0.014737 | 1.163499e-03 |

### R1 and R2 forecasts

| Version | Model | Book | Gross SD | Net 5 bps | Net 10 bps | Mean turnover | Proxy variance |
|---|---|---|---:|---:|---:|---:|---:|
| r1 | RW01 | unconstrained | 0.471297 | 0.471345 | 0.471439 | 1.014416 | 6.551737e-04 |
| r1 | RW01 | long_only | 0.489184 | 0.488961 | 0.488771 | 0.595084 | 7.216466e-04 |
| r2 | RW01 | unconstrained | 0.464412 | 0.464540 | 0.464692 | 0.612162 | 6.335494e-04 |
| r2 | RW01 | long_only | 0.484149 | 0.484037 | 0.483942 | 0.392206 | 7.103499e-04 |
| r1 | EWMA01 | unconstrained | 0.484275 | 0.484159 | 0.484049 | 0.205736 | 6.498420e-04 |
| r1 | EWMA01 | long_only | 0.488752 | 0.488656 | 0.488565 | 0.128207 | 7.309518e-04 |
| r2 | EWMA01 | unconstrained | 0.479371 | 0.479288 | 0.479208 | 0.148781 | 6.451010e-04 |
| r2 | EWMA01 | long_only | 0.487739 | 0.487660 | 0.487582 | 0.093817 | 7.269254e-04 |
| r1 | HARDRD01 | unconstrained | 0.460630 | 0.460700 | 0.460776 | 0.318982 | 6.376926e-04 |
| r1 | HARDRD01 | long_only | 0.485700 | 0.485679 | 0.485662 | 0.188324 | 7.207373e-04 |
| r2 | HARDRD01 | unconstrained | 0.461109 | 0.461219 | 0.461337 | 0.281537 | 6.419168e-04 |
| r2 | HARDRD01 | long_only | 0.485085 | 0.485078 | 0.485076 | 0.164613 | 7.247653e-04 |
| r1 | HARQDRD01 | unconstrained | 0.460940 | 0.460898 | 0.460868 | 0.451172 | 6.385085e-04 |
| r1 | HARQDRD01 | long_only | 0.485581 | 0.485375 | 0.485180 | 0.295552 | 7.186501e-04 |
| r2 | HARQDRD01 | unconstrained | 0.461575 | 0.461580 | 0.461597 | 0.395111 | 6.407513e-04 |
| r2 | HARQDRD01 | long_only | 0.485066 | 0.484884 | 0.484710 | 0.256907 | 7.208587e-04 |
| r1 | LWLIN01 | unconstrained | 0.480395 | 0.480395 | 0.480394 | 0.024076 | 7.458503e-04 |
| r1 | LWLIN01 | long_only | 0.494506 | 0.494503 | 0.494500 | 0.014731 | 7.946704e-04 |
| r2 | LWLIN01 | unconstrained | 0.467608 | 0.467599 | 0.467591 | 0.024780 | 6.975766e-04 |
| r2 | LWLIN01 | long_only | 0.491962 | 0.491954 | 0.491946 | 0.011054 | 7.565805e-04 |
| r1 | LWNL01 | unconstrained | 0.476692 | 0.476692 | 0.476693 | 0.027366 | 7.323425e-04 |
| r1 | LWNL01 | long_only | 0.493789 | 0.493789 | 0.493789 | 0.014878 | 7.875620e-04 |
| r2 | LWNL01 | unconstrained | 0.467841 | 0.467832 | 0.467824 | 0.024861 | 6.975824e-04 |
| r2 | LWNL01 | long_only | 0.492194 | 0.492187 | 0.492181 | 0.010786 | 7.563131e-04 |
| r1 | RIDGE01 | unconstrained | 0.460630 | 0.460700 | 0.460776 | 0.318982 | 6.376926e-04 |
| r1 | RIDGE01 | long_only | 0.485700 | 0.485679 | 0.485662 | 0.188324 | 7.207373e-04 |
| r2 | RIDGE01 | unconstrained | 0.461109 | 0.461219 | 0.461337 | 0.281537 | 6.419168e-04 |
| r2 | RIDGE01 | long_only | 0.485085 | 0.485078 | 0.485076 | 0.164613 | 7.247653e-04 |
| r1 | XGB08 | unconstrained | 0.479234 | 0.479040 | 0.478887 | 0.787226 | 6.714505e-04 |
| r1 | XGB08 | long_only | 0.487098 | 0.486907 | 0.486744 | 0.521507 | 7.334635e-04 |
| r2 | XGB08 | unconstrained | 0.472246 | 0.472121 | 0.472024 | 0.607205 | 6.618490e-04 |
| r2 | XGB08 | long_only | 0.484197 | 0.484007 | 0.483837 | 0.410260 | 7.292041e-04 |
| r1 | LSTM19 | unconstrained | 0.474900 | 0.474912 | 0.474932 | 0.189338 | 6.972345e-04 |
| r1 | LSTM19 | long_only | 0.489827 | 0.489818 | 0.489813 | 0.117239 | 7.525347e-04 |
| r2 | LSTM19 | unconstrained | 0.469572 | 0.469636 | 0.469703 | 0.113693 | 6.833276e-04 |
| r2 | LSTM19 | long_only | 0.489313 | 0.489334 | 0.489358 | 0.069726 | 7.462707e-04 |
| r1 | EWMA21 | unconstrained | 0.485344 | 0.485276 | 0.485221 | 0.378054 | 6.462487e-04 |
| r1 | EWMA21 | long_only | 0.486855 | 0.486733 | 0.486620 | 0.231146 | 7.225490e-04 |
| r2 | EWMA21 | unconstrained | 0.479103 | 0.479051 | 0.479004 | 0.264888 | 6.352239e-04 |
| r2 | EWMA21 | long_only | 0.485198 | 0.485058 | 0.484922 | 0.166142 | 7.149798e-04 |
| r1 | LSTM22 | unconstrained | 0.473778 | 0.473766 | 0.473761 | 0.193129 | 6.987814e-04 |
| r1 | LSTM22 | long_only | 0.490956 | 0.490930 | 0.490907 | 0.121757 | 7.573789e-04 |
| r2 | LSTM22 | unconstrained | 0.469028 | 0.469078 | 0.469131 | 0.110672 | 6.833961e-04 |
| r2 | LSTM22 | long_only | 0.489730 | 0.489747 | 0.489766 | 0.069543 | 7.486357e-04 |
| r1 | DCC02 | unconstrained | 0.511667 | 0.511433 | 0.511248 | 0.520663 | 7.345347e-04 |
| r1 | DCC02 | long_only | 0.501114 | 0.500835 | 0.500587 | 0.358914 | 7.852226e-04 |
| r2 | DCC02 | unconstrained | 0.484003 | 0.483949 | 0.483909 | 0.267468 | 6.821015e-04 |
| r2 | DCC02 | long_only | 0.490273 | 0.490105 | 0.489949 | 0.185101 | 7.500256e-04 |
| r1 | DCCNL02 | unconstrained | 0.511190 | 0.510961 | 0.510778 | 0.511721 | 7.332288e-04 |
| r1 | DCCNL02 | long_only | 0.501120 | 0.500835 | 0.500580 | 0.355369 | 7.847430e-04 |
| r2 | DCCNL02 | unconstrained | 0.484125 | 0.484072 | 0.484034 | 0.267473 | 6.819695e-04 |
| r2 | DCCNL02 | long_only | 0.490363 | 0.490200 | 0.490049 | 0.185847 | 7.504009e-04 |
| r1 | DCC03 | unconstrained | 0.510676 | 0.510496 | 0.510348 | 0.423680 | 7.387706e-04 |
| r1 | DCC03 | long_only | 0.501949 | 0.501715 | 0.501496 | 0.257848 | 7.911628e-04 |
| r2 | DCC03 | unconstrained | 0.479743 | 0.479804 | 0.479872 | 0.179263 | 6.802753e-04 |
| r2 | DCC03 | long_only | 0.489163 | 0.489096 | 0.489034 | 0.112213 | 7.460772e-04 |
| r1 | DCCNL03 | unconstrained | 0.510786 | 0.510597 | 0.510441 | 0.420063 | 7.381772e-04 |
| r1 | DCCNL03 | long_only | 0.501933 | 0.501695 | 0.501474 | 0.257291 | 7.909153e-04 |
| r2 | DCCNL03 | unconstrained | 0.479856 | 0.479911 | 0.479974 | 0.178135 | 6.800516e-04 |
| r2 | DCCNL03 | long_only | 0.489255 | 0.489188 | 0.489126 | 0.111632 | 7.463135e-04 |
| r1 | SHARDRD01 | unconstrained | 0.460084 | 0.460027 | 0.459986 | 0.445629 | 6.433149e-04 |
| r1 | SHARDRD01 | long_only | 0.482177 | 0.482039 | 0.481914 | 0.282368 | 7.243391e-04 |
| r2 | SHARDRD01 | unconstrained | 0.460830 | 0.460823 | 0.460827 | 0.374415 | 6.432644e-04 |
| r2 | SHARDRD01 | long_only | 0.481930 | 0.481825 | 0.481729 | 0.239846 | 7.246623e-04 |
| r1 | BEKKS01 | unconstrained | 0.484625 | 0.484758 | 0.484902 | 0.254488 | 6.948960e-04 |
| r1 | BEKKS01 | long_only | 0.491998 | 0.492031 | 0.492070 | 0.148226 | 7.666679e-04 |
| r2 | BEKKS01 | unconstrained | 0.473131 | 0.473270 | 0.473414 | 0.136956 | 6.750401e-04 |
| r2 | BEKKS01 | long_only | 0.488340 | 0.488385 | 0.488433 | 0.082274 | 7.467833e-04 |
| r1 | LSTMBEKK-noC | unconstrained | 0.482022 | 0.482090 | 0.482166 | 0.194472 | 6.902686e-04 |
| r1 | LSTMBEKK-noC | long_only | 0.491240 | 0.491244 | 0.491252 | 0.116208 | 7.649457e-04 |
| r2 | LSTMBEKK-noC | unconstrained | 0.473357 | 0.473456 | 0.473558 | 0.116117 | 6.742936e-04 |
| r2 | LSTMBEKK-noC | long_only | 0.488396 | 0.488417 | 0.488440 | 0.071299 | 7.472193e-04 |
| r1 | V10 | unconstrained | 0.471752 | 0.471727 | 0.471709 | 0.195360 | 6.977791e-04 |
| r1 | V10 | long_only | 0.489621 | 0.489602 | 0.489588 | 0.127024 | 7.537033e-04 |
| r2 | V10 | unconstrained | 0.468067 | 0.468091 | 0.468118 | 0.109580 | 6.853962e-04 |
| r2 | V10 | long_only | 0.489470 | 0.489491 | 0.489514 | 0.070472 | 7.477861e-04 |
| r1 | V01 | unconstrained | 0.470687 | 0.470734 | 0.470788 | 0.181485 | 6.778601e-04 |
| r1 | V01 | long_only | 0.487386 | 0.487384 | 0.487385 | 0.109338 | 7.488399e-04 |
| r2 | V01 | unconstrained | 0.468795 | 0.468868 | 0.468945 | 0.116577 | 6.739422e-04 |
| r2 | V01 | long_only | 0.488537 | 0.488546 | 0.488557 | 0.070387 | 7.455788e-04 |
| r1 | V11 | unconstrained | 0.468748 | 0.468764 | 0.468785 | 0.183200 | 6.737699e-04 |
| r1 | V11 | long_only | 0.487183 | 0.487188 | 0.487197 | 0.113686 | 7.430221e-04 |
| r2 | V11 | unconstrained | 0.467371 | 0.467433 | 0.467498 | 0.108228 | 6.729447e-04 |
| r2 | V11 | long_only | 0.488653 | 0.488668 | 0.488684 | 0.067303 | 7.428219e-04 |

## 8. Docs

The E1 register row is completed. The result commit is `251172c9528514b07e7ab7ca107f78abcacb4420`. `docs/PROJECT_STATE.md` records the checkpoint and names G1 next. `README.md` records that the SCREEN evaluation has been run and that CONFIRM has not been evaluated.

## 9. Next task

T2, the dated run-log tidy. Not started.
