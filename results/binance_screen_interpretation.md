# Binance SCREEN descriptive interpretation

Date 2026-09-21. This note uses existing SCREEN artifacts only. No model was fit or refit. VALIDATION was not rerun. SPA and MCS bootstraps were not rerun. CONFIRM remains locked and was not read.

Sign convention on every comparison below. \(d_t=L_{\mathrm{HARQ},t}-L_{\mathrm{comp},t}\). Negative \(d_t\) means HARQ-DRD has lower loss on that date. Lower loss is better. QLIKE differences are not reported as percentages.

These quantities are out-of-sample SCREEN selection diagnostics on the five-asset Binance panel. They are not confirmatory rankings.

---

## A. Empirical facts

Verified hashes match `results/binance_screen_manifest.json`. Production panel `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`. SCREEN configuration `b07011a798c36fc25e117b69d71f6c8eb5eb1ed756e97c39c19e29542af6203c`. VALIDATION selection `f6d9830302b03afa2f3f45cd15576d70fbae4b96482af5c71d935822bc132f4b`.

Calendar. 500 SCREEN targets from 2023-11-30 through 2025-04-12. Frozen 125-target blocks. Block 1 2023-11-30 through 2024-04-02. Block 2 2024-04-03 through 2024-08-05. Block 3 2024-08-06 through 2024-12-08. Block 4 2024-12-09 through 2025-04-12. Loss-panel means reproduce the saved summary to floating-point roundoff of order \(10^{-15}\). SPA saved mean differentials reproduce exactly as \(\mathrm{mean}(L_{\mathrm{HAR}}-L_k)\) from the loss panels. The SPA and MCS bootstrap p-values were not recomputed.

Support. Nine of eleven representatives have complete 500-target primary-loss support. DCC and DCC-NL have \(n=0\). They failed at origin 2023-12-20, the second SCREEN refit, because ZeroMean GARCH(1,1) for asset 1 returned \(\omega\approx 4.966\times 10^{-12}\), \(a=0\), \(b=1\), which violates the project contract \(a+b<1\). No substitute specification was introduced. SPA and MCS used the nine complete columns.

Exact HAR/RIDGE tie. The saved HAR-DRD and Ridge-DRD QLIKE series are identical, and the squared-Frobenius series are identical (`numpy.array_equal` true, max absolute gap 0). RIDGE01 is labeled `exact_benchmark_tie` and is excluded only from SPA studentization. It remains in tables and MCS. Descriptive ranks in the summary still list HAR-DRD as 3 and Ridge-DRD as 4 because those ranks used `argsort` rather than average ranks.

Finalists. Econometric HARQ-DRD (`HARQDRD01`). Deep-learning LSTM-BEKK (`LSTM19` equal-weight covariance ensemble), by construction. Full-SCREEN mean-QLIKE robustness leader among complete econometric families is also HARQ-DRD.

Headline losses from `results/binance_screen_model_summary.csv`.

| Family | ID | n | Mean QLIKE | Mean \(L_F^2\) | QLIKE rank | \(L_F^2\) rank | Repairs | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RW | RW01 | 500 | -31.270726 | \(1.078193\times 10^{-4}\) | 7 | 9 | 0 | complete |
| EWMA | EWMA01 | 500 | -31.776804 | \(7.800251\times 10^{-5}\) | 2 | 4 | 0 | complete |
| HAR-DRD | HARDRD01 | 500 | -31.759431 | \(7.489333\times 10^{-5}\) | 3 | 2 | 0 | complete |
| HARQ-DRD | HARQDRD01 | 500 | -31.862063 | \(7.487763\times 10^{-5}\) | 1 | 1 | 2 | complete |
| LW-linear | LWLIN01 | 500 | -30.235152 | \(9.262341\times 10^{-5}\) | 8 | 8 | 0 | complete |
| LW-NL | LWNL01 | 500 | -30.162208 | \(9.202238\times 10^{-5}\) | 9 | 7 | 0 | complete |
| DCC | DCC01 | 0 |  |  |  |  | 0 | failed |
| DCC-NL | DCCNL01 | 0 |  |  |  |  | 0 | failed |
| Ridge-DRD | RIDGE01 | 500 | -31.759431 | \(7.489333\times 10^{-5}\) | 4 | 3 | 0 | complete |
| XGBoost-DRD | XGB08 | 500 | -31.711532 | \(7.856037\times 10^{-5}\) | 5 | 5 | 0 | complete |
| LSTM-BEKK | LSTM19 | 500 | -31.408752 | \(8.269241\times 10^{-5}\) | 6 | 6 | 0 | complete |

Descriptive ranks are not confirmatory winners.

SPA versus HAR-DRD, \(B=5000\), seed 20260913, \(\ell=7\), consistent recentering. QLIKE statistic 3.001030, consistent p-value 0.0016. Frobenius statistic 0.010172, consistent p-value 0.8792. Exact tie Ridge-DRD on both channels.

Primary QLIKE MCS \((T_R,e_R)\) at \(\alpha=0.10\). Membership EWMA and HARQ-DRD. p-values. RW 0.0002. EWMA 0.3154. HAR-DRD 0.0124. HARQ-DRD 1.0. LW-linear 0.0002. LW-NL 0.0004. Ridge-DRD 0.0124. XGBoost-DRD 0.0124. LSTM-BEKK 0.001.

Primary Frobenius MCS \((T_R,e_R)\) at \(\alpha=0.10\). Membership RW, EWMA, HAR-DRD, HARQ-DRD, Ridge-DRD, XGBoost-DRD, and LSTM-BEKK. p-values. RW 0.1084. EWMA 0.286. HAR-DRD 0.9932. HARQ-DRD 1.0. LW-linear 0.011. LW-NL 0.0112. Ridge-DRD 0.9932. XGBoost-DRD 0.46. LSTM-BEKK 0.1244.

Econometric median ranks among the six complete eligible families.

| Family | B1 | B2 | B3 | B4 | Median | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RW | 4 | 6 | 4 | 4 | 4.0 | 4.50 |
| EWMA | 1 | 3 | 1 | 3 | 2.0 | 2.00 |
| HAR-DRD | 3 | 2 | 3 | 1 | 2.5 | 2.25 |
| HARQ-DRD | 2 | 1 | 2 | 2 | 2.0 | 1.75 |
| LW-linear | 5 | 5 | 5 | 5 | 5.0 | 5.00 |
| LW-NL | 6 | 4 | 6 | 6 | 6.0 | 5.50 |

HARQ-DRD and EWMA share median rank 2.0. HARQ-DRD is selected by the frozen mean-rank tie-break.

LSTM19 seeds 0 through 4 all completed. Seed mean QLIKE \(-31.258560\), \(-31.289213\), \(-31.211141\), \(-31.351885\), \(-31.286159\). Range \([-31.351885,-31.211141]\). Sample standard deviation \(0.051210\). Seed mean squared Frobenius \(8.347721\times 10^{-5}\), \(8.424119\times 10^{-5}\), \(8.610060\times 10^{-5}\), \(8.148141\times 10^{-5}\), \(8.191727\times 10^{-5}\). Range \([8.148141\times 10^{-5},8.610060\times 10^{-5}]\). Sample standard deviation \(1.862579\times 10^{-6}\). Ensemble mean QLIKE \(-31.408752\). Ensemble mean squared Frobenius \(8.269241\times 10^{-5}\). The ensemble scores averaged covariance matrices, then evaluates loss. It is not a selected seed. Seeds are repeated training draws of one configuration, not five independent markets.

HARQ-DRD repairs. Headline diagnostics record `repair_count=2` and do not store dates. The existing SCREEN checkpoint `results/binance_screen_checkpoints/HARQ-DRD_HARQDRD01.npz` stores a length-500 `repaired` flag. The two true flags are target 2024-12-10 (origin 2024-12-09, position 376, `refit=False`) and target 2025-03-03 (origin 2025-03-02, position 459, `refit=False`). Both sit in block 4. That checkpoint does not store the per-date validity flag or a `repair_method` string. The frozen HARQ-DRD contract uses the current-origin estimation-window mean as the only headline repair. Which raw-forecast check failed on those two dates is not recorded in the saved SCREEN diagnostics.

Effect sizes. Mean QLIKE difference is \(\mathrm{mean}(L_{\mathrm{HARQ}}-L_{\mathrm{comp}})\). Mean squared-Frobenius difference uses the same sign. Percentage reduction in squared Frobenius uses the comparator mean as denominator, \((\bar L_{\mathrm{comp}}^F-\bar L_{\mathrm{HARQ}}^F)/\bar L_{\mathrm{comp}}^F\). Full-SCREEN values.

| Comparator | Mean QLIKE \(d\) | Mean \(L_F^2\) \(d\) | \(L_F^2\) reduction versus comparator |
| --- | ---: | ---: | ---: |
| LSTM-BEKK | -0.453310 | \(-7.814785\times 10^{-6}\) | 9.450% |
| XGBoost-DRD | -0.150531 | \(-3.682745\times 10^{-6}\) | 4.688% |
| EWMA | -0.085259 | \(-3.124884\times 10^{-6}\) | 4.006% |
| HAR-DRD | -0.102632 | \(-1.570803\times 10^{-8}\) | 0.021% |
| RW | -0.591337 | \(-3.294170\times 10^{-5}\) | 30.553% |

Block-level HARQ-DRD versus LSTM-BEKK.

| Block | Mean QLIKE \(d\) | Mean \(L_F^2\) \(d\) | \(L_F^2\) reduction | Share of cumulative QLIKE \(d\) | Share of cumulative \(L_F^2\) \(d\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.319510 | \(-4.231530\times 10^{-6}\) | 11.049% | 17.6% | 13.5% |
| 2 | -0.325798 | \(-3.630347\times 10^{-6}\) | 4.179% | 18.0% | 11.6% |
| 3 | -0.573981 | \(-3.686079\times 10^{-6}\) | 12.776% | 31.7% | 11.8% |
| 4 | -0.593952 | \(-1.971118\times 10^{-5}\) | 11.152% | 32.8% | 63.1% |

Daily HARQ-DRD versus LSTM-BEKK. Mean QLIKE \(d\) \(-0.453310\). Median QLIKE \(d\) \(-0.294072\). Mean squared-Frobenius \(d\) \(-7.814785\times 10^{-6}\). Median squared-Frobenius \(d\) \(-1.320123\times 10^{-7}\). HARQ-DRD has lower QLIKE on 69.6% of dates and lower squared Frobenius on 52.0% of dates. Cumulative QLIKE path at block ends. \(-39.939\), \(-80.664\), \(-152.411\), \(-226.655\). Cumulative squared-Frobenius path at the same ends. \(-5.289\times 10^{-4}\), \(-9.827\times 10^{-4}\), \(-1.443\times 10^{-3}\), \(-3.907\times 10^{-3}\).

Block-level HARQ-DRD versus EWMA QLIKE \(d\). Block 1 \(+0.111164\) (EWMA lower). Block 2 \(-0.375981\). Block 3 \(+0.048094\) (EWMA lower). Block 4 \(-0.124312\). Block-level HARQ-DRD versus HAR-DRD QLIKE \(d\). Blocks 1–3 negative. Block 4 \(+0.054310\) (HAR-DRD lower). Full table in `results/binance_screen_effect_sizes.csv`.

---

## B. Interpretation

Primary reduced QLIKE and squared Frobenius do not tell the same screening story.

On QLIKE, HARQ-DRD is the lowest full-SCREEN mean among complete models, the econometric median-rank finalist, the last MCS survivor, and the alternative that produces the SPA statistic versus HAR-DRD. LSTM-BEKK is outside the primary QLIKE MCS at \(\alpha=0.10\) (p-value 0.001). EWMA remains inside that MCS with HARQ-DRD. The QLIKE SPA consistent p-value 0.0016 therefore records that some alternative, here HARQ-DRD, beats the HAR-DRD benchmark on SCREEN. It does not test HARQ-DRD against LSTM-BEKK, and it is not a CONFIRM result.

On squared Frobenius, HARQ-DRD and HAR-DRD are almost the same (0.021% reduction). The Frobenius SPA consistent p-value is 0.8792, so there is no SCREEN rejection of the HAR-DRD benchmark under that loss. The primary Frobenius MCS at \(\alpha=0.10\) is large and includes both finalists. LSTM-BEKK belongs to that Frobenius MCS (p-value 0.1244) while remaining outside the QLIKE MCS. A large MCS is an allowed result when forecasts are close.

The HARQ-DRD versus LSTM-BEKK QLIKE gap is present in all four frozen blocks. Mean \(d\) is negative in each block. Later blocks contribute more. Blocks 3 and 4 together account for 64.4% of the cumulative QLIKE differential. The advantage is therefore broad rather than confined to a single 125-day window, and it is not uniform. The corresponding Frobenius advantage is more concentrated. Block 4 alone accounts for 63.1% of the cumulative squared-Frobenius differential. The median daily Frobenius differential is much closer to zero than the mean, which is consistent with a few large dates rather than a steady gap.

HARQ-DRD versus EWMA is the relevant econometric-finalist race. EWMA has lower mean QLIKE in blocks 1 and 3. HARQ-DRD has lower mean QLIKE in blocks 2 and 4 and over the full 500 dates. The frozen median-rank rule therefore selected HARQ-DRD on a tie of median rank 2.0, not because EWMA was dominated in every block.

HARQ-DRD versus HAR-DRD is a small QLIKE gap and a negligible Frobenius gap. Block 4 even favours HAR-DRD on both losses. The quarticity term helped overall SCREEN QLIKE and did not produce a large Frobenius separation.

RIDGE01 is not a distinct SCREEN forecast. It is the nested \(\lambda=0\) HAR solution, as selected on VALIDATION, and it copies HAR-DRD exactly.

LSTM seed scores vary by 0.141 QLIKE units. That spread is smaller than the 0.453 ensemble QLIKE gap versus HARQ-DRD. The ensemble mean QLIKE is lower than every seed mean because the reported LSTM column averages covariance matrices before scoring. No seed is a SCREEN representative.

DCC and DCC-NL are absent from SPA, MCS, and the econometric ranking because they failed the existing IGARCH contract. That is a surfaced fit failure, not evidence against DCC on dates they never produced.

None of these statements is a locked-test-set result, a crypto-market generalization, or a completed paper claim.

---

## C. Resume-safe statements

### Conservative

On the 500-target Binance SCREEN block, HARQ-DRD (`HARQDRD01`) is the frozen econometric finalist and LSTM-BEKK (`LSTM19` equal-weight covariance ensemble) is the deep-learning finalist by construction. Primary reduced-QLIKE MCS at \(\alpha=0.10\) retains HARQ-DRD and EWMA and excludes LSTM-BEKK. Squared-Frobenius MCS at the same level retains both finalists. CONFIRM was not evaluated. These SCREEN quantities do not establish final superiority.

### Quantitative

Across the 500 SCREEN targets, mean reduced QLIKE of HARQ-DRD minus LSTM-BEKK equals \(-0.453310\), so HARQ-DRD has lower mean QLIKE. The same pair has mean squared-Frobenius difference \(-7.814785\times 10^{-6}\), a 9.450% reduction using the LSTM ensemble as denominator. The QLIKE gap is negative in all four 125-day blocks. The Frobenius gap is concentrated in the last block (63.1% of the cumulative differential). Hansen SPA versus HAR-DRD rejects equal predictive ability on QLIKE (statistic 3.001, consistent p-value 0.0016) and does not reject on squared Frobenius (p-value 0.8792). CONFIRM remains locked.

### Highly technical

Under the pre-registered Stage-2 rule, the econometric headline slot is filled by HARQDRD01 among the six complete eligible families by lowest median of four QLIKE block ranks, with the mean-rank tie-break against EWMA (median 2.0 versus 2.0, mean 1.75 versus 2.00). LSTM19 is the unique first-stage deep-learning architecture, so it occupies the DL slot without a SCREEN architecture contest. The equal-weight covariance ensemble, not a seed, is the LSTM loss column. Primary \((T_R,e_R)\) MCS membership at \(\alpha=0.10\) is \(\{\mathrm{EWMA},\mathrm{HARQ}\text{-}\mathrm{DRD}\}\) for reduced QLIKE and a seven-model set including both finalists for squared Frobenius. RIDGE01 is an identically zero HAR-DRD loss differential and is labeled `exact_benchmark_tie`. DCC and DCC-NL contribute no SCREEN losses after the origin-2023-12-20 IGARCH failure. Pairwise Diebold–Mariano, Giacomini–White, Mincer–Zarnowitz, and Giacomini–Rossi tests were not run. The locked CONFIRM block 2025-04-13 through 2026-08-25 has not been opened.
