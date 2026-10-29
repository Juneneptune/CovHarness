# Task B2 report

## 1. Outcome

Task B2. Executed. SCREEN SPA and MCS were rerun at mean block lengths 7, 14, and 30 with the existing procedures, \(B=5000\), and seed 20260913. Length 7 matches the frozen 2026-09-21 p-values exactly. QLIKE SPA rejects at 5 percent at every length. Frobenius SPA does not. The only Model Confidence Set membership change at \(\alpha=0.10\) is that RW leaves the Frobenius range set when the block length rises from 7 to 14. The finalist rule was not changed. The CONFIRM block length was not chosen.

## 2. Starting state

Branch `main` at spec commit `8d0cbfb7184290f7db819e56c45e0da3548c831c`. The working tree was clean. Documents and artifacts read were the Post-SCREEN register, the A1 report and addendum, `src/covharness/inference/spa.py`, `mcs.py`, and `bootstrap.py`, `src/covharness/protocol/binance_screen.py`, and the frozen SPA, MCS, and sub-block rank files.

## 3. Spec commit

`8d0cbfb7184290f7db819e56c45e0da3548c831c`, recorded before any B2 empirical number was computed.

## 4. Changes

`screen_spa` and `screen_mcs` accept `block_length` and pass it to the existing bootstrap. The default remains the cube-root rule. The entry script is `scripts/post_screen_block_sensitivity.py`. RIDGE01 remains an exact benchmark tie and is excluded from the SPA statistic. Range and max MCS pairs are not crossed. Consistent recentering is the primary SPA p-value. Lower and upper recenterings are reported.

The CONFIRM block length is not chosen by these p-values. The register states that it is decided at G1 and tied to B1's frozen bandwidth. Neither G1 nor B1 was run.

## 5. Problems

The wrapper unit test failed twice before the empirical run. `screen_spa` requires 500 rows, and its default benchmark label is HAR-DRD. The synthetic panel was corrected to that contract. The corrected test passed. The SCREEN script then exited 0, including the exact length-7 comparison. No second empirical attempt was required.

## 6. Verification

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_post_screen_b2.py`

Exit code 0. 1 passed.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -u scripts/post_screen_block_sensitivity.py`

Exit code 0. Elapsed 6.5 seconds. Length 7 matched saved SPA p-values, the SPA statistic, exact ties, and MCS p-values and 0.10 membership on both losses. Frozen file hashes were identical before and after. CONFIRM was not read. `unlock_confirm` was not passed. No new bootstrap was written.

## 7. Artifacts

| Path | SHA-256 |
| --- | --- |
| `results/post_screen/B2/b2_summary.json` | `4766f8592a71c7da65fa36108a716b61255490d13cc22e6f528a6234c09d2e30` |
| `results/post_screen/B2/b2_spa.csv` | `68f5b03a7e6ea3c088776b658bf0bd8ca7b97889674061e97ac7eddcfcddf86b` |
| `results/post_screen/B2/b2_mcs.csv` | `2c94da97df5f5471c23083cc3e70ca478a8aa22eb18e2649b3be156926744345` |

Spec commit `8d0cbfb7184290f7db819e56c45e0da3548c831c`. Result commit `f9c6b58a79941bcc72392f6d95943d54b336a9dd`. Seed 20260913. \(B=5000\). Untouched frozen hashes are SPA `992dd16cb0d9cdf3d2511b39c093c1f1daf4d0062fc9852c74c6f0fc41e46420`, QLIKE MCS `234f195c5c74d3b1b42de921bd545895e9ac37cf7d39ede59c2f9a7ade59cefb`, Frobenius MCS `ef1e2fd798b0d3ec82a23033100e9b23bbc03e2a8b618aecb8e1b792f820735d`, and sub-block ranks `fb2decc2c158171ab22d8ac2df23c53096fd5d3c6e9afedf634cf96338578514`.

SPA p-values. Consistent is primary.

| Loss | Block | Lower | Consistent | Upper |
| --- | --- | --- | --- | --- |
| QLIKE | 7 | 0.0016 | 0.0016 | 0.0016 |
| QLIKE | 14 | 0.0030 | 0.0030 | 0.0032 |
| QLIKE | 30 | 0.0020 | 0.0022 | 0.0022 |
| Frobenius | 7 | 0.5692 | 0.8792 | 0.9634 |
| Frobenius | 14 | 0.5578 | 0.8574 | 0.9398 |
| Frobenius | 30 | 0.5526 | 0.8688 | 0.9092 |

QLIKE range membership at 0.10 is EWMA and HARQ-DRD at every length. The max companion adds HAR-DRD, Ridge-DRD, and XGBoost-DRD at every length. LSTM-BEKK is outside both. Frobenius range membership drops RW after length 7. RW's range p-values are 0.1084, 0.0714, and 0.0496. The Frobenius max companion already excludes RW at length 7, and its 0.10 membership does not change. Sub-block ranks, which determine the econometric finalist, were not recomputed.

## 8. Docs

The register status for B2 is completed. The result commit cell holds `f9c6b58a79941bcc72392f6d95943d54b336a9dd`. `docs/PROJECT_STATE.md` records the membership change and that the CONFIRM length was not chosen. `README.md` names `scripts/post_screen_block_sensitivity.py`.

## 9. Next task

D3a, shifted-day realized-covariance panels for training augmentation. Not started.
