# Task T1 report

## 1. Outcome

Task T1. Executed. Public wording was revised on branch `docs/editorial-pass`. No design commitment, result, or caveat was changed. The branch is unmerged and unpushed. This report and `results/post_screen/T1/README_REVIEW.md` are on `main`.

## 2. Starting state

Branch `main` at `fd6cb6b4dc68d63ba055739698d65c83f4d6fe79`, after the D3a report commit. The working tree was clean. `docs/editorial-pass` was created from that commit. Documents read were `INITIAL_CORE_BENCHMARK_PLAN.md`, `docs/project_ledger.md`, `README.md`, `AGENTS.md`, and the adopted-rule passages in `PREREGISTRATION_DRAFT.md`, `docs/PROJECT_STATE.md`, and `docs/INFERENCE_METHODS.md`.

## 3. Spec commit

None. T1 is exempt from the register-spec rule. It changes no protocol. `PREREGISTRATION_DRAFT.md` was not edited.

## 4. Changes

Branch commits, one file each.

| Commit | File | Change |
| --- | --- | --- |
| `646c88e63fd884ff4b0b4941f4af5503e53daf2d` | `INITIAL_CORE_BENCHMARK_PLAN.md` | Project voice. 55 insertions, 98 deletions. |
| `fcc0a67c6985216eb3cfcf1fc878d8fd3515f4e8` | `docs/project_ledger.md` | Removed. 395 deletions. |
| `e1449301c9d24fb995ccfec088c22d2116628eb7` | `README.md` | "resume" became "resumed run". Ledger line removed from the tree. 2 insertions, 3 deletions. |
| `be2a6e0c87c2cb3d2f1d83b4522a52c13bd7cdd5` | `AGENTS.md` | The assistant-handoff sentence was removed. 2 deletions. |

The parent of the first editorial commit is `fd6cb6b4dc68d63ba055739698d65c83f4d6fe79`.

The plan keeps the roster, losses, proxies, splits, cadence, tuning budget, seeds, inference sequence, diagnostics, data gate, and limitations. The editorial line under the title points at `913df10b236ffebeab463f20c2df8376bf8f8ff2`, the last commit that had touched the file before this pass. "Scope tiers" replaces "PART -1". "Scope reduction rules" replaces "PART 7 — CUT ORDER". The three colon headings are now "0.2 Acquisition sources that were considered", "2.6 Analytical nonlinear shrinkage, not QuEST", and "3.4 Screening with SPA and MCS". The vendor price table and the dollar figures in that table were removed. The sources and the reason each was considered remain. The five-dollar price screen in the universe rule remains, because it is a design threshold rather than a vendor quote.

Ledger check, before deletion. No adopted rule lived only in the ledger, so the file was removed. Git history keeps it.

| Adopted rule | Where it now lives |
| --- | --- |
| Median rank across four 125-day subperiods, then mean rank, with full-period mean loss as robustness | `README.md` selection rule. `PREREGISTRATION_DRAFT.md` finalist note. |
| Screening, then a later locked confirmation | `README.md` protocol. `PREREGISTRATION_DRAFT.md`. |
| Quarticity as a measurement-error scale, a HARQ feature, and a Giacomini–White instrument | `README.md` realized-quarticity definition and HARQ-DRD. `PREREGISTRATION_DRAFT.md` Block 3C. `INITIAL_CORE_BENCHMARK_PLAN.md` section 0.8. |
| DCC-NL shrinks the correlation target with analytical nonlinear shrinkage, not QuEST | `README.md` DCC paragraph. `docs/PROJECT_STATE.md` Ledoit–Wolf and DCC notes. |
| Two Giacomini–White tests, three instruments each, Bonferroni | `PREREGISTRATION_DRAFT.md` Block 3C. `docs/INFERENCE_METHODS.md`. `docs/PROJECT_STATE.md`. |
| Fixed window of 250 days and a common re-estimation cadence | `README.md` protocol. `INITIAL_CORE_BENCHMARK_PLAN.md` sections 0.9 and 2.5. |
| Clark–West only for nested scalar squared error | `README.md`, `PREREGISTRATION_DRAFT.md`, `docs/INFERENCE_METHODS.md`. |
| Same information-set and tuning-budget constraint | `README.md` opening and the LSTM versus HARQ note. |

[M] decisions. None was upgraded to a checked citation.

1. The cleaning checklist is still attributed to Barndorff-Nielsen, Hansen, Lunde, and Shephard, "Realized kernels in practice", and is marked [M] (unverified). The checklist items were not re-read against the paper.
2. Analytical nonlinear shrinkage is still Ledoit–Wolf 2020. The claim that its accuracy is essentially the same as QuEST stays [M] (unverified).
3. The plan's 100-portfolio Mincer–Zarnowitz design stays attributed to Patton and Sheppard as [M] (unverified). That design was not replaced by the later pooled-vech implementation.

[V] tags on the LSTM-BEKK recursion, the training description, and the paper's 70/15/15 split were left in place.

`README_REVIEW.md` is a proposal. The README was not reorganized.

## 5. Problems

None observed. The ledger check did not find a rule that existed only there, so deletion was not stopped.

## 6. Verification

Voice counts are `git grep` word matches on `main` (before) and `docs/editorial-pass` (after).

| File | you/your before | you/your after | ChatGPT, Claude, or Cursor before | after | listed hype phrases before | after |
| --- | --- | --- | --- | --- | --- | --- |
| `INITIAL_CORE_BENCHMARK_PLAN.md` | 32 | 0 | 0 | 0 | 6 | 0 |
| `README.md` | 0 | 0 | 0 | 0 | 0 | 0 |
| `AGENTS.md` | 0 | 0 | 2 | 0 | 0 | 0 |
| `docs/project_ledger.md` | 3 | 0 | 2 | 0 | 0 | 0 |

`git diff --stat main...docs/editorial-pass`

```
 AGENTS.md                      |   2 -
 INITIAL_CORE_BENCHMARK_PLAN.md | 153 ++++++----------
 README.md                      |   5 +-
 docs/project_ledger.md         | 395 -----------------------------------------
 4 files changed, 57 insertions(+), 498 deletions(-)
```

No file outside that list changed on the branch. `src/`, `tests/`, `configs/`, `scripts/`, `results/`, `notebooks/`, `PREREGISTRATION_DRAFT.md`, and `docs/PROJECT_STATE.md` are absent from the diff.

Markdown links in `README.md` on the branch were checked by resolving the path before any `#` anchor. Every internal link resolved, including `docs/PROJECT_STATE.md`, `docs/MODEL_IMPLEMENTATION.md`, `docs/INFERENCE_METHODS.md`, `docs/EQUITY_MEASUREMENT.md`, `PREREGISTRATION_DRAFT.md`, `INITIAL_CORE_BENCHMARK_PLAN.md`, the Binance measurement, panel, and protocol notes, the two result images, the model source files, `configs/binance_open_data_core.yaml`, and `LICENSE`. No link pointed at `docs/project_ledger.md`.

Remaining `docs/*.md` files were not edited. A case-insensitive search for second person, ChatGPT, Claude, and the listed hype phrases found one hit. `docs/EMPIRICAL_DATA_FEASIBILITY.md` line 53 quotes a WRDS error, "You do not have permission to access the taqmsec library". That is a vendor message, not an address to the reader.

`git checkout main` left `docs/editorial-pass` in place. It was not merged and not pushed.

## 7. Artifacts

`results/post_screen/T1/REPORT.md` and `results/post_screen/T1/README_REVIEW.md` are on `main`. The editorial branch tip is `be2a6e0c87c2cb3d2f1d83b4522a52c13bd7cdd5`. No 2026-09-21 artifact, A1 file, or `configs/binance_open_data_core.yaml` was modified.

## 8. Docs

`PREREGISTRATION_DRAFT.md` and `docs/PROJECT_STATE.md` were not updated. T1 is not a protocol amendment. The README change on the branch is the one resumed-run phrase and the ledger line in the tree. `AGENTS.md` lost only the handoff sentence.

## 9. Next task

None in this queue. B1, C1, C2, D1, D2, D3b, D4, E1, G1, and F1 were not started.
