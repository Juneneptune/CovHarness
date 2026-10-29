# README review

Proposal only. `README.md` was not restructured. Counts are word-boundary matches, case-insensitive, on the README at `docs/editorial-pass`. The same five words have the same counts on `main`. The branch changes one phrase from "resume" to "resumed run" and drops the ledger line from the tree.

## Hedging-word counts

| Word | Count |
| --- | --- |
| not | 120 |
| never | 6 |
| remains | 15 |
| pending | 3 |
| unimplemented | 5 |

"Not" includes ordinary negation as well as status caveats. The count is not a count of caveats alone.

## Caveats repeated in more than one section

The locked final holdout, and the statement that no confirmatory forecast or test has been run, appear in the opening status paragraph, the protocol calendar, the selection-results lead, the limitations paragraph, the reproduction commands, and the research-integrity section.

The statement that selection results are interim and do not establish superiority appears in the opening, the selection-results lead, the SPA paragraph, and the limitations paragraph.

The long-panel U.S.-equity experiment remaining pending for lack of production TAQ access appears in the opening, the equity-measurement section, and the limitations paragraph.

LSTM-BEKK-RC, GHAR, and a neural graph model being unimplemented appears in the opening, the LSTM-BEKK section, and the unresolved-work paragraph.

Pairwise Diebold–Mariano, Giacomini–White, Mincer–Zarnowitz, and Giacomini–Rossi tests not having been run on the selection sample appears in the inference overview and again in the procedure paragraphs.

The disagreement between the executed five-asset one-proxy design and the plan's N=30 two-proxy design appears in the opening discussion of the plan and again in the limitations comparison.

DCC and DCC-NL remaining in the roster after failing the admissibility checks appears in the opening and in the DCC section.

The statement that a passing test suite does not prove the absence of leakage appears in the limitations paragraph and in the research-integrity section.

## Proposed outline

State the status once, near the top, and do not repeat it under each model or test.

1. Status and limitations. What is built. What has been run, which is the development period and the selection period only. What is pending, including the locked final holdout, confirmatory inference, the long equity panel, and the unimplemented models. The repeated caveats above live only in this section.
2. Measurement. The frozen five-minute proxy, the halt, and the training-only shifted phases.
3. Protocol. Splits, the holdout lock, the 250-day window, and the 21-origin cadence.
4. Models. Contracts and information sets, without a second status paragraph.
5. Losses and inference. What exists in the harness, and which procedures were executed on the selection period.
6. Selection-period results. The numbers, with one pointer back to Status.
7. Reproduction and repository structure.
8. References.

This outline is not an edit.
