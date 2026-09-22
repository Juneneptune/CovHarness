# Covariance Forecasting Benchmark. Agent instructions

## Project objective

This repository is a reusable evaluation harness for one-day-ahead multivariate covariance forecasts. It is intended to remain applicable to models that are not yet in the repository.

The research question is whether machine-learning and deep-learning methods outperform strong econometric covariance forecasts under a leak-free, statistically defensible protocol. The priority is correctness and fairness, not a favourable result.

To the best of our knowledge, the protocol is constructed to make the comparison fair. A common pattern in the literature is to fine-tune a neural model carefully while leaving conventional baselines at default or poorly parameterized settings. That practice is not followed here. HAR, DCC, shrinkage, and the other prior models receive the same hyperparameter budget and the same rolling schedule as any method added later. Stochastic methods are trained on multiple pre-specified seeds, and the full seed distribution is reported rather than a single favourable draw. Whether a difference is large enough to support a claim is decided by HAC Diebold–Mariano tests, the Model Confidence Set, and Giacomini–White tests, not by a ranking of average losses.

A new architecture does not receive a different target, a richer information set, extra tuning, or a more favourable scoring rule. The harness is the stable object. The model roster is not.

---

## Writing and narrative

Public documentation, including `README.md` and `docs/PROJECT_STATE.md`, explains our design choices. It is not a lecture and not a survey of the literature.

Write in a formal register. Use first-person plural for project decisions. State the constraint, then the choice. For example, the oracle covariance is latent, unlike a fully observed supervised label, and therefore we evaluate against realized covariance under a pre-registered protocol.

Do not use a colon in headings or running text. Introduce lists with a full stop. Do not number work by "Day 1" or similar writing-day labels. Record calendar dates only.

Avoid hype, motivational prose, recruiter-facing commentary, and claims that experiments have not supported. Label work as implemented, validated, experimental, or planned. Never claim that deep learning outperforms econometrics unless the locked confirmatory evaluation supports that conclusion.

Do not turn `docs/PROJECT_STATE.md` into a transcript or chronological diary.

---

## Source of truth

Before substantial work, read these repository documents in this order.

1. `docs/PROJECT_STATE.md` for the current checkpoint, exact tests, unresolved issues, and next task.
2. `README.md` for the adopted methodological design and how implemented components are defined.
3. `INITIAL_CORE_BENCHMARK_PLAN.md` for the initial core plan and remaining planned scope. That file is not a locked roster or a final evaluation specification.
4. The existing implementation and tests for any component being changed.

Do not silently alter methodological choices. If implementation requirements conflict with the design, explain the conflict before changing the methodology.

ChatGPT continuation handoffs and chat-log lists are for other assistants. They are not a Cursor source of truth. Recover state from the repository documents above.

---

## Document roles

`AGENTS.md` tells coding agents how to work. `README.md` is the public methodological specification. `docs/PROJECT_STATE.md` is the detailed internal checkpoint. `INITIAL_CORE_BENCHMARK_PLAN.md` is the initial core plan and remaining planned scope. It is not final. Later models and evaluation extensions may be added.

Do not treat planned evaluation capabilities as implemented. Do not turn discussed or planned work into completed work.

Every substantive implemented component must be accompanied by an update to the relevant `README.md` section and an exact `docs/PROJECT_STATE.md` checkpoint. The README should describe the adopted definition, the implementation, boundary behaviour, and limitations. Rationale belongs where it changes how a reader should interpret the harness, for example proxy-robust scoring, information-set parity, or where nonlinear shrinkage enters a forecast.

---

## Persistent project state

`docs/PROJECT_STATE.md` is the persistent working memory for this repository. Read it at the beginning of substantial work. Update it after a meaningful unit of work.

A meaningful unit includes implementing a model, a realized-covariance estimator, a loss or inference procedure, or a data-pipeline stage; making an important methodological decision; fixing a bug that changes research results; completing a milestone; or discovering an unresolved limitation.

Record what was completed, files created or materially changed, tests actually run and their results, methodological decisions, known problems, and the next recommended task. Do not claim that something works unless it was run and verified. Do not invent test counts, dataset entitlements, or extraction results.

Do not modify `AGENTS.md` unless the user explicitly asks to change project instructions.

---

## Research integrity

Never introduce look-ahead bias.

- maintain chronological train, validation, and test ordering
- fit preprocessing and scalers on training data only
- never use future observations in feature construction
- never tune hyperparameters on the locked confirmation set
- never inspect confirmatory-test results during model development
- never silently remove failed models or configurations
- never choose losses or proxies after seeing which favours a model
- never choose crisis windows, fluctuation windows, thresholds, or state definitions after seeing which favours a model
- use explicit seeds where randomness is involved
- tune conventional models with the same budget and cadence as newer models
- report the full seed distribution, not the best seed

If a requested implementation risks leakage, stop and explain why.

---

## Covariance-specific requirements

The forecast target is a conditional covariance matrix. Realized covariance is a noisy proxy for that latent matrix, not ground truth.

For every covariance forecast, verify dimensions and symmetry, monitor eigenvalues, positive definiteness where required, and condition number, and never silently repair an invalid matrix. If a numerical repair is required, record the failure, record the repair method, and preserve diagnostics so that the frequency of repairs can be reported.

Keep the training objective, the statistical evaluation loss, and the economic evaluation criterion separate.

---

## Initial benchmark scope

Tier-1 models include random-walk realized covariance, EWMA, HAR-DRD, HARQ-DRD, Ledoit–Wolf linear shrinkage, Ledoit–Wolf nonlinear shrinkage, DCC, DCC-NL, and Ridge-DRD. Later models enter the same interface.

Primary evaluation includes multivariate QLIKE / Stein loss, Frobenius loss, variance and correlation decomposition, global-minimum-variance portfolio evaluation, Diebold–Mariano tests with HAC standard errors, SPA, the Model Confidence Set, Giacomini–White tests, standard and state-augmented Mincer–Zarnowitz diagnostics, covariance-aware MZ-GLS, and Giacomini–Rossi fluctuation analysis. State-dependent threshold forecast evaluation is an optional robustness extension that requires a separately specified implementation and valid inference for threshold search.

Do not replace these with easier approximations without discussing it first.

---

## Implementation workflow

For each new method, proceed in this order.

1. State what is being implemented.
2. State the mathematical inputs and outputs.
3. Identify leakage risks.
4. Identify numerical risks.
5. Implement the smallest correct version.
6. Add synthetic or unit tests.
7. Run the tests.
8. Integrate it into the benchmark only after validation.
9. Update `docs/PROJECT_STATE.md` with what was completed, files changed, tests actually run, and the next task.
10. Update the relevant `README.md` section if the public definition, implementation, boundary behaviour, or limitation of a component changed.

Do not generate large amounts of code before verifying the mathematical specification.

Do not begin WRDS extraction, guess vendor table names, or change estimator behaviour unless the current task explicitly authorizes that work.

---

## Code quality

Do not write sloppy code. Do not break the repository or package structure to take a shortcut.

The layout under `src/covharness/` is the implementation home. New code goes in the existing package that owns it. Do not dump modules at the repo root, in `scripts/`, or in `notebooks/` when they belong in the package. Do not flatten, rename, or invent a parallel folder tree. If a change would require moving that layout, stop and explain why before doing it.

Prefer the smallest correct implementation that fits the current tree, small modules, explicit interfaces, readable numerical code, descriptive variable names, type hints where useful, docstrings with dimensions and mathematical meaning, centralized configuration, reproducible seeds, and tests for important mathematical routines.

For nontrivial matrix code, include expected dimensions in comments or docstrings. Avoid clever vectorization when it makes the mathematics difficult to audit. A passing linter or a one-file rewrite is not a substitute for a clean module boundary.

Write a one-line block comment above each coherent chunk of implementation logic. A chunk is a unit a reader would use as a debugging checkpoint, such as input checks, a filter, a sort, a group aggregation, a neighborhood statistic, or an output assembly. Do not comment every line. Do not restate syntax. Name the operation in plain language so a failure can be located by scanning, for example "drop quotes outside 09:30-16:00" or "collapse duplicate timestamps to median bid and ask". Keep each of these comments to one line. Docstrings remain the place for the mathematical definition. These block comments are for orientation while reading the body.

---

## Testing

For important estimators, models, losses, and inference procedures, test expected shape, symmetry, PSD or PD behaviour, known special cases, no-lookahead indexing, deterministic behaviour under fixed seeds, comparison against a trusted implementation, and synthetic data where the true covariance or null hypothesis is known.

A passing import is not sufficient validation.
