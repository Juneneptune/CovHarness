# Covariance Forecasting Benchmark — Agent Instructions

## Project objective

This repository implements a rigorous benchmark for multivariate

conditional covariance forecasting in financial markets.

The central research question is whether modern machine-learning and

deep-learning methods genuinely outperform strong econometric covariance

forecasting methods under a fair, leak-free, statistically defensible

evaluation protocol.

The priority is correctness and defensibility, not producing a favorable

result.

---

## Source of truth

Before substantial work:

1. Read `docs/PROJECT_STATE.md`.

2. Read the relevant sections of `design_v4_demo_plan.md`.

3. Inspect the existing implementation before adding new abstractions.

`design_v4_demo_plan.md` defines the current benchmark methodology.

Do not silently alter methodological choices from it.

If implementation requirements conflict with the design, explain the

conflict before changing the methodology.

---

## Persistent project state

`docs/PROJECT_STATE.md` is the persistent working memory for this repo.

At the beginning of substantial work, read it.

After completing a meaningful unit of work, update it.

A meaningful unit includes:

- implementing a model

- implementing a realized-covariance estimator

- implementing a loss or inference procedure

- completing a data-pipeline stage

- making an important methodological decision

- fixing a bug that changes research results

- completing a milestone

- discovering an unresolved limitation or blocker

Update the state file with:

- what was completed

- files created or materially changed

- tests actually run and their results

- methodological decisions made

- known problems or limitations

- the next recommended task

Do not claim something works unless it was actually run and verified.

Do not turn PROJECT_[STATE.md](http://STATE.md) into a transcript or chronological diary.

Do not modify `AGENTS.md` unless the user explicitly asks to change

project instructions.

---

## Research integrity

Never introduce look-ahead bias.

Specifically:

- maintain chronological train / validation / test ordering

- fit preprocessing and scalers on training data only

- never use future observations in feature construction

- never tune hyperparameters on the locked test set

- never inspect confirmatory-test results during model development

- never silently remove failed models or configurations

- never choose losses or proxies after seeing which favors a model

- use explicit seeds where randomness is involved

If a requested implementation risks leakage, stop and explain why.

---

## Covariance-specific requirements

The forecast target is a conditional covariance matrix.

For every covariance forecast:

- verify matrix dimensions

- verify symmetry

- monitor eigenvalues

- monitor positive definiteness where required

- monitor condition number

- never silently repair an invalid matrix

If a numerical repair is required:

1. record the failure

2. record the repair method

3. preserve diagnostics so the frequency of repairs can be reported

Realized covariance is a noisy proxy for latent covariance, not ground truth.

Keep separate:

- training objective

- statistical evaluation loss

- economic evaluation criterion

---

## Initial benchmark scope

Tier-1 models:

- Random-walk realized covariance

- EWMA

- HAR-DRD

- HARQ-DRD

- Ledoit-Wolf linear shrinkage

- Ledoit-Wolf nonlinear shrinkage

- DCC

- DCC-NL

- Ridge-DRD

Primary evaluation includes:

- multivariate QLIKE / Stein loss

- Frobenius loss

- variance/correlation decomposition

- global-minimum-variance portfolio evaluation

- Diebold-Mariano with HAC standard errors

- Model Confidence Set

- Giacomini-White

- Mincer-Zarnowitz diagnostics

Do not replace these with easier approximations without discussing it first.

---

## Implementation workflow

For each new method:

1. State what is being implemented.

2. State the mathematical inputs and outputs.

3. Identify leakage risks.

4. Identify numerical risks.

5. Implement the smallest correct version.

6. Add synthetic or unit tests.

7. Run the tests.

8. Integrate it into the benchmark only after validation.

9. Update `docs/PROJECT_STATE.md`.

Do not generate large amounts of code before verifying the mathematical

specification.

---

## Code quality

Do not write sloppy code.

Do not break the repository or package structure to take a shortcut.

The layout under `src/covharness/` is the implementation home. New code goes in

the existing package that owns it. Do not dump modules at the repo root, in

`scripts/`, or in `notebooks/` when they belong in the package. Do not flatten,

rename, or invent a parallel folder tree.

If a change would require moving that layout, stop and explain why before doing

it.

Prefer:

- the smallest correct implementation that fits the current tree

- small modules

- explicit interfaces

- readable numerical code

- descriptive variable names

- type hints where useful

- docstrings with dimensions and mathematical meaning

- centralized configuration

- reproducible seeds

- tests for important mathematical routines

For nontrivial matrix code, include expected dimensions in comments or

docstrings.

Avoid clever vectorization when it makes the mathematics difficult to audit.

A passing linter or a clever one-file rewrite is not a substitute for a clean

module boundary.

---

## Testing

For important estimators, models, losses, and inference procedures, test

where applicable:

- expected shape

- symmetry

- PSD / PD behavior

- known special cases

- no-lookahead indexing

- deterministic behavior under fixed seeds

- comparison against a trusted implementation

- synthetic data where the true covariance or null hypothesis is known

A passing import is not sufficient validation.

---

## Documentation

Public-facing documentation must be concise and professional.

Avoid:

- hype

- motivational prose

- recruiter-facing commentary

- claims unsupported by completed experiments

- AI-style meta commentary

Clearly label work as:

- implemented

- validated

- experimental

- planned

Never claim that deep learning beats econometrics unless the locked

evaluation supports that conclusion.