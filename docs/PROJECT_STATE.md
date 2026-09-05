# Project state

Last updated: 2026-09-04.

## Current milestone

Repository scaffold only. No estimators, models, losses, or inference code yet.

## Completed

- Created the package layout under `src/covharness/` with empty subpackages.
- Added `pyproject.toml`, `.gitignore`, and this state file.
- Expanded `.gitignore` for raw data, large intermediates, checkpoints, run logs, secrets, and LaTeX debris. `.python-version` is tracked.
- Added a hard cleanliness / structure-preservation rule to `AGENTS.md`.
- Moved `project_ledger.md` to `docs/project_ledger.md`.
- Updated the README repository layout to match the tree.

## Files created or moved

- `pyproject.toml`, `.gitignore`, `docs/PROJECT_STATE.md`, `AGENTS.md`
- `src/covharness/` and subpackages (`data`, `realized`, `models`, `features`, `losses`, `inference`, `portfolio`, `protocol`, `diagnostics`, `simulation`, `utils`)
- empty placeholders: `configs/`, `data/`, `scripts/`, `tests/`, `experiments/`, `results/`, `notebooks/`, `paper/`
- `docs/project_ledger.md` (moved from repo root)

## Tests run

None. There is no implementation to test.

## Methodological decisions

None this session. Design source of truth remains the consolidated v4 plan in `README.md` (the document also referred to as `design_v4_demo_plan.md` in `AGENTS.md`).

## Known problems or limitations

- No data source has been acquired yet (WRDS / Databento / simulation fallback still open).
- `AGENTS.md` still points at a root-level `design_v4_demo_plan.md`; that content currently lives in `README.md`.
- Package submodules are empty placeholders.

## Next recommended task

Day 0 / Day 1 data work: settle the data source, then implement cleaning, synchronization, and the two RCov proxies with PD/conditioning logs and Epps validation.
