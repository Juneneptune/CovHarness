# Project state

Last updated: 2026-09-06.

## Current milestone

Day 1: primary daily realized-covariance composition (`RCov_5min` pipeline).

Status: **implemented** and **validated**. Grid frequency is caller-supplied; tests used a 5-minute-like grid.

## Completed

- Unscaled synchronized RCov: `realized_covariance(R) = R' R`.
- Previous-tick sync onto a caller-supplied grid; synchronized log returns `r_j = log(P_j / P_{j-1})`.
- `daily_realized_covariance(prices)` composes those two functions: prices -> log returns `R` `(M, N)` -> `R' R` `(N, N)`.
- Complete-return-matrix requirement: if any return is missing, raise. No pairwise deletion, no `/M`, annualization, shrinkage, or PD repair.

## Files created or changed

- `src/covharness/realized/daily.py`
- `tests/unit/test_daily_rcov.py`
- `docs/PROJECT_STATE.md`

Existing pieces reused, not duplicated: `src/covharness/data/returns.py`, `src/covharness/realized/rcov.py`.

## Tests run

```
pytest tests/unit/test_daily_rcov.py -v
```

**8 passed** in 0.29s.

```
pytest tests/unit/test_synchronization.py -v
```

**11 passed** in 0.30s.

```
pytest tests/unit/test_rcov.py -v
```

**5 passed** in 0.16s.

```
pytest -v
```

**24 passed** in 0.36s.

## Methodological decisions

- Daily RCov is the unscaled Gram matrix of synchronized log returns for one day.
- Missing returns abort the day; they are not pairwise-deleted (that would mix intervals and can destroy PSD).
- Opening/overnight convention is **not** decided here. The function starts from an already valid synchronized price grid.

## Known problems or limitations

- No regular grid builder yet; the caller supplies timestamps.
- Opening price / overnight return treatment is unresolved (design options a/b/c in the v4 plan).
- No subsampled proxy, Epps diagnostics, sale-condition cleaning, or real data.
- `AGENTS.md` still points at a root-level `design_v4_demo_plan.md`; that content currently lives in `README.md`.

## Next recommended task

Implement the subsampled realized-covariance proxy (`RCov_5min_ss`: average of offset grids).
