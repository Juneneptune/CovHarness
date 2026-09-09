# Project state

Last updated: 2026-09-09.

Project conda environment is `covharness` (Python 3.11). Recreate with `conda env create -f environment.yml` from the repository root.

## Current milestone

Literature-grounded high-frequency NBBO quote cleaning, placed before previous-tick synchronization. Realized-kernel estimation is not implemented. The long historical extract is not solved.

## Completed

- Previous-tick synchronization of irregular ticks onto a caller-supplied grid.
- Synchronized log returns $r_j = \log(P_j/P_{j-1})$.
- Unscaled daily realized covariance $\mathrm{RCov} = R^{\top} R$ from a complete synchronized return matrix.
- Subsampled proxy $\mathrm{RCov}^{SS} = (1/5)\sum_{s=0}^{4} \mathrm{RCov}^{(s)}$, where each offset $s$ takes every fifth synchronized one-minute price beginning at $s$, then reuses the existing log-return and Gram-matrix functions.
- High-frequency NBBO quote-cleaning layer adapted from Barndorff-Nielsen, Hansen, Lunde, and Shephard (2011), Section 5.1, following the 2009 trade-and-quote cleaning paper. Stages P1, P2, Q1, Q2, Q3, and Q4, with per-stage diagnostics. P3 single-exchange retention is not applied because the input is consolidated NBBO.
- Integration of cleaning before previous-tick synchronization. Midquotes are formed after the quote filters. A synthetic JPM-style fixture confirms that raw previous-tick sampling can capture an isolated 22.6-style quote while cleaned previous-tick sampling does not.

The forced 09:30 and 16:00 endpoint convention discussed for shifted grids is **not** implemented. Offsets are regular `iloc[s::5]` slices of the supplied panel.

## Files that own the implementation

- `src/covharness/data/quotes.py`
- `src/covharness/data/synchronization.py`
- `src/covharness/data/returns.py`
- `src/covharness/realized/rcov.py`
- `src/covharness/realized/daily.py`
- `src/covharness/realized/subsampled.py`
- `src/covharness/simulation/intraday.py`
- `tests/unit/test_quote_cleaning.py`
- `tests/unit/test_synchronization.py`
- `tests/unit/test_rcov.py`
- `tests/unit/test_daily_rcov.py`
- `tests/unit/test_subsampled_rcov.py`
- `notebooks/data_cleaning_example.ipynb` (empirical JPM case study, not production code)

## Tests run

Command `pytest -q` on 2026-09-09.

**48 passed** in 0.56s.

`tests/unit/test_quote_cleaning.py` was also run verbosely. **13 passed**, covering Q1 median collapse, Q2 negative-spread removal, locked-quote retention, Q3 wide-spread removal, Q4 isolated-midpoint removal, Q4 retention of a stable midpoint, Q4 zero local dispersion, stock-day neighborhood isolation, chronological ordering, cleaning-before-synchronization integration, a JPM-style previous-tick regression fixture, P1 regular-hours filtering, and P2 positive bid/ask filtering.

## Methodological decisions already in code

- RCov is the unscaled Gram matrix of synchronized log returns.
- Missing returns abort the day. Pairwise deletion is rejected.
- Subsampling averages five daily Gram matrices. It does not average returns and does not divide by $M$. Overlapping grids are not treated as independent.
- Quote cleaning is an adaptation of BN-HLS (2011) for consolidated NBBO. The paper's P3 single-exchange rule is not reproduced.
- Q2 removes crossed quotes only. Locked (zero-spread) quotes are retained.
- Q4 uses mean absolute deviation from a centered local median of 50 neighbors, not the median-absolute-deviation statistic. Incomplete edge windows are retained by default.
- Q4 is an ex-post measurement filter. Neighborhoods do not cross stock or day boundaries and are not used as forecasting features.

## Known problems or limitations

- Opening and overnight treatment remain unresolved.
- No regular NYSE calendar grid builder. The caller supplies the 1-minute panel.
- No Epps curve. No per-day eigenvalue or condition-number log across a dataset.
- No sale-condition cleaning of the trade tape. No production rerun of the five-asset TAQ pilot through `clean_nbbo_quotes` has been recorded in this checkpoint.
- Early-close calendars are not encoded in P1. The implemented window is 09:30–16:00 inclusive on the timestamps supplied by the caller.
- WRDS access is a user-reported fact from 2026-09-08. Account permissions, subscribed libraries, and table schemas have **not** been verified here as a completed extract.
- Shifted-grid coverage of a common 09:30–16:00 open-to-close window is a proposed implementation choice, not current code.
- Realized kernels are not implemented. The long historical data problem is not solved.

## Next recommended task

Rerun the five-asset real-data pilot through the production cleaning code. Verify synchronized returns and matrix diagnostics. Then compute the first cleaned real-data realized covariance.

Do not start HAR, DCC, shrinkage, losses, inference, portfolios, LSTM-BEKK, or realized-kernel estimation while that cleaned real-data proxy remains uncomputed.
