# Binance open-data production panel

Panel date 2026-09-21. This document freezes the branch-specific segmented UTC calendar and records the production five-asset measurement arrays. It does not modify `TemporalProtocol`. It does not change U.S.-equity TAQ mathematics. VALIDATION and SCREEN fitting are recorded in `docs/BINANCE_OPEN_DATA_PROTOCOL.md`. CONFIRM remains locked. Final `PREREGISTRATION.md` remains absent.

Blocks 1–3 and 4A-1 through 4A-10 remain closed.

---

## 1. Hard temporal break

The constraint is that 2023-03-24 is not a statistical day. The choice is a **hard temporal break**, not a packed complete-day index.

Packed lags were rejected. On a packed production index, 2023-03-23 and 2023-03-25 would become adjacent rows. That packing is not an equity weekend. The elapsed time from 23:59 UTC on 23 March to 23:59 UTC on 25 March is 48 hours. HAR daily, weekly, and monthly lags, EWMA recursion, DCC state, and LSTM-BEKK recurrent state would then treat a two-calendar-day gap as one statistical step. We do not alter those recursions to skip the halt.

2023-03-24 remains on the calendar as a boundary-price day. Its valid 23:59 close is the frozen anchor for the 2023-03-25 measurement. It is not a production statistical date, not an origin, and not a target.

The storage representation is one aligned array plus per-date segment labels. Downstream code must not treat the last VALIDATION row as the previous statistical day of the first HISTORY_B row.

---

## 2. Frozen segmented calendar

Independent UTC inclusive counts were verified before construction. Advertised lengths match the actual dates.

DEVELOPMENT SEGMENT

HISTORY_A. 2021-11-09 through 2022-07-16 inclusive. 250 UTC dates.

VALIDATION. 2022-07-17 through 2023-03-23 inclusive. 250 UTC dates.

EXCLUDED HALT. 2023-03-24.

CONFIRMATORY SEGMENT

HISTORY_B. 2023-03-25 through 2023-11-29 inclusive. 250 UTC dates.

SCREEN. 2023-11-30 through 2025-04-12 inclusive. 500 UTC dates. Includes the 2024 leap day.

CONFIRM. 2025-04-13 through 2026-08-25 inclusive. 500 UTC dates.

Dates 2026-08-26 through 2026-08-31 are outside this benchmark sample and were not used.

The reason for HISTORY_B is methodological. No one-day forecast crosses 2023-03-24. No HAR lag, EWMA state, DCC state, or LSTM-BEKK recurrent state crosses the break. SCREEN and CONFIRM begin only after 250 new consecutive complete post-halt statistical days.

This calendar is branch-specific. The generic `TemporalProtocol` implementation was not modified.

Verified production length is 1750 retained statistical dates. The halt is omitted. Asset order is frozen as BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, ADAUSDT.

Implementation. `src/covharness/data/binance_calendar.py`.

---

## 3. Additional official archives

The previous coverage cache began at 2022-01. HISTORY_A starts on 2021-11-09 and needs the 2021-11-08 23:59 boundary anchor.

We downloaded only the official November and December 2021 monthly 1m ZIP and CHECKSUM files for the five frozen names from `https://data.binance.vision`. Every checksum verified before parse. Existing 2022-01 through 2026-08 archives were not redownloaded.

2021-11-08 23:59 UTC exists on all five names. HISTORY_A dates in 2021 have every required five-minute endpoint.

Raw ZIPs remain gitignored under `data/cache/open_data/binance/spot/monthly/klines/1m/`. They are not committed.

---

## 4. Production measurement

A distinct path consumes official 1-minute kline **close** prices. It does not call quote, midpoint, NBBO, or previous-tick TAQ classes.

For retained date $t$, 288 five-minute log returns use prior-day 23:59, then 00:04 through 23:59. The previous UTC day need not be a retained statistical day. Missing required endpoints are rejected. There is no forward fill.

```math
\mathrm{RCov}_{t}
=
\sum_{j=1}^{288}
r_{t,j}\,r_{t,j}^{\top},
\qquad
\mathrm{RQ}_{i,t}
=
\frac{288}{3}
\sum_{j=1}^{288}
r_{i,t,j}^{4},
\qquad
R_{i,t}
=
\log P_{i,t,23{:}59}
-
\log P_{i,t-1,23{:}59}.
```

No annualization, normalization, winsorization, clipping, nearest-PD, eigenvalue floor, or jitter. An invalid RCov stops construction for that date. No production date failed.

Implementation. `src/covharness/data/binance_klines.py` and `src/covharness/realized/binance_panel.py`.

---

## 5. Artifact

Path `data/processed/binance_five_asset_panel.npz`. Gitignored under the existing `data/**/*.npz` rule. The policy was not changed.

SHA-256 `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`. Size 359563 bytes.

Contents. `dates`, `assets`, `segments`, `daily_returns` with shape `(1750, 5)`, `rcov_5min` with shape `(1750, 5, 5)`, `rq_per_asset` with shape `(1750, 5)`, `n_intervals=288`, `halt_date`, and `measurement`. Raw minute bars are omitted.

Metadata sidecar `data/processed/binance_five_asset_panel.json`. SHA-256 `38ed1d4843ea6138d7f87fc0405b668b1ee2fd7403d4cddfba0ca49160737394`. It stores the NPZ hash and 290 verified monthly archive checksums (58 months times 5 names, 2021-11 through 2026-08). JSON remains committable under the existing `!data/**/*.json` exception.

Descriptive validation `results/binance_five_asset_panel_validation.json`.

---

## 6. Descriptive validation

Focused unit tests `tests/unit/test_binance_measurement.py`, 13 passed.

Built-panel checks. $T=1750$, $N=5$. Range 2021-11-09 through 2026-08-25. Segment counts 250, 250, 250, 500, 500. Unique dates 1750. Halt absent. No date after 2026-08-25. Every SCREEN date after HISTORY_B. Every CONFIRM date after SCREEN. Development dates all precede confirmatory dates. Calendar gap between VALIDATION and HISTORY_B is two days.

RCov. Nonfinite 0. Symmetry failures 0. PSD failures 0. Minimum eigenvalue $3.51325218252141\times 10^{-6}$. Maximum eigenvalue $0.9172798620889434$. Numerical rank 5 on all 1750 dates. Condition-number min / median / max $12.66$ / $52.39$ / $605.18$.

RQ. Nonfinite 0. Minimum $1.1168563091513678\times 10^{-10}$. Median $1.5940250809368411\times 10^{-6}$. Maximum $17.41941235552369$. All nonnegative.

Daily returns. Nonfinite 0. Minimum $-0.28001695783272296$. Maximum $0.5432092637962491$.

Invalid production dates. None.

These are measurement diagnostics. They are not forecast comparisons.

---

## 7. Remaining first-stage blockers

The segmented calendar and production arrays exist. The first-stage protocol, roster, VALIDATION grids, and SCREEN representatives remain frozen in `docs/BINANCE_OPEN_DATA_PROTOCOL.md`. VALIDATION and SCREEN fitting completed on 2026-09-21. SCREEN is selection data, not confirmation. Artifacts are listed in that protocol document.

- SCREEN completed. CONFIRM has not been authorized.
- CONFIRM remains locked and has not been run.
- Subsampled five-minute robustness is planned, not implemented.
- U.S.-equity DATA GATE remains closed.
- Final `PREREGISTRATION.md` remains absent.
- LSTM-BEKK-RC, GHAR, and the graph-neural slot remain outside this first-stage core.

---

## 8. DATA-GATE component status

| Component | Status |
| --- | --- |
| Binance raw source/access | PASS |
| Binance history length | PASS |
| Binance cross-sectional synchronization | PASS |
| Binance universe identity | PASS |
| Binance measurement definition | PASS |
| Binance temporal segmentation | PASS |
| Binance production daily-return panel | PASS |
| Binance production RCov panel | PASS |
| Binance production RQ panel | PASS |
| Binance forecasting configuration/tuning rules | PASS |
| Binance empirical model fitting | VALIDATION COMPLETE. SCREEN COMPLETE. CONFIRM NOT RUN |
| SCREEN | COMPLETE / SELECTION DATA ONLY |
| CONFIRM | LOCKED / NOT RUN |
| U.S.-equity DATA GATE | UNCHANGED / CLOSED |
