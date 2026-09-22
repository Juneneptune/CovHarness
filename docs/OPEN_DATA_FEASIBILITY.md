# Open-data feasibility

Audit date 2026-09-21. This note records a bounded public-data investigation. It does not open the U.S.-equity DATA GATE. It does not change the confirmatory protocol. Blocks 1–3 and 4A-1 through 4A-10 remain closed. CONFIRM remains locked. Final `PREREGISTRATION.md` remains absent.

Later status. Binance VALIDATION and SCREEN fitting proceeded after this audit under `docs/BINANCE_OPEN_DATA_PROTOCOL.md`. The counts and source checks below remain the feasibility record. They are not SCREEN results.

The investigation answers two questions with evidence. First, whether the official M6 price file can support a daily-return auxiliary experiment for models that already consume daily returns. Second, whether official Binance public historical klines can support a separate free high-frequency multivariate covariance experiment with at least approximately 1,500 common usable UTC dates for a small universe.

Counts below are labeled exact, listing-derived, or sample-only. Monthly ZIP presence is not usable-day coverage. A later long-panel 1m audit for a frozen five-name coverage universe is recorded in section 8.

---

## 1. Scope and constraints

We used only official public sources. The M6 file is `assets_m6.csv` from `https://github.com/Mcompetitions/M6-methods`. Binance files are from `https://github.com/binance/binance-public-data`, `https://data.binance.vision`, and the matching S3 listing on `data.binance.vision`. No WRDS connection. No paid API. No yfinance. No unofficial scrape. No API key.

Raw downloads sit under gitignored `data/cache/open_data/`. They are not committed.

The 2026-09-21 feasibility and coverage audits did not construct production RCov or RQ panels. Those arrays now exist under the frozen segmented calendar. See `docs/BINANCE_OPEN_DATA_PANEL.md`. We did not call the equity quote/midquote estimator. We did not fit RW, EWMA, HAR, HARQ, LW, DCC, Ridge, XGBoost, LSTM-BEKK, GHAR, or any other model.

Selected 5-asset and 10-asset groups are labeled FEASIBILITY SUBSETS. Selection used coverage and deterministic tie-breaking only. They are not adopted universes.

---

## 2. M6 official data audit

### Source

Local path `data/cache/open_data/m6/assets_m6.csv`.

| Field | Value | Count basis |
| --- | --- | --- |
| Source URL | `https://raw.githubusercontent.com/Mcompetitions/M6-methods/main/assets_m6.csv` | Exact |
| Retrieval date (UTC) | 2026-09-21T02:38:13Z | Exact, file mtime |
| SHA-256 | `48c67aa0976ae63de3a6d7a42228ef374d39221734a433c9391572d01e85b20a` | Exact |
| File size | 590752 bytes | Exact |
| Columns | `symbol`, `date`, `price` | Exact |
| dtypes before date parse | object, object, float64 | Exact |
| Date strings | `YYYY/MM/DD` | Exact |
| Rows | 26446 | Exact |
| Unique symbols | 100 | Exact, verified from the file |
| Verified 100 M6 assets | True | Exact |
| Min date | 2022-01-31 | Exact |
| Max date | 2023-02-17 | Exact |
| Union of observed dates | 273 | Exact |
| Duplicate symbol/date rows | 0 | Exact |
| Nonfinite prices | 0 | Exact |
| Nonpositive prices | 0 | Exact |

The file contains 100 unique symbols. That count is taken from the downloaded table, not from a README claim.

### Observations per symbol

| Statistic | Value |
| --- | --- |
| Minimum | 209 (`DRE`) |
| Maximum | 267 (`SEGA.L`) |
| Median | 265 |
| Symbols with 265 dates | 98 |
| Symbols with 209 dates | 1 (`DRE`) |
| Symbols with 267 dates | 1 (`SEGA.L`) |

`DRE` ends on 2022-11-28 and is missing 64 union dates. Full per-symbol missingness is in `results/open_data_feasibility.json`.

No symbol is observed on every union date. Dates complete for all 100 assets therefore equal the 100-asset intersection, not the union.

### Common-date coverage

Log-return length uses successive dates in the intersection. That is the usable daily-return length after one lagged price. Outer products of those daily returns are not the existing high-frequency realized-covariance target.

| Universe | Union dates | Intersection dates | Successive log returns | Min intersection date | Max intersection date | ≥1,500 dates | ≥1,250 dates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All 100 assets | 273 | 201 | 200 | 2022-01-31 | 2022-11-28 | No | No |
| FEASIBILITY SUBSET 5, greedy coverage | 272 | 260 | 259 | 2022-01-31 | 2023-02-17 | No | No |
| FEASIBILITY SUBSET 10, greedy coverage | 272 | 260 | 259 | 2022-01-31 | 2023-02-17 | No | No |
| FEASIBILITY SUBSET 5, greedy max intersection | 267 | 265 | 264 | 2022-01-31 | 2023-02-17 | No | No |
| FEASIBILITY SUBSET 10, greedy max intersection | 269 | 263 | 262 | 2022-01-31 | 2023-02-17 | No | No |

Greedy coverage ranks by observation count, then alphabetical symbol. The 5-name prefix is `SEGA.L`, `ABBV`, `ACN`, `AEP`, `AIZ`. Greedy maximum intersection adds the name that preserves the largest remaining intersection, then coverage, then alphabetical order. The 5-name set is `SEGA.L`, `HIGH.L`, `IEAA.L`, `IEVL.L`, `IUMO.L`. The 10-name set adds `IUVL.L`, `JPEA.L`, `MVEU.L`, `SPMV.L`, `IEFM.L`.

Longest calendar-consecutive common streak is 5 days on every subset examined (a Monday–Friday block). Longest weekend-tolerant trading-day streak is 47 dates for all 100 assets and 69 dates for the 5-name maximum-intersection subset. Neither construction reaches 1,250 common dates.

No 5-asset or 10-asset availability-only subset of this file can form 1,500 common daily observations. The entire sample spans 383 calendar days from 2022-01-31 through 2023-02-17.

### Model API compatibility

Under the present `FitInput` contract, an M6 daily-return panel could be consumed by Ledoit–Wolf linear, Ledoit–Wolf nonlinear, DCC, DCC-NL, and LSTM-BEKK. Those models take a `(T, N)` daily-return window.

Random-walk RCov, EWMA, HAR-DRD, Ridge-DRD, and XGBoost-DRD cannot, because they require a `(T, N, N)` realized-covariance history. HARQ-DRD additionally requires the aligned `(T, N)` per-asset realized-quarticity window.

Even for the daily-return models, the harness evaluation target remains the open-to-close realized-covariance proxy. M6 cannot supply that proxy. Scoring those models against outer products of M6 daily returns would be a different experiment, not the existing high-frequency protocol.

M6 can support only a short daily-return auxiliary. With at most 265 common dates, rolling `m=250` leaves at most 15 one-day-ahead origins. That is far below VALIDATION/SCREEN/CONFIRM allocation, which requires `T-m ≥ 1000`.

---

## 3. Binance archive-span audit

### Official infrastructure

Monthly spot kline ZIP URL

`https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY}-{MM}.zip`

Matching `.CHECKSUM` files use the same stem. Object inventories were read from the official S3 `ListBucket` prefix listing on `data.binance.vision`, not from a first/last binary search. An initial bounded HEAD search (335 requests) misstated some endpoints when used as a binary search, including BTCUSDT 1m earliest month. The S3 listings replace those HEAD first/last claims. Follow-up HEAD checks confirmed that BTCUSDT 1m objects exist in 2017-08, 2018-01, and 2019-09, consistent with the listing.

The official `binance-public-data` README states that Spot timestamps from 2025-01-01 onward are microseconds. Timestamp scale in downloaded files was inferred from the integer open-time values, not assumed.

The candidate USDT pairs are BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, LTCUSDT, BCHUSDT, XLMUSDT, TRXUSDT, ETCUSDT, LINKUSDT, and EOSUSDT. That list is not a frozen empirical universe.

`https://data-api.binance.vision` was not required and was not accessed.

### Per-symbol monthly ZIP presence (1m and 5m)

For every candidate, 1m and 5m earliest month, latest month, and interior-hole count agree. Interior missing months between first and last listed ZIP are 0.

| Symbol | Earliest month | Latest month | Monthly ZIPs | Inclusive calendar days | Mean 1m ZIP bytes |
| --- | --- | --- | --- | --- | --- |
| BTCUSDT | 2017-08 | 2026-08 | 109 | 3318 | 2136703 |
| ETHUSDT | 2017-08 | 2026-08 | 109 | 3318 | 1990191 |
| BNBUSDT | 2017-11 | 2026-08 | 106 | 3226 | 1734143 |
| LTCUSDT | 2017-12 | 2026-08 | 105 | 3196 | 1715425 |
| ADAUSDT | 2018-04 | 2026-08 | 101 | 3075 | 1648394 |
| XRPUSDT | 2018-05 | 2026-08 | 100 | 3045 | 1726726 |
| XLMUSDT | 2018-05 | 2026-08 | 100 | 3045 | 1443914 |
| TRXUSDT | 2018-06 | 2026-08 | 99 | 3014 | 1656764 |
| ETCUSDT | 2018-06 | 2026-08 | 99 | 3014 | 1423016 |
| LINKUSDT | 2019-01 | 2026-08 | 92 | 2800 | 1592042 |
| EOSUSDT | 2018-05 | 2025-05 | 85 | 2588 | 1538317 |
| BCHUSDT | 2019-11 | 2026-08 | 82 | 2496 | 1521858 |

EOSUSDT monthly archives end in 2025-05. That listing end is treated as a coverage fact, not as a trading-halt diagnosis.

### Common archive spans

These figures count months for which a ZIP object exists. They are not verified usable UTC days.

| FEASIBILITY SUBSET | Interval | Common start | Common end | Common months | Inclusive days | Longest consecutive days | ≥1,500 archive-span days | Interior holes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 names BTC, ETH, BNB, LTC, ADA | 1m | 2018-04 | 2026-08 | 101 | 3075 | 3075 | Yes | None |
| Same 5 names | 5m | 2018-04 | 2026-08 | 101 | 3075 | 3075 | Yes | None |
| 10 names, drop EOS and BCH | 1m | 2019-01 | 2026-08 | 92 | 2800 | 2800 | Yes | None |
| Same 10 names | 5m | 2019-01 | 2026-08 | 92 | 2800 | 2800 | Yes | None |
| All 12 candidates | 1m | 2019-11 | 2025-05 | 67 | 2039 | 2039 | Yes | None |
| All 12 candidates | 5m | 2019-11 | 2025-05 | 67 | 2039 | 2039 | Yes | None |

The 10-name group is BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, ADAUSDT, XRPUSDT, XLMUSDT, TRXUSDT, ETCUSDT, and LINKUSDT. EOSUSDT is omitted because its archive ends in 2025-05. BCHUSDT is omitted because it is the latest listed. Selection is coverage-only.

Archive-span feasibility is yes for at least 5 assets and for at least 10 assets, on both 1m and 5m, with a common listed span exceeding 1,500 UTC calendar days. Actual verified usable-day coverage is not claimed from ZIP presence.

---

## 4. Bounded Binance content probe

Hard cap 150 MiB. Bytes actually downloaded in the listing-plus-sample script 8,640,819 (about 8.2 MiB). Additional earlier HEAD responses are not counted in that total.

Preferred sample is the complete month 2025-02 for BTCUSDT, ETHUSDT, and BNBUSDT, both 1m and 5m, with official `.CHECKSUM` files. A cheap millisecond-era control is BTCUSDT 5m for 2024-12.

Every downloaded ZIP matched its official SHA-256 checksum.

### 2025-02 (microsecond era)

| File | ZIP bytes | Rows | Inferred unit | UTC open-time range | Complete UTC days | Duplicates | Non-monotone | Nonpositive/nonfinite OHLC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT 1m | 1993970 | 40320 | microseconds | 2025-02-01 00:00 through 2025-02-28 23:59 | 28 / 28 | 0 | 0 | 0 |
| ETHUSDT 1m | 1930330 | 40320 | microseconds | same | 28 / 28 | 0 | 0 | 0 |
| BNBUSDT 1m | 1718092 | 40320 | microseconds | same | 28 / 28 | 0 | 0 | 0 |
| BTCUSDT 5m | 428947 | 8064 | microseconds | 2025-02-01 00:00 through 2025-02-28 23:55 | 28 / 28 | 0 | 0 | 0 |
| ETHUSDT 5m | 411687 | 8064 | microseconds | same | 28 / 28 | 0 | 0 | 0 |
| BNBUSDT 5m | 370748 | 8064 | microseconds | same | 28 / 28 | 0 | 0 | 0 |

February 2025 has 28 UTC days. Expected complete coverage is 28 × 1440 = 40320 one-minute bars and 28 × 288 = 8064 five-minute bars. All six files match those counts. Every UTC day starts at midnight and ends on the last in-grid bar. Zero-volume rows are 0 in this sample. UTC boundaries are represented cleanly in the downloaded month.

### 2024-12 control (millisecond era)

BTCUSDT 5m, 474874 ZIP bytes, 8928 rows, inferred milliseconds, 2024-12-01 00:00 through 2024-12-31 23:55 UTC, 31 / 31 complete days, checksum verified. December has 31 × 288 = 8928 five-minute bars. Timestamp scale therefore changes across 2025-01-01 as documented, and must be detected rather than hard-coded.

### 1m versus 5m close diagnostic (2025-02)

For each of BTCUSDT, ETHUSDT, and BNBUSDT, every 5m open exists on the 1m grid, and every 1m open on a 5-minute boundary exists as a 5m open. The 5m close equals the 1m close of the bar that opens four minutes later for all 8064 rows, with maximum absolute difference 0. This is a schema diagnostic. It is not a production estimator.

### Synchronized log-close diagnostic

Inner-joined 1m log closes for the three names in 2025-02 yield 40320 synchronized rows, none nonfinite, from 2025-02-01 00:00 UTC through 2025-02-28 23:59 UTC. That construction uses kline closes on a regular UTC grid. It is not the equity TAQ midquote previous-tick estimator and is not a production RCov panel.

---

## 5. Size and compute estimates

Estimates use official S3 `Size` fields on monthly ZIP objects for the coverage-only 5-name and 10-name subsets over the latest consecutive listed window with at least 1,500 calendar days (2022-07 through 2026-08, 1523 days), then scale linearly to 1,500 days. Parsed size uses the mean uncompressed-CSV / ZIP ratio from the downloaded sample of the same interval. No bulk download was started.

| Request | Compressed ZIP (scaled to 1,500 days) | Approximate parsed CSV | Rows if every UTC day is complete |
| --- | --- | --- | --- |
| 5 assets × 1,500 days × 1m | 451,533,935 bytes (430.6 MiB) | 1,485,542,683 bytes (1.42 GiB) | 2,160,000 per asset, 10,800,000 total |
| 10 assets × 1,500 days × 1m | 821,295,969 bytes (783.2 MiB) | 2,702,056,529 bytes (2.58 GiB) | 2,160,000 per asset, 21,600,000 total |
| 10 assets × 1,500 days × 5m | 182,667,470 bytes (174.2 MiB) | 563,835,086 bytes (537.7 MiB) | 432,000 per asset, 4,320,000 total |

A complete UTC day on this 24-hour market is 1,440 one-minute bars and 288 five-minute bars. That grid was verified on every day of the 2025-02 sample for BTC, ETH, and BNB, and on every day of the 2024-12 BTCUSDT 5m control. It is not yet verified across 1,500 days.

If a UTC day has 288 five-minute closes, within-day successive log returns are 287. Including the midnight-crossing return from the previous day's last close yields 288 returns, matching 288 five-minute intervals. Expected synchronized 1m observations per complete UTC day are 1,440.

---

## 6. Methodological compatibility

We do not change methodology in this task. Binance is not identical to the existing equity measurement path. The differences do not by themselves make a later experiment invalid. They are not irrelevant either. Until a later protocol decision states otherwise, a Binance panel is a distinct open-data empirical experiment, not a drop-in substitute for Daily TAQ midquotes.

Trade/kline price versus quote midpoint. Official spot klines are OHLCV aggregations of trades. The close is a last trade in the interval, not a cleaned bid/ask midpoint. Block 1 P2–Q4 operate on spreads and cannot run on a single close. Venue P3 and share-class suffix identity do not exist in this market. Using kline closes inside the equity previous-tick class would disguise the measurement.

24/7 UTC calendar versus the 09:30–16:00 equity session. A complete crypto UTC day has 1,440 one-minute bars. The equity session grid is much shorter and exchange-local. Overnight equity returns are out of scope for the current statistical RCov and are reserved for a later GMV channel. A 24-hour market has no equity-style overnight interval. Midnight is a calendar seam, not an exchange close.

Symbol listing and common-history selection. Pair listing dates differ. EOSUSDT monthly archives end in 2025-05. A common-history rule will drop names or shorten the panel. That selection must remain coverage-only until a universe is frozen. Common USDT quoting removes FX conversion among the candidates, and it also concentrates quote-currency risk. It is not the CRSP/TAQ identity rule.

Missing-bar treatment. The sample month has no missing bars. Other months may omit quiet minutes or halt intervals. The equity estimator drops incomplete synchronized grid rows (`drop_incomplete_open`). A Binance branch needs its own missing-bar rule. Silent forward fill would be a different proxy.

One-minute observations and subsampling. Regular 1m closes can support a separate subsampling robustness construction, analogous in spirit to the implemented five-minute subsampled equity proxy, but not identical to previous-tick on irregular quotes. Direct 5m klines in the sample agree with every-fifth 1m close. That agreement may fail where bars are missing.

Realized quarticity. Per-asset RQ from synchronized intraday returns is feasible on a complete 5m (or subsampled 1m) grid using the existing formula $\mathrm{RQ}_i=(M/3)\sum_l r_{i,l}^4$. $M$ on a 24-hour day is not the equity-session $M$. HARQ can consume that panel only after a Binance measurement branch exists. The present TAQ-to-RQ path is not that branch.

Synchronized covariance. Regular timestamps make synchronization trivial when bars are complete. The daily Gram matrix $R^{\top}R$ of synchronized log returns remains well-defined. The resulting $S_t$ is still a noisy proxy for a latent covariance, now over a UTC day of trade-close returns rather than an equity session of midquote previous-tick returns. Reduced QLIKE and squared Frobenius could score such a proxy only after the measurement convention is declared separately.

---

## 7. Feasibility conclusions

M6. The official file is a clean 100-asset daily price panel from 2022-01-31 through 2023-02-17. It cannot support 1,250 or 1,500 common daily observations, for all names or for any coverage-only 5- or 10-name subset. It can support only a short daily-return auxiliary for LW, DCC, DCC-NL, and LSTM-BEKK. It cannot feed RCov or HARQ models. It cannot replace the high-frequency evaluation target.

Binance archive span. Official monthly 1m and 5m spot archives show a hole-free listed span longer than 1,500 calendar days for a coverage-only 5-asset subset and for a coverage-only 10-asset subset. The 2025-02 sample is schema-complete and checksum-verified. Timestamps are milliseconds through 2024-12 and microseconds from 2025-01.

Binance usable-day coverage. The 2026-09-21 long-panel 1m audit on BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, and ADAUSDT, months 2022-01 through 2026-08, found 1,703 five-way complete UTC dates out of 1,704 requested calendar dates. That meets the preferred raw calendar threshold of 1,500. The first Binance empirical universe is that five-name set. Measurement formulas are frozen in `docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md`. 2023-03-24 is excluded as a statistical day after a documented venue-wide halt. Packed complete-day lags across that hole are rejected. The production RCov, RQ, and daily-return panel is recorded in `docs/BINANCE_OPEN_DATA_PANEL.md`. The branch is distinct from U.S.-equity TAQ.

The U.S.-equity DATA GATE is unchanged. Local TAQ remains the 2009-02-13 five-name cache. Production WRDS SELECT remains unverified. Alpaca SIP remains unprobed.

---

## 8. Long-panel five-asset 1m coverage audit

Audit date 2026-09-21. Coverage-audit universe only. BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, ADAUSDT. Not the final empirical universe. Official monthly 1m klines from `https://data.binance.vision`, months 2022-01 through 2026-08 inclusive. UTC calendar 2022-01-01 through 2026-08-31 (1,704 dates). No model fitting. No RCov or RQ production. No forward fill.

### Download

| Field | Value |
| --- | --- |
| ZIP files requested | 280 |
| Checksums requested | 280 |
| Verified SHA-256 | 280 |
| Absent archives | 0 |
| Checksum mismatches | 0 |
| Compressed bytes | 512308203 (488.6 MiB) |
| Uncompressed CSV bytes inside ZIPs | 1772426179 (1.65 GiB), not retained as files |
| Local cache | `data/cache/open_data/binance/spot/monthly/klines/1m/` (491 MiB), gitignored |
| Download wall clock | 90.672 s |
| Parse and audit wall clock | 57.422 s |
| Rows per symbol | 2453680 |
| Total rows | 12268400 |

A complete 1-minute UTC day requires exactly 1,440 unique minute open times covering 00:00 through 23:59 UTC, no duplicates, no missing slots, no non-monotone open times, and finite strictly positive OHLC. Return, volume, and liquidity screens were not applied.

### Per-symbol complete dates

Each symbol has 1,703 complete dates and one incomplete date. Earliest complete date 2022-01-01. Latest complete date 2026-08-31. Longest complete-day streak 1,256 dates from 2023-03-25 through 2026-08-31. Duplicates 0. Invalid prices 0. Off-grid timestamps 0. Non-monotone timestamps 0.

The sole incomplete date is 2023-03-24. All five names are missing 80 of 1,440 minutes that UTC day. Parsed row counts equal $1703\times 1440+1360=2453680$ per symbol.

### Five-way intersection

| Field | Value |
| --- | --- |
| Requested UTC dates | 1704 |
| Five-way complete dates | 1703 |
| Earliest common complete date | 2022-01-01 |
| Latest common complete date | 2026-08-31 |
| Longest consecutive common complete streak | 1256 dates, 2023-03-25 through 2026-08-31 |
| Excluded dates | 1 (`2023-03-24`) |
| Exclusion reason | missing 80 minutes on every name |
| preferred_full_design (≥1,500) | Yes |
| minimum_confirmatory_design (≥1,250) | Yes |
| common complete dates minus 250 | 1453 |

The 1,453 figure is the maximum number of one-step targets under rolling $m=250$ before any VALIDATION/SCREEN/CONFIRM split. No split was allocated.

Per-date flags are in `results/binance_five_asset_calendar_complete_dates.csv.gz`. Machine-readable audit `results/binance_five_asset_calendar_audit.json`.

### Timestamp units

Every 2022-01 through 2024-12 monthly file is milliseconds. Every 2025-01 through 2026-08 monthly file is microseconds. No month disagrees with the official 2025-01-01 documentation cut. Open times were converted to UTC only after per-file unit detection.

### Structural synchronized-panel diagnostic

On all 1,703 common complete dates the five names share the identical 1,440-minute support. Inner joining loses no rows. All closes are finite and strictly positive. Temporary within-day 1m log returns are finite on all 1,703 dates (1,439 returns). Full 1,440-return construction, including the midnight cross from the prior 23:59 close, is finite on 1,701 dates. It is unavailable on 2022-01-01 (no prior date in the window) and on 2023-03-25 (prior date incomplete). Within-day 5m log returns (287) are finite on all 1,703 dates. The 288-return midnight-inclusive 5m construction is finite on the same 1,701 dates.

No covariance matrix was stored. The primary return convention is now frozen in `docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md` as the boundary-anchored 288 five-minute construction, not the 287-return within-day version.

---

## 9. Measurement specification freeze

Specification date 2026-09-21. Distinct open-data empirical benchmark. First universe BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, ADAUSDT, selected before fitting from coverage only. UTC statistical day $[00{:}00,24{:}00)$. Price is the official 1m kline close. Primary RCov uses 288 five-minute returns including the midnight-crossing interval from the previous 23:59 close. RQ uses $M=288$ on those same returns. Daily return is close-to-close between consecutive 23:59 UTC prices.

2023-03-24 forensics. All five names miss one identical contiguous block, 12:40 through 13:59 UTC (80 minutes). 00:00 and 23:59 exist. The first bar after the hole is 14:00 UTC. That resume matches Binance’s announced 14:00 UTC restoration after a matching-engine spot halt. The kline hole sits inside the announced 11:27–14:00 window and does not begin at 11:27. Recommended treatment is exclusion of the UTC day as origin and target, while retaining 23:59 as the 2023-03-25 boundary anchor. Option B (venue-halt previous-observation fill) is not adopted. Packed complete-day lags are rejected. The production calendar uses a hard break and a second 250-day HISTORY burn-in. See section 11.

Full text `docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md`. Forensics `results/binance_20230324_halt_forensics.json`.

---

## 10. Exact next step

Freeze Binance forecasting configuration and tuning rules, and implement a branch-specific rolling schedule that respects the hard 2023-03-24 break without modifying generic `TemporalProtocol`. Do not fit models. Do not run SCREEN or CONFIRM. Do not wire klines into the TAQ classes. Do not open the U.S.-equity DATA GATE. Do not unlock CONFIRM.

Machine-readable companions. `results/open_data_feasibility.json`, `results/binance_five_asset_calendar_audit.json`, `results/binance_five_asset_calendar_complete_dates.csv.gz`, `results/binance_20230324_halt_forensics.json`, and `results/binance_five_asset_panel_validation.json`.

---

## 11. Segmented calendar and production panel

Construction date 2026-09-21. Official 2021-11 and 2021-12 1m ZIP+CHECKSUM files were added for the five frozen names so that HISTORY_A can start on 2021-11-09 with a 2021-11-08 23:59 anchor. Existing 2022-01 through 2026-08 archives were reused. All checksums verified. Distinct measurement path. No TAQ previous-tick call. No model fit.

Frozen segments, independently counted.

| Segment | Inclusive UTC dates | Count |
| --- | --- | --- |
| HISTORY_A | 2021-11-09 through 2022-07-16 | 250 |
| VALIDATION | 2022-07-17 through 2023-03-23 | 250 |
| excluded halt | 2023-03-24 | 0 |
| HISTORY_B | 2023-03-25 through 2023-11-29 | 250 |
| SCREEN | 2023-11-30 through 2025-04-12 | 500 |
| CONFIRM | 2025-04-13 through 2026-08-25 | 500 |

Total retained production dates 1750. Dates 2026-08-26 through 2026-08-31 unused. Packed lags across the halt rejected. HISTORY_B exists so that no HAR/EWMA/DCC/LSTM state crosses 2023-03-24 and so that SCREEN begins only after 250 post-halt complete days.

Production artifact `data/processed/binance_five_asset_panel.npz`. SHA-256 `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`. Size 359563 bytes. Sidecar SHA-256 `38ed1d4843ea6138d7f87fc0405b668b1ee2fd7403d4cddfba0ca49160737394`. T=1750, N=5. RCov rank 5 on every date. PSD failures 0. Symmetry failures 0. Invalid production dates none.

Full write-up `docs/BINANCE_OPEN_DATA_PANEL.md`. Spec `docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md`. Generic `TemporalProtocol` was not modified. Binance forecasting configuration remains unfrozen. SCREEN was not run. CONFIRM remains locked.
