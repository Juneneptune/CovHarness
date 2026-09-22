# Empirical data feasibility

Audit date 2026-09-20. This note records a bounded local inventory and an Alpaca credential check. It does not commit a panel. It does not begin empirical fitting. Blocks 1–3 and 4A-1 through 4A-10 remain closed. CONFIRM remains locked. The DATA GATE remains unresolved. A separate pilot protocol amendment is required before any exploratory fitting.

Counts below are labeled exact, catalog-derived, or still unknown. Catalog visibility is not SELECT access. A successful constant-only permission probe is not a downloaded panel.

---

## 1. Existing local data

### Repository data tree

`data/` contains `.gitkeep` plus a gitignored cache. No raw quotes, trades, bars, daily-return panels, or multi-day realized-covariance cubes are stored under the repository root. Nearby scratch directories `/local/scratch/a/lim316/data` and `/local/scratch/a/lim316/research/data` hold unrelated MNIST and model-weight caches, not equity ticks.

### Downloaded and cached one-day TAQ panel

Local path. `data/cache/taq_five_stock_single_exchange_cleaned_20090213.pkl` (13,257,184 bytes). Companion metadata `data/cache/taq_five_stock_single_exchange_cleaning_20090213.json`.

Source. WRDS Daily TAQ millisecond sample `taqmsamp_all.cqm_20090213`, listing venues from `crsp.stocknames`, processed by `experiments/taq_five_stock_single_exchange_pilot.py`. Raw licensed ticks were not written to the repository. The pickle stores cleaned midquotes as previous-tick ticks (`timestamp`, `asset`, `price`), not bid/ask quotes.

| Field | Value | Count basis |
| --- | --- | --- |
| Content | Processed midquote ticks after P3 and Q1–Q4 | Exact pickle read |
| Assets | AAPL, IBM, JPM, MSFT, XOM | Exact |
| Earliest observation | 2009-02-13 09:30:00.050000 | Exact, naive local clock |
| Latest observation | 2009-02-13 15:59:59.982000 | Exact, naive local clock |
| Distinct sessions per asset | 1 | Exact |
| Common-date coverage | All five names on 2009-02-13 only | Exact |
| Timezone | Naive `datetime64[ns]`. Constructed as calendar date plus `time_m` seconds after midnight. Documented as exchange-local. No `America/New_York` tzinfo. | Exact |
| Identity fields in pickle | `asset` ticker only. Venue and `sym_suffix` are in the JSON companion, not in the pickle. Ordinary-share suffix was SQL NULL on this CQM table. | Exact |
| Rows after Q4 | AAPL 232484, IBM 49451, JPM 142740, MSFT 199598, XOM 73315. Total 697588. | Exact |
| Raw quotes locally | Absent | Exact |
| Trades or bars locally | Absent | Exact |
| Multi-day RCov cube | Absent | Exact |
| Matched daily returns | Absent | Exact |
| Per-asset RQ panel | Absent | Exact |

One-day derived RCov exists only as JSON diagnostics, not as a reusable `(T,N,N)` panel. `results/taq_five_stock_single_exchange_20090213.json` records a 5×5 unscaled previous-tick five-minute RCov on that date. After dropping incomplete 09:30 grid rows, 78 synchronized prices and 77 returns. That matrix is a measurement diagnostic. It is not a 500-session target.

The labeled NBBO alternative is documented in `results/taq_five_stock_pilot_20090213.json` from `taqmsamp_all.nbbom_20090213`. No NBBO tick pickle is present locally. That path remains a labeled alternative, not the production cleaner input.

### WRDS catalog versus exercised access

Recorded 2026-09-14 in `results/data_gate_wrds_access_inventory_20260914.json`. Connection succeeded for username `lim316`. Credentials were not recorded. This audit did not start a new WRDS extraction and did not guess table names.

**Sample Daily TAQ (downloaded in Block 1, not a long panel).**

- `taqmsamp` / `taqmsamp_all`, pattern `cqm_YYYYMMDD`. Catalog-visible. USAGE true. In `wrds.list_libraries()`. Distinct CQM date tables 1. Earliest and latest identifier 2009-02-13. Constant-only `SELECT 1` on `taqmsamp_all.cqm_20090213` succeeded. The five-stock extract was exercised on that date.
- `taqsamp` / `taqsamp_all`. Catalog-visible. USAGE true. In library list. Discovered `cq_*` dates 2008-01-07 through 2008-01-11 (5 identifiers). No quote-row extract was exercised in this audit.

**Production Daily TAQ (catalog-visible, SELECT not exercised successfully).**

- `taqmsec` views. Catalog-visible. USAGE true. Not in `wrds.list_libraries()`. `list_tables` error “You do not have permission to access the taqmsec library”. Information-schema identifiers span 2003-09-10 through 2026-08-19, 5772 distinct `cqm_YYYYMMDD` names (catalog-derived, not SELECT-verified). Bounded `SELECT 1 LIMIT 1` on `taqmsec.cqm_20030910`, `cqm_20150227`, and `cqm_20260819` failed with permission denied for underlying schemas `taqm_2003`, `taqm_2015`, and `taqm_2026`. View privilege `SELECT=true` did not imply readable rows.
- Year schemas `taqm_2003` through `taqm_2026`. Catalog-visible. USAGE false. Not in library list. Combined catalog identifiers 5788. Latest year-schema identifier in that inventory 2026-09-11. No successful SELECT.

**Empty or unusable TAQ schemas.**

- `taq` and `taq_common`. USAGE true, not in library list, zero information-schema tables, `list_tables` permission denied.
- Annual `taqm_YYYY` USAGE denied as above.
- `contrib_liquidity_taq` catalog-visible, USAGE denied.

**CRSP identity and daily returns (permission probed, not downloaded).**

- `crsp.stocknames`. Constant-only SELECT succeeded. Name-history columns include `permno`, `ticker`, `exchcd`, `namedt`, `nameenddt`, `ncusip`, `cusip`, `shrcd`, `shrcls`. Date fields span 1925-12-31 through 2024-12-31 (catalog min/max on those columns). No local extract.
- `crsp.dsf`. Constant-only SELECT succeeded. Columns include `permno`, `date`, `ret`, `retx`, `prc`, `openprc`, `bid`, `ask`, `vol`, `cfacpr`, `cfacshr`. Min date 1925-12-31, max date 2024-12-31. No local daily-return panel. Open-to-close versus close-to-close remains unresolved.

Calendar implication from the 2026-09-14 inventory. Catalog identifier counts could in principle exceed 1250 and 1500 source dates. `select_verified_production_cqm_date_tables` is 0. `production_select_access_verified` is false. `usable_panel_not_yet_proven` is true. Those flags are unchanged because this audit did not rerun WRDS.

---

## 2. Block 1 measurement compatibility

The production price input is irregular quotes, not trade bars.

Sequence. Exchange-level quotes (`bid`, `ask`, `ex`, `sym_root`, `sym_suffix`) restricted to the CRSP listing venue and ordinary-share suffix (P3 plus identity). Then P1 (regular session 09:30–16:00 inclusive on the caller-supplied exchange-local clock), P2 (strictly positive finite bid and ask), Q1 (median bid and ask at a duplicated timestamp), Q2 (drop negative spreads, keep locked zero spreads), Q3 (drop spreads above ten times the stock-day median), Q4 (centered 50-neighbor midquote filter). Midquote `(bid+ask)/2` is the price passed to previous-tick synchronization.

Previous-tick. For grid time `g`, the price is the latest observation with timestamp `<= g`. No interpolation and no backward fill. Duplicate `(timestamp, asset)` rows are rejected before sync. Missing cells on a grid row are dropped rather than pairwise-deleted (`drop_incomplete_open`). Daily RCov is the unscaled Gram matrix `R'R` of synchronized log returns. Opening and overnight returns are out of scope for that estimator.

What historical quotes can satisfy. Bid/ask, venue, suffix identity, millisecond timestamps, Q1–Q4, midquote previous-tick, five-minute and subsampled proxies, per-asset RQ from the same synchronized returns, and origin-day BNS instruments. The one-day sample already exercised that chain.

What would change if trade-bar closes were used. A bar close is an aggregated last trade (or an OHLCV close), not a cleaned midquote. P2–Q4 operate on spreads. They cannot run on a single close. Venue P3 and suffix identity are typically already collapsed in a SIP or IEX bar. Previous-tick on bar closes would treat a regular grid as if it were irregular quotes and would disguise microstructure. Subsampled RCov from 1-minute bars is a different proxy from quote previous-tick RCov. We do not treat bars as quotes and we do not alter the existing estimator.

Alpaca historical quotes, if later entitled, map onto bid/ask fields (`bp`, `ap`) with RFC-3339 UTC timestamps and exchange codes (`bx`, `ax`). That schema could feed the cleaner after timezone conversion to exchange-local session bounds and after an explicit identity rule. Alpaca minute bars (`o,h,l,c,v,n,vw`) cannot.

---

## 3. Alpaca SIP probe

Official documentation read on 2026-09-20.

- Historical overview `https://docs.alpaca.markets/us/docs/historical-stock-data-1`
- Bars `GET https://data.alpaca.markets/v2/stocks/bars` (`https://docs.alpaca.markets/us/reference/stockbars`)
- Quotes `GET https://data.alpaca.markets/v2/stocks/quotes` (`https://docs.alpaca.markets/us/reference/stockquotes-1`)

Documented feeds. `sip` is consolidated UTP/CTA (all US exchanges). `iex` is IEX only, approximately 2.5 percent of volume, and is the only feed usable without a subscription. Defaults in the OpenAPI schema list `sip` as the feed default. Free-tier accounts still cannot retrieve SIP. We would have set `feed=sip` explicitly and would not have fallen back to IEX.

Intended bounded request (not sent).

- Symbols AAPL and MSFT.
- Dates 2017-06-06 and 2023-06-06.
- Bars `timeframe=1Min`, `start`/`end` RFC-3339 session bounds, `feed=sip`, `adjustment=raw`, `asof=-` (skip symbol mapping), `sort=asc`, pagination via `next_page_token`, hard request cap.
- Quotes one page, `limit=100` per symbol/date, `feed=sip`, same start/end, `asof=-`.

**Credentials.** Absent. No `ALPACA_*` or `APCA_*` environment variables. No repository `.env`. No `configs/` secrets. No `~/.alpaca.env`. No Alpaca-named credential files in the project or user store. The network probe was skipped.

HTTP status, body, date ranges, counts, fields, truncation, and pagination are therefore unknown. Untested SIP access is not described as available. An HTTP 200 with empty bars would not have been treated as coverage even if credentials had existed.

---

## 4. Feasibility result

### What existing intraday data can support

The local one-day five-name cleaned-tick cache can support measurement-chain regression tests, Epps diagnostics already stored in `results/`, and reconstruction of that day’s previous-tick RCov and per-asset RQ from the pickle. It cannot support rolling `m=250` windows, VALIDATION/SCREEN/CONFIRM allocation, or any 5–10 asset exploratory horse race.

WRDS sample CQM on 2009-02-13 was successfully extracted once. It remains a one-session demonstration. Monthly-TAQ sample dates 2008-01-07 through 2008-01-11 were catalog-visible only in this audit.

### What existing daily-only data can support

Nothing locally. Remotely, `crsp.dsf` passed a constant-only SELECT probe with a catalog date span 1925-12-31 through 2024-12-31. That permission is not a downloaded panel and does not produce quote-based RCov, synchronized intraday returns, or RQ. Daily CRSP `ret`/`openprc` could later supply matched close-to-close or open-to-close returns for DCC, LW, and LSTM-BEKK if an extract is authorized. Statistical evaluation still requires the open-to-close RCov proxy defined in the harness. Daily-only data cannot substitute for that proxy.

### Whether the SIP probe succeeded

It did not run. Credentials were absent.

### What remains unverified for a 5–10 asset, approximately 500-session exploratory pilot

- Production Daily TAQ SELECT on a multi-year `cqm_YYYYMMDD` span.
- Quote-row completeness, session coverage, and suffix identity after P3 for a frozen 5–10 name universe.
- Common-date intersection after cleaning and previous-tick (usable sessions, not catalog table counts).
- A stored `(T,N,N)` RCov cube and aligned `(T,N)` RQ and daily-return windows.
- Open-to-close versus close-to-close daily returns.
- SIP quote entitlement at Alpaca or any other vendor.
- Legal/license terms for a local multi-year extract.
- A trading calendar that maps catalog identifier gaps to NYSE closures.

A 500-session exploratory window is still shorter than the confirmatory allocation. The protocol requires `T-m >= 1000` with SCREEN and CONFIRM minima of 500 targets each. Even a complete 500-session quote panel would not unlock CONFIRM or final `PREREGISTRATION.md`.

---

## 5. Exact next step

Request Purdue WRDS entitlement to read production Daily TAQ millisecond quotes (`taqm_YYYY` underlying tables used by `taqmsec.cqm_YYYYMMDD`). After entitlement, rerun only the already-specified bounded constant-only `SELECT 1 LIMIT 1` on three production dates (one early, one middle, one late). If those three succeed, the following assembly step is a 5–10 name, two-week quote extract used solely to measure cleaning yield and common-session counts. Do not download a 500-session panel until that two-week yield is recorded. Do not fit models.

If WRDS Daily TAQ SELECT remains denied, the alternative is a paid consolidated-quote source. Alpaca SIP is documented but untested here because credentials are absent and SIP requires a subscription. IEX bars are not a headline measurement path.

A separate pilot protocol amendment is required before exploratory fitting. That amendment is not this document. CONFIRM stays locked. Final `PREREGISTRATION.md` stays absent. The DATA GATE stays unresolved.
