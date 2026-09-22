# Binance open-data measurement specification

Specification date 2026-09-21. This document freezes the measurement convention for a distinct five-asset Binance open-data empirical branch of covharness. Production realized-covariance, realized-quarticity, and daily-return arrays are recorded in `docs/BINANCE_OPEN_DATA_PANEL.md`. This document does not fit models. It does not change the U.S.-equity TAQ estimator. CONFIRM remains locked. Final `PREREGISTRATION.md` remains absent.

Blocks 1–3 and 4A-1 through 4A-10 remain closed.

---

## 1. Experiment role

The constraint is that the U.S.-equity Daily TAQ path is not yet entitled for a long panel. The choice is to run a separate open-data empirical benchmark on official Binance public spot klines so that the already implemented forecasting, loss, and inference stack can be exercised on a long synchronized real-data panel.

This branch is a real-data empirical benchmark of covharness. It is suitable for testing the implemented covariance-forecasting framework on a freely reproducible panel. It is separate from the eventual U.S.-equity TAQ experiment.

This branch is not a substitute that retroactively changes the TAQ measurement definition. It is not evidence that the original U.S.-equity DATA GATE passed. It is not a representative sample of the cryptocurrency market. It is not a model-selected universe.

The first Binance empirical universe is exactly BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, and ADAUSDT. That set was taken from the earlier twelve-name candidate list using long common archive coverage and data completeness only, before any model fitting. No symbol is added or removed in this specification.

---

## 2. Source

Official Binance public monthly 1-minute spot kline archives from `https://data.binance.vision`. Matching `.CHECKSUM` files are required. SHA-256 must verify before parse. No REST backfill. No unofficial mirror. No yfinance.

The raw observation is the kline **close**. That close is a regular transaction-price observation on a 1-minute grid. It is not a quote, not a midpoint, not NBBO, and not the existing TAQ previous-tick estimator.

Primary production measurements are derived from 1-minute files only. Direct Binance 5-minute archives are not mixed into the primary construction. The earlier 1m-versus-5m close equality check remains a validation diagnostic.

Timestamp units are detected per monthly file. Official documentation states that Spot timestamps from 2025-01-01 onward are microseconds. The long-panel audit found milliseconds through 2024-12 and microseconds from 2025-01. Open times are converted to UTC only after that detection.

The production implementation lives in a distinct data/measurement path. It does not call `previous_tick_sync` on kline closes as if they were irregular equity quotes.

---

## 3. UTC statistical day

A Binance statistical day $t$ is the half-open UTC interval $[00{:}00{:}00,\ 24{:}00{:}00)$.

Realized covariance, realized quarticity, and the daily return for date $t$ all refer to that same 24-hour interval. There is no equity-style separate overnight interval. The equity 09:30–16:00 session convention is not used.

---

## 4. Primary five-minute return convention

The primary sampling frequency is five minutes. Endpoints are taken from the one-minute close grid, not from a separate 5m archive.

Let $p$ denote the synchronized log close. For UTC date $t$, the full-day construction uses **288** non-overlapping five-minute returns covering the complete 24-hour interval. The first return of date $t$ is anchored on the immediately preceding 23:59 UTC one-minute close.

```math
\begin{aligned}
r_{t,1}
&=
p_{t,00{:}04}-p_{t-1,23{:}59},\\
r_{t,2}
&=
p_{t,00{:}09}-p_{t,00{:}04},\\
&\vdots\\
r_{t,288}
&=
p_{t,23{:}59}-p_{t,23{:}54}.
\end{aligned}
```

The 288 interval endpoints on date $t$ are therefore $00{:}04,00{:}09,\ldots,23{:}59$. That grid is the close of each five-minute interval $[00{:}00,00{:}05),\ldots,[23{:}55,24{:}00)$. It matches the earlier diagnostic in which a 5m kline close equalled the 1m close four minutes after the 5m open.

The 287-return within-day version is not primary. It would discard the first five-minute interval of every UTC day. The boundary-anchored 288-return construction covers the full statistical day.

Requiring the previous 23:59 close as a boundary anchor does **not** require every minute of the previous UTC day to be complete. 2023-03-24 is excluded as a statistical day, but its 23:59 close remains the calendar anchor for 2023-03-25.

Missing required endpoints are rejected. There is no forward fill.

---

## 5. Realized covariance

For the synchronized five-asset five-minute return vectors $r_{t,j}\in\mathbb{R}^{5}$,

```math
\mathrm{RCov}_{t}
=
\sum_{j=1}^{288}
r_{t,j}\,r_{t,j}^{\top}.
```

Unscaled Gram matrix. No annualization. No winsorization. No clipping. No post-hoc eigenvalue repair. Invalid matrices are recorded, not silently repaired. No production date required a repair.

This is the primary Binance proxy. It is not the equity open-to-close previous-tick RCov.

---

## 6. Realized quarticity

Using the same 288 five-minute returns, per-asset realized quarticity is

```math
\mathrm{RQ}_{i,t}
=
\frac{288}{3}
\sum_{j=1}^{288}
r_{i,t,j}^{4}.
```

No annualization, winsorization, clipping, or standardization. Zero is allowed. Negative or non-finite values are rejected. This is the HARQ $(T,N)$ input, not the GW aggregate $\mathrm{RQ}_{\mathrm{agg}}$.

---

## 7. Daily return

Daily-return models receive the close-to-close log return over the same UTC 24-hour interval.

```math
R_{i,t}
=
\log P_{i,t,23{:}59}
-
\log P_{i,t-1,23{:}59}.
```

DCC, Ledoit–Wolf, and LSTM-BEKK use this $R_{t}$. They do not use a different midnight or session-open convention. Daily returns, RCov, and RQ therefore refer to the same statistical day.

The production arrays are stored in `data/processed/binance_five_asset_panel.npz` with segment labels. See `docs/BINANCE_OPEN_DATA_PANEL.md`.

---

## 8. Missing-data and halt hierarchy

No generic fill rule is created merely to recover one day.

1. Ordinary complete day. Use observed one-minute closes directly.
2. Missing minute(s) with no verified market-structure explanation. The UTC day is unusable. No interpolation and no forward fill.
3. Documented venue-wide spot-trading halt. Any retention of that day must be an explicitly labeled venue-halt measurement rule.

---

## 9. Evidence for 2023-03-24

### Kline forensics

Read from the already downloaded official 2023-03 monthly 1m ZIPs. No model and no return-based screen.

All five symbols share **one identical contiguous missing block**.

| Field | Value |
| --- | --- |
| First missing minute | 2023-03-24 12:40 UTC |
| Last missing minute | 2023-03-24 13:59 UTC |
| Number of missing 1m slots | 80 |
| Number of blocks | 1 |
| Minute immediately before | 12:39 UTC, present on all five |
| Minute immediately after | 14:00 UTC, present on all five |
| 00:00 UTC present | Yes, all five |
| 23:59 UTC present | Yes, all five |
| Present 1m bars that day | 1360 of 1440 |
| Five-minute endpoints inside the hole | 16 of 288 ($12{:}44$ through $13{:}59$ by fives) |

Closes immediately before and after the block (diagnostic prices, not returns).

| Symbol | Close at 12:39 | Close at 14:00 |
| --- | --- | --- |
| BTCUSDT | 28080.00 | 27925.59 |
| ETHUSDT | 1789.52 | 1763.69 |
| BNBUSDT | 323.80 | 322.00 |
| LTCUSDT | 91.79 | 90.54 |
| ADAUSDT | 0.3629 | 0.3584 |

Machine-readable companion `results/binance_20230324_halt_forensics.json`.

### Official halt evidence

The task supplied the following statement. Binance reported a spot-trading system interruption on 2023-03-24, with spot trading disabled during a matching-engine incident and trading subsequently resumed that day.

Indexed excerpts of Binance’s own CEO blog and Square republication give the same timeline. Spot trading disabled 11:27 UTC. Matching-engine trailing-stop bug. Fix 13:30 UTC. Platform resume 14:00 UTC except trailing-stop orders. Cited pages.

- `https://www.binance.com/zh-CN/blog/from-our-ceo/6789340645608890113`
- `https://www.binance.com/en/square/post/696655`
- `https://www.binance.com/en/support/announcement/detail/813a31506e9f478ea8c1058b425df87a`

Live GET of those URLs on 2026-09-21 returned a Binance WAF / CloudFront challenge. The times above are therefore indexed official-page excerpts plus the statement supplied for this task, not a locally archived HTML capture. No alternative unofficial cause is invented.

### Match to the kline hole

The missing 1m block $12{:}40$–$13{:}59$ UTC lies inside the announced disable-to-resume window $11{:}27$–$14{:}00$ UTC. The first bar after the hole is exactly 14:00 UTC, which matches the announced resume. The kline gap does **not** begin at 11:27. Bars continue through 12:39, so the published kline hole is a subset of the announced halt, not a minute-for-minute map of the disable timestamp.

The pattern is a documented venue-wide absence of spot matching, not an unexplained idiosyncratic data-feed dropout. It is also not a complete 1m blackout of the announced interval.

---

## 10. Treatment of 2023-03-24

We recommend **A**. Exclude the entire UTC day from origins and from targets.

Reasons. Sixteen of 288 primary five-minute endpoints fall in the hole. A venue-halt previous-observation rule (option B) would insert a block of zero five-minute returns. That is a different proxy, not a recovered observation. We do not choose B to keep one day.

2023-03-24 is therefore not a statistical day for RCov, RQ, or the daily-return target. It is not a forecast origin.

The 23:59 close on 2023-03-24 **is** retained as the boundary anchor for 2023-03-25. That price exists on all five names. Daily return and the first five-minute return on 2023-03-25 still use $P_{2023-03-24,23{:}59}$ under the frozen 24-hour definitions. Completeness of every minute on the 24th is not required for that anchor.

---

## 11. Segmented temporal calendar

Excluding 2023-03-24 as a statistical day leaves a two-calendar-day gap between 2023-03-23 and 2023-03-25. Packed complete-day lags are **rejected**. 2023-03-23 and 2023-03-25 are not adjacent statistical days.

The frozen protocol is a hard temporal break with a second 250-day HISTORY burn-in after the halt. No one-day forecast, HAR lag, EWMA recursion, DCC state, or LSTM-BEKK recurrent state crosses 2023-03-24. SCREEN and CONFIRM begin only after those 250 post-halt statistical days. Model recursions are not modified to skip the hole. Generic `TemporalProtocol` is unchanged.

Exact dates and production hashes are in `docs/BINANCE_OPEN_DATA_PANEL.md`. Implementation is `src/covharness/data/binance_calendar.py`.

---

## 12. Planned subsampled robustness

Because the raw branch has a synchronized one-minute close grid, a later five-minute subsampling robustness estimator can be built from offset five-minute grids on that 1m series. It is not implemented here. It is not claimed to be numerically identical to the equity TAQ midquote-subsampled estimator in `covharness.realized.subsampled`. The exact offset and averaging definition must be reviewed before coding. Direct 5m archives remain out of the primary path.

---

## 13. Distinction from U.S.-equity TAQ

| Item | Equity TAQ branch | Binance open-data branch |
| --- | --- | --- |
| Price | Quote midpoint after P1–P2, P3, Q1–Q4 | 1m kline close |
| Clock | Exchange-local 09:30–16:00 | UTC $[00{:}00,24{:}00)$ |
| Overnight | Out of statistical RCov. Later GMV channel | None. Midnight is a calendar seam |
| Synchronization | Previous-tick on irregular quotes | Regular 1m close grid |
| Primary $M$ | Session five-minute returns after dropping incomplete opens | 288 five-minute returns including the midnight-crossing interval |
| Identity | CRSP listing venue and share class | USDT spot pair names |
| Missing data | Drop incomplete grid rows | Complete-day rule, halt hierarchy above |

These differences do not make the Binance experiment invalid. They make it a different experiment.

---

## 14. Limitations

- Five USDT pairs are not the crypto market.
- Kline closes omit spread and venue microstructure that TAQ midquotes capture.
- 2023-03-24 is excluded as a statistical day. Packed complete-day lags across that hole are rejected. Recursions do not cross the break.
- Subsampled robustness is planned, not implemented.
- Realized kernels remain unimplemented on both branches.
- Binance forecasting configuration and tuning rules are not frozen. No model has been fit.
- CONFIRM remains locked. U.S.-equity DATA GATE remains closed.

---

## 15. DATA-GATE component status

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
| Binance forecasting configuration/tuning rules | NOT YET FROZEN |
| Binance empirical model fitting | NOT STARTED |
| SCREEN | NOT RUN |
| CONFIRM | LOCKED / NOT RUN |
| U.S.-equity DATA GATE | UNCHANGED / CLOSED |

Measurement, calendar, and production arrays are frozen. Forecasting configuration is not. The complete Binance empirical DATA GATE has not passed for model fitting. No model has been fit.
