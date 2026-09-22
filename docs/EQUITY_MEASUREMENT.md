# U.S. equity measurement

This document records the implemented TAQ quote path. The public research narrative is [`README.md`](../README.md). The executed first-stage empirical experiment uses Binance kline closes, not these quotes. The two paths are not interchangeable. The U.S.-equity DATA GATE remains closed.

Implementation lives in `covharness.data.quotes`, `covharness.data.exchange_quotes`, `covharness.data.synchronization`, `covharness.realized`, and `covharness.diagnostics.epps`.

---

## Why quote cleaning precedes covariance estimation

Raw high-frequency quotes are not a clean observation of the latent price. Consolidated NBBO records include crossed markets, extremely wide transient spreads, and isolated midquotes that are inconsistent with neighboring quotes and with contemporaneous trades. Previous-tick synchronization cannot repair those states. It carries the most recent supplied price onto the sampling grid, so an invalid quote immediately before a grid time becomes the synchronized price.

That matters because realized covariance is a quadratic form. If $r_j$ is the synchronized return vector at interval $j$, then

```math
\mathrm{RCov}_t = \sum_{j=1}^{M} r_{t,j} r_{t,j}^{\top}.
```

An artificial return $r_{i,j}$ contributes $r_{i,j}^{2}$ on the variance diagonal and $r_{i,j} r_{k,j}$ to every covariance involving asset $i$. One contaminated interval can therefore distort an entire row and column of the daily proxy.

The primary equity measurement pipeline is

```math
\text{exchange-level quotes}
\to
\text{P1, P2}
\to
\text{P3 (listing venue)}
\to
\text{Q1–Q4}
\to
\text{midquote}
\to
\text{previous-tick synchronization}
\to
\text{log returns}
\to
\text{realized covariance}.
```

The implemented procedure is adapted from Section 5.1 of Barndorff-Nielsen, Hansen, Lunde, and Shephard (2011), who follow the more detailed trade-and-quote cleaning discussion in Barndorff-Nielsen, Hansen, Lunde, and Shephard (2009). The 2011 paper applies P3 before Q1–Q4 and retains quotes from one listing venue.

---

## Cleaning rules

Q1–Q4 are a single implementation, `covharness.data.quotes.clean_nbbo_quotes`. Two input adapters feed that cleaner.

**Paper-style single-exchange path.** Exchange-level quotes from `cqm_*` are restricted to the listing venue (P3) and to the requested TAQ suffix, then passed through P1, P2, and Q1–Q4. Listing venue is taken from CRSP `stocknames.exchcd` on the sample date (1 = NYSE, 3 = NASDAQ) and mapped to Daily TAQ Exchange-field codes (`N` for NYSE, `T`/`Q` for NASDAQ). Ordinary/common shares in the current five-stock extract use the blank or NULL suffix. The adapter is `covharness.data.exchange_quotes.clean_single_exchange_quotes`. It does not copy Q1–Q4.

**Consolidated NBBO alternative path.** The same Q1–Q4 function can be applied to `best_bid` / `best_ask` without P3. That path is retained and labeled.

P1. Delete observations outside regular exchange hours, 09:30–16:00 inclusive on the caller-supplied exchange-local clock.

P2. Delete observations with non-positive or non-finite bid or ask.

P3. On the single-exchange path only, retain quotes issued by the selected listing venue for the requested `sym_root` and `sym_suffix` identity.

Q1. If several quotes for the same security have the same timestamp, replace them by one observation using the median bid and the median ask.

Q2. Delete quotes with a negative spread, $\mathrm{ask}-\mathrm{bid}<0$. Locked quotes with zero spread are retained.

Q3. For each stock-day, compute the median spread after Q1 and Q2, and delete quotes whose spread exceeds ten times that median.

Q4. For each stock-day quote with a complete centered neighborhood, take the 25 observations immediately before and the 25 immediately after the quote, excluding the quote itself. Let $N_i$ be those 50 neighboring midquotes, $\widetilde{m}_i=\mathrm{median}(N_i)$, and $D_i$ the mean absolute deviation of $N_i$ from $\widetilde{m}_i$. Delete the quote if $|\mathrm{mid}_i-\widetilde{m}_i|>10 D_i$. If $D_i=0$, retain the center only when it equals $\widetilde{m}_i$. This $D_i$ is a mean absolute deviation from the local median, not the usual median-absolute-deviation statistic.

Q4 is a symmetric ex-post filter on historical measurement data. Neighborhoods stay inside a stock-day. After the filters, the midquote is $(\mathrm{bid}+\mathrm{ask})/2$.

---

## JPM identity and listing-venue case study

Cleaning rules depend on the market-data representation to which they are applied. On 13 February 2009, the same Q1–Q4 rules applied to consolidated JPM NBBO left clustered five-minute dislocations near 22.6 at 09:45 and 10:40. Restoring the paper's P3 construction, NYSE-issued quotes for JPM, removed those five-minute artifacts. Trades around 09:45 remained near 24.93–24.99. On the single-exchange five-stock panel, no name has a five-minute return above 2 percent. JPM realized variance is 0.00138 versus 0.0477 on cleaned NBBO, and the $5\times 5$ condition number is 39 versus about 931. The production Q1–Q4 code was not changed to obtain that result.

The anomalous fine-frequency JPM results reported before the identity repair were not a Q1–Q4 failure on common stock. Root-only TAQ extraction mixed the blank/NULL-suffix JPM common share with preferred-share series that share `sym_root='JPM'` and NYSE `ex='N'`. Those preferred-share observations were valid quotes for different securities. TAQ `sym_suffix` is therefore required when identifying the security. On this millisecond CQM table the ordinary share is stored as SQL NULL. After the extract was restricted to that identity, no nonblank JPM suffix remains in the common-stock stream.

The case study is in [`notebooks/data_cleaning_example.ipynb`](../notebooks/data_cleaning_example.ipynb). The single-exchange five-stock panel is produced by `experiments/taq_five_stock_single_exchange_pilot.py`. The NBBO alternative remains in `experiments/taq_five_stock_pilot.py`. Derived tables include [`results/taq_jpm_one_minute_forensics_20090213.json`](../results/taq_jpm_one_minute_forensics_20090213.json) and [`results/taq_jpm_provenance_closeout_20090213.json`](../results/taq_jpm_provenance_closeout_20090213.json).

---

## Previous-tick synchronization

Intraday quotes do not arrive on a common clock across assets. After cleaning, covariance estimation constructs a synchronized midquote panel and only then forms returns. The implementation is `covharness.data.synchronization.previous_tick_sync`.

At a grid time $g$, asset $i$ is assigned

```math
P[i,g]
=
\text{latest observed price for } i \text{ with timestamp } \le g.
```

Observations with timestamp $>g$ are never used. There is no linear interpolation and no backward fill from future prices. If asset $i$ has no observation at or before $g$, the synchronized value remains missing. Duplicate `(timestamp, asset)` rows are rejected before this step; Q1 already collapsed same-timestamp quotes to median bid and ask. The grid is supplied by the caller, so the same function supports 1-minute, 5-minute, 15-minute, and offset sampling schemes. Duplicate grid timestamps are rejected. The implementation forward-fills in calendar time on the union of tick times and grid times, then restricts to the grid.

---

## Sampling and realized covariance

**Five-minute baseline.** The primary realized-covariance proxy is based on non-overlapping five-minute returns. This sampling frequency is the baseline compromise between very fine sampling, where market-microstructure effects become more pronounced, and coarser sampling, where fewer intraday observations remain available for covariance estimation. The implementation uses the unscaled Gram matrix. It does not divide by the number of intervals, annualize, apply shrinkage, clip eigenvalues, or otherwise repair the estimator.

**Five-minute subsampling.** The second equity proxy reduces dependence on a single five-minute grid alignment. Starting from the synchronized one-minute price panel, the implementation constructs five shifted price grids with offsets $s\in\{0,1,2,3,4\}$. For each offset, every fifth price observation is selected before log returns are formed, and the existing realized-covariance function is then applied. The five covariance estimates are averaged as

```math
\mathrm{RCov}^{\mathrm{SS}}_t
= \frac{1}{5}\sum_{s=0}^{4}\mathrm{RCov}^{(s)}_t.
```

The averaging occurs across the five covariance estimates. Individual realized-covariance matrices are not divided by their numbers of intervals. The shifted estimates overlap and are not treated as independent measurements, so the construction does not imply a five-fold reduction in measurement-error variance. Missing observations are not handled through pairwise deletion, because doing so would allow different covariance entries to be estimated from different information sets. This subsampled proxy is implemented. It was not the Binance SCREEN target.

Statistical covariance evaluation uses the realized-covariance proxy defined for that market. When global-minimum-variance evaluation is implemented later, the primary economic risk measure on the equity path will include the overnight return outer product. An open-to-close-only economic version will be retained as a robustness channel. Overnight realized covariance and GMV portfolios are not implemented here.

---

## Epps-effect diagnostic

A machine-learning pipeline can treat realized covariance as a label and treat a finer sampling grid as more data. Calendar-time realized covariance records co-movement only when two assets update inside the same sampling interval. If two assets react to the same shock at slightly different observed times, a very fine grid can place those responses in different return intervals, and the contemporaneous covariance contribution of that shock is then near zero. Epps (1979) documented the resulting decline in measured correlation at short intervals. Barndorff-Nielsen, Hansen, Lunde, and Shephard (2011) treat nonsynchronous trading, together with microstructure noise, as a reason that simple calendar-time realized covariance is a noisy and potentially attenuated proxy. We keep unscaled calendar-time $\mathrm{RCov}$ as the baseline proxy and use the frequency scan as a diagnostic rather than as a search for a true matrix.

The implemented diagnostic is `covharness.diagnostics.epps`. It reuses the production previous-tick synchronizer and the unscaled Gram-matrix realized covariance. Realized covariance and implied correlation are recomputed on previous-tick grids of 1, 2, 5, 10, 15, and 30 minutes from the same cleaned quotes. For each frequency it reports the number of synchronized returns, the covariance and correlation matrices, off-diagonal correlation summaries, the individual pairwise correlations, eigenvalues, numerical rank, minimum eigenvalue, PSD status, and a condition number when the smallest eigenvalue is strictly positive. The average off-diagonal series is not forced to be monotonic.

On 13 February 2009, the five-stock single-exchange panel is a one-day demonstration, not an estimate of the magnitude of the Epps effect in U.S. equities. Average off-diagonal correlation is 0.629 at one minute, 0.587 at two minutes, 0.661 at five minutes, and 0.752 at thirty minutes. The figure in the README is that corrected identity-restricted diagnostic. Derived tables are in [`results/taq_five_stock_epps_20090213.json`](../results/taq_five_stock_epps_20090213.json).

---

## Data access and provenance

The equity implementation uses institutional market data obtained through WRDS. The 13 February 2009 five-stock sample (IBM, AAPL, MSFT, JPM, XOM) is a validation extract, not a multi-year dataset. The paper-style path uses `taqmsamp_all.cqm_20090213`. The labeled NBBO alternative uses `taqmsamp_all.nbbom_20090213`. Raw licensed market data are not committed.

WRDS retrieval lives under `experiments/`. Quote cleaning, P3 venue selection, synchronization, return construction, and realized-covariance estimation remain library stages. On 13 February 2009, IBM, JPM, and XOM are NYSE (`N`). AAPL and MSFT are NASDAQ (`T`/`Q`). The production quote extract matches `sym_root`, listing-venue `ex`, and the ordinary-share `sym_suffix`.

The later paper will rerun the Epps diagnostic on a larger panel and report cross-day distributions. That work waits on the DATA GATE. Realized kernels are not implemented.
