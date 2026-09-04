[design_v4_demo_plan.md](https://github.com/user-attachments/files/31852884/design_v4_demo_plan.md)
# Covariance Benchmark — Consolidated Design (v4)

**This file supersedes and replaces `week1_plan.md`, `design_draft_v2.md`, and
`design_draft_v3.md`.** Delete those — everything in them is here, nothing was dropped.

Companion: `project_ledger.md` (paper status + saved concepts). You will write
`PREREGISTRATION.md` on Day 2 from Part 1 below.

---

# PART -1 — SCOPE TIERS (read this first)

The rest of this document describes the **full** project. Not all of it fits in a week. This part
splits it into what ships, what is a stretch, and what is weeks 2-4.

## The discipline that makes the week work

**By end of Day 5 you must have a complete, shippable artifact.** Not "mostly done" — runnable
end to end, with results, figures, and a README. Days 6-7 are then *bonus*, spent on the risky
deep-learning model. If Day 6 goes badly you ship what you had on Day 5 and lose nothing.

Most one-week projects fail because the risky item is attempted in parallel with the core and both
end up half-finished. Sequence them instead.

## TIER 1 — SHIP IN WEEK 1 (Days 1-5). This alone is the resume artifact.

| Component | Scope |
|---|---|
| Data | N = 30, **two** proxies (`RCov_5min`, `RCov_5min_subsampled`), full cleaning, PD logging |
| Validation | Stylized-fact unit tests + **Epps curve** |
| Losses | Stein/multivariate-QLIKE, Frobenius, DRD decomposition, GMV (gross) |
| Protocol | Rolling driver, m = 250, monthly cadence, time-ordered splits, **confirm lock** |
| Inference | DM + HAC, MCS, GW (3 instruments), standard MZ |
| Econometric models (8) | `RW`, `EWMA`, `HAR-DRD`, `HARQ-DRD`, `LW-linear`, `LW-NL`, `DCC`, `DCC-NL` |
| ML model (1) | `Ridge-DRD` |
| Figures | Epps curve; eigenvalue shrinkage (sample vs linear vs NL vs true); **the three synthetic demos**; cumulative `d_t` |
| Docs | `PREREGISTRATION.md`, README |

**Nine models, complete inference stack, six figures.** The three synthetic demos (a non-robust
loss flipping the ranking; a naive t-test over-rejecting at 25%; DM blind to regime dependence where
GW is not) cost about two hours total and are the most distinctive content in the repo.

## TIER 2 — WEEK 1 STRETCH (Days 6-7, only if Day 5 shipped)

In priority order:
1. **`LSTM-BEKK`** (faithful, daily returns) — the headline DL competitor. 1.5-2 days honestly.
2. `XGBoost-DRD` — 31 fits, few hours once features exist.
3. SPA (benchmark = `HAR-DRD`) — the "does anything beat the workhorse?" headline.
4. Augmented MZ-GLS — regime miscalibration testing.
5. Third proxy (`RCov_refresh_kernel`).

## TIER 3 — WEEKS 2-4 (this is the paper, not the artifact)

`LSTM-BEKK-RC` (information parity — the most valuable number in the paper); `GHAR`; universe
variant (C), the 20-draw robustness run; N = 50 with kernel proxies; `GPVar`; correlation-change
diagnostics; multi-horizon forecasts; net-of-cost GMV; and the learned-shrinkage agenda in Part 6.

## Why Tier 1 alone is resume-worthy — including without a deep model

Counterintuitive but true for quant recruiting: **the harness is the credential, not the models.**
A hiring reader is checking whether you can (a) handle intraday data correctly, (b) estimate
high-dimensional covariance without fooling yourself, and (c) evaluate forecasts without
manufacturing significance. Tier 1 demonstrates all three. A deep model demonstrates none of them
and is the easiest part to fake.

Resume/GitHub bullets Tier 1 supports, all literally true on delivery:

- Built a leak-free benchmark for multivariate covariance forecasting: 9 models from random walk
  through DCC with nonlinear eigenvalue shrinkage, evaluated under proxy-robust matrix losses with
  Model Confidence Set screening and pre-registered, chronologically separated confirmatory tests.
- Implemented realized-covariance estimation from intraday trades with synchronization and
  microstructure diagnostics (Epps-effect validation, positive-definiteness monitoring).
- Demonstrated that three standard evaluation shortcuts in the volatility literature — non-robust
  loss functions, unadjusted t-tests on serially correlated loss differentials, and unconditional
  average comparison — each produce the wrong answer on data with a known ground truth.

That third bullet is the one that gets an interview, and it is Tier 1 work.

**What Tier 1 cannot claim:** anything about deep learning. The README must say so plainly —
"deep-learning comparison in progress" is credible; an overclaimed MLP is not.

---

# PART 0 — PRE-FLIGHT

## 0.1 The two blocking risks

**Risk A: data acquisition.** Free intraday equity data is the single most likely thing to eat two
days. Oxford-Man Realized Library is discontinued. Options in order of preference:
- **Alpaca** free API — historical minute bars, generous history, clean API. Best default.
- **Polygon.io** free/starter tier — minute aggregates.
- **Databento** — paid but cheap for a one-off pull; highest quality.
- **Fallback:** if intraday acquisition stalls past Day 1 noon, build the entire harness on
  **simulated data from a known DCC-NL process** (you can generate this — see Day 2). The harness
  is testable without real data, and simulated data has a *known* true Sigma, which is strictly
  better for validating losses and tests. Swap in real data when it arrives.

**Risk B: reading gap.** Read **Liu, Patton & Sheppard (2015)** (`H14`) before writing the RCov
pipeline. It tells you which realized measure actually wins and why 5-min is the default. Two hours.
Skim **Archakov & Hansen (2021)** (`M13`) before Day 6 for the matrix-transform justification.

## 0.2 Acquisition: what actually works

**Yahoo / yfinance is not viable.** Intraday history is capped at roughly 60 days for 5-minute
bars and 7-30 days for 1-minute. You need 8-15 years. Rule it out now.

Ranked options:

| Source | Cost | Notes |
|---|---|---|
| **WRDS / NYSE TAQ** | Free with university access | **Check this first.** What GHAR, SpotV2Net, Christensen et al. use. Trade-and-quote level, full history. Makes your results directly comparable to the papers you are refereeing. |
| Databento | Pay-as-you-go, ~$100-500 for a one-off historical pull | Highest quality commercial option; clean API; MBP/trades. |
| Polygon.io | ~$30-200/mo | Minute aggregates, full history on paid tiers. |
| Alpaca | Free tier | **Caution:** free tier is IEX-only, roughly 2-3% of consolidated volume. RCov from IEX-only prints is materially noisier and more Epps-affected than SIP data. Usable for pipeline development, not for headline results. |
| FirstRate Data / Kibot | One-off purchase | Pre-cleaned minute bars; convenient, less transparent cleaning. |

**Decision rule:** try WRDS today. If unavailable within 24 hours, buy one Databento pull for
N=50 tickers over 10 years and move on. Do not burn three days on free-data archaeology.

**Fallback that keeps the week alive:** build and validate the entire harness on **simulated data
from a known DCC-NL process**. Simulation gives you a *known true Sigma*, which is strictly better
for testing losses and inference than real data. Swap in real data when it lands. Nothing on
Days 2-3 requires real data.

## 0.3 Universe selection — why "random" needs restructuring, and how to make it a strength

Your instinct to randomize is right; the framing needs two fixes.

**Leak 1: survivorship.** Selecting stocks "continuously listed 2012-2024" conditions on survival.
Firms that were acquired, delisted, or blew up are exactly the ones with interesting covariance
behavior. This biases every model, but plausibly biases them *unequally* — a model that adapts fast
gains less when nothing ever breaks.

**Leak 2: liquidity.** A uniformly random draw from all listed equities pulls in names where
5-minute RCov is close to meaningless: stale prices, many zero returns, severe Epps attenuation.
Your covariance target would be measuring illiquidity, not co-movement.

### The protocol

**Step 1 — point-in-time eligible pool.** At the start of each year Y, using only data through
year Y-1:
- listed and trading for all of Y-1
- median daily dollar volume in the top 500
- price > $5 throughout
- fewer than 5% zero-return 5-minute bars in Y-1 (the liquidity screen that actually matters for RCov)

Take the top 200 survivors → `pool_Y`. This uses no future information.

**Step 2 — universe draw.** Three variants, all reported:
- **(A) Headline, deterministic:** top N by liquidity in `pool_{Y0}`, fixed for the sample.
  Comparable to the DJIA-30 papers.
- **(B) Random draw, seeded:** one random N-subset of `pool_{Y0}`, seed fixed and published.
- **(C) Universe robustness — this is the contribution.** Draw **B = 20 independent N-subsets**
  from `pool_{Y0}` and run the entire benchmark on each. Report the **distribution of model
  rankings and loss ratios across universes**.

Variant (C) answers a question no paper in your literature answers: *does the winner depend on
which 30 stocks you happened to pick?* If DL wins on 4 of 20 universes, that is a finding — and it
is the cross-sectional analogue of the temporal-streakiness diagnostic you already committed to.
Cost: B times the compute, so run it with the cheap models first and only extend to the expensive
ones if the pipeline is fast enough.

**Survivorship:** variants A-C still condition on surviving year Y0. Document it as a limitation
and, in week 3, add the annual-redraw version where the universe is refreshed each January and
covariance models are re-initialized at the boundary.

## 0.4 The dimension constraint nobody writes down

RCov built from M intraday returns has rank at most min(M, N). Stein loss needs
`logdet(Sigma_proxy)`, so you need **M > N**, and for a decently conditioned estimate you want
**M / N >= 3**.

| Sampling | M per day | Max N at M/N >= 3 | Note |
|---|---|---|---|
| 5-min | 78 | **26** | Microstructure-safe; the standard choice |
| 1-min | 390 | **130** | Noise-contaminated; needs subsampling or kernels |
| 1-min subsampled / realized kernel | effective ~200-390 | 60-130 | The route to larger N |

**Decision:** N = 30 headline with 5-min (M/N = 2.6, marginal but comparable to the literature),
plus N = 50 secondary using 1-min subsampled and kernel proxies (M/N ~ 7). State the M/N ratio in
the paper — this constraint is why the multivariate literature is stuck at small N, and naming it
is itself a contribution to the discussion.

## 0.5 Cleaning checklist (Barndorff-Nielsen et al., "Realized kernels in practice" [M])

Apply in order, log the count removed at each step:
1. Trading hours only (09:30-16:00 ET); handle NYSE early closes explicitly.
2. Drop zero prices, and entries with abnormal sale conditions / corrected trades.
3. Aggregate simultaneous trades to the median price.
4. Outlier filter: drop prints deviating more than k mean-absolute-deviations from a centered
   rolling median of 50 neighbouring prints (k = 10 is standard).
5. Corporate actions: split/dividend adjust, and verify no adjustment-induced jumps remain.

## 0.6 The overnight decision (declare it, most papers hide it)

RCov from intraday returns is **open-to-close**. Portfolios hold overnight. Three options:
- (a) open-to-close only — clean, but the GMV channel then evaluates a risk measure that excludes
  a large share of actual return variance;
- (b) add the squared overnight return / outer product of overnight returns;
- (c) scale open-to-close by a constant estimated from the ratio of close-to-close to open-to-close
  variance.

**Pre-commit to (b) for the GMV channel and (a) for the statistical channel**, and report the
other as a robustness check. This asymmetry is defensible and explicit; silence is not.

## 0.7 Proxy set (three, pre-declared)

1. `RCov_5min` — previous-tick on a fixed 5-min grid. Primary.
2. `RCov_5min_ss` — subsampled: average of 5 grids offset by 1 minute. Variance reduction, no bias
   change.
3. `RCov_refresh_kernel` — refresh-time synchronization + multivariate realized kernel
   (Barndorff-Nielsen-Hansen-Lunde-Shephard). The Epps-robust proxy; PSD by construction.

Rank stability across all three is a pre-committed requirement, not a bonus check.

## 0.8 Realized quarticity (you asked for it — here is where it enters)

Per-asset realized quarticity `RQ_i,t = (M/3) sum_j r_{i,t,j}^4` estimates the measurement-error
variance of `RV_i,t`. Multivariate analogue for covariance entries is messier; use the per-asset
version plus the average across assets as a matrix-level reliability scalar.

Three distinct uses — keep them separate:
- **As a model feature** (HARQ-style attenuation correction), in `HARQ-DRD` and as an ML feature.
- **As a GW conditioning variable** — "does the DL model win when measurement is poor?"
- **As a proxy-quality filter** — flag days where the proxy is unusually unreliable, and report
  results with and without them.

Carry the caveat: RQ is a fourth-moment estimator, hence very noisy and heavy-tailed, and jumps
contaminate it badly. Use a jump-robust variant (tri-power quarticity) as a robustness check.

---

## 0.9 Frozen design decisions (write these down, do not revisit mid-week)

| Decision | Value | Why |
|---|---|---|
| Universe | N = 30 large-cap US equities, continuously listed over sample | RCov rank ≤ 78 with 5-min returns; N < 78 needed for PD proxy (Stein loss needs log det) |
| Frequency | Daily forecasts of next-day covariance | Matches HAR/DCC literature |
| Sampling | 5-min primary; 1-min subsampled; 15-min | Three proxies for rank-stability check |
| Rolling window | **m = 250 days, fixed, identical for every model** | (a) GW validity requires fixed finite m; (b) c = N/T = 0.12 gives nonlinear shrinkage something to do |
| Re-estimation | **Every 21 trading days (monthly), identical for every model** — see 2.5 | Silent confound if models re-fit at different rates |
| Horizon | 1-day ahead only (week 1) | Multi-step is week 3 |

**The m = 250 choice is load-bearing.** At m = 1000, c = 0.03 and DCC-NL collapses onto DCC — you
would ship a benchmark that handicaps your main econometric competitor. Report sensitivity to
m in {250, 500, 1000} in the final paper; use 250 as the headline.

---

---

# PART 1 — THE PRE-REGISTRATION (write as PREREGISTRATION.md, commit Day 2, never edit)

## 1.1 Model universe

See Part 2.4 for the 12-model roster. It is frozen before Day 4; adding a model after seeing
results reopens the data-snooping problem the SPA/MCS stage exists to close.

## 1.2 Data splits (time-ordered, no shuffling, ever)

```
[ burn-in: m days ][ VALIDATION ][ SCREEN ][ CONFIRM ]
                     tune only     MCS       DM + GW
```

- **VALIDATION** — hyperparameter selection ONLY. Never enters any reported result.
- **SCREEN** — Stage 1 MCS and Stage 2 finalist selection. Never used for confirmatory inference.
- **CONFIRM** — locked. Touched exactly once, at the end of Day 7. Enforce this in code: a
  `LOCKED = True` flag that raises unless an explicit `--unlock-confirm` argument is passed.

Target sizes: validation >= 250 days, screen >= 500 days, confirm >= 500 days. If the sample cannot
support this, shorten validation first, never confirm.

**Preprocessing (standardization, feature scaling, any winsorization) is fit on the estimation
window only and applied forward.** Global standardization is the most common silent leak in this
literature.

## 1.3 Seed protocol

- S = 5 seeds per stochastic model (XGBoost, MLP), pre-specified list `[0,1,2,3,4]`.
- Report the **full distribution** of losses across seeds.
- The pre-specified object entering `M_0` is the **seed ensemble**
  `Sigma_bar = (1/S) * sum_s Sigma^(s)`, which is PD whenever the components are.
- **No best-seed selection, ever.** Seed variation is algorithmic uncertainty, not forecast evidence.
- Declare in the paper that ensembling is a variance-reduction advantage deterministic econometric
  models do not receive; it is part of the *method* under GW's method-evaluation framing.

## 1.4 Equal tuning budget — including for the baselines

**K = 20 hyperparameter configurations per model**, evaluated on the validation block under the
primary loss, best config frozen. This applies to econometric models too:
- `EWMA`: lambda over a 20-point grid.
- `HAR-DRD`: 20 configs over lag structures {(1,5,22), (1,5,10), (1,3,10), ...} and optional ridge penalty.
- `LW-NL` / `DCC` / `DCC-NL`: fewer natural knobs; spend the budget on window/targeting variants so
  the *opportunity* is equal even if the model does not need it. Document unused budget.

Tuning the baselines with the same budget is the fairness move almost no DL paper makes. It is
cheap, and it is the first thing a hostile referee checks.

## 1.5 Inference sequence

Specified in full in **Part 3.5**. Summary: SPA (benchmark = `HAR-DRD`) then MCS on the screening
block; median-rank finalist rule; DM and GW on the locked confirmation block; MZ / augmented MZ-GLS
for calibration; GMV for economic value.

## 1.6 Economic channel

GMV weights `w = Sigma_hat^-1 1 / (1' Sigma_hat^-1 1)`, realized loss `w' Sigma_proxy w`.
Report: out-of-sample GMV variance, **turnover**, and variance **net of costs** at 5 and 10 bps.
Run the same DM and GW structure on the GMV loss differential. This is the precision-matrix channel
— a model can improve Sigma under Frobenius and worsen Sigma^-1, and portfolios consume the inverse.

## 1.7 Mandatory diagnostics (not optional decoration)

For every headline pairwise comparison, report:
- **Cumulative sum plot of d_t** across the evaluation period. Straight line = broad stable win.
  Staircase = regime-dependent. Single cliff = the whole result is one episode.
- **ACF of d_t** and the implied HAC variance-inflation factor kappa, with effective sample size
  `T_eff = T / kappa`.
- **Per-asset distribution of loss ratios**, not just the cross-sectional mean.

These three exhibits are what convert "our model wins" into "here is the temporal structure of the
win," which is the project's methodological signature.

---

---

# PART 2 — MODEL ROSTER

## 2.1 Why the MLP is gone

A small MLP on lagged RCov features is not a covariance model anyone published — it is a generic
regressor with no PSD structure, no econometric prior, and no paper behind it. Its only role was to
occupy the "nonlinear network" rung of the ladder. **LSTM-BEKK occupies that rung properly, with a
real architecture from a real paper.** The MLP adds a cell to the factorial that the RCov-augmented
LSTM-BEKK already covers, at the cost of a day you don't have. Cut.

## 2.2 LSTM-BEKK (Wang, Liu, Tran & Wang 2025) — the primary DL competitor

**What the paper actually specifies** [V, read this session]:
```
H_t = C C'  +  C_t C_t'  +  a · r_{t-1} r_{t-1}'  +  b · H_{t-1}
```
- `CC'` — static long-run intercept (the scalar-BEKK constant).
- `C_t C_t'` — **the innovation**: a time-varying intercept generated by the LSTM.
  `C_t = LowerTriangular(C̃_t)`, where `C̃_t` comes from the hidden state `h_t`; the LSTM processes
  `r_{t-1}` and `h_{t-1}`.
- `a, b` — scalars, constant, with `a, b >= 0`, `a + b < 1`. **Interpretability survives**: `a` is
  still reaction, `b` still persistence.
- Theorem 1 gives sufficient non-explosion conditions: beyond `a+b<1`, bound the maximum eigenvalue
  of `C_t C_t'`.

**Why PSD is free.** Every term is PSD — `CC'` and `C_tC_t'` are Gram matrices (PSD for *any*
network output, which is the whole point of the lower-triangular parameterization),
`r_{t-1}r_{t-1}'` is an outer product, `H_{t-1}` is PSD by induction — combined with nonnegative
weights. This is exactly the argument you derived for DCC and for BEKK's quadratic sandwich. No
projection, no clipping, no eigenvalue repair.

**Training** [V]: Gaussian negative log-likelihood, RMSprop, Cholesky for `logdet` and inverse
(numerical stability), gradient clipping, dropout 0.1-0.2, 3-5 hidden layers with hidden size =
input size, early stopping on validation NLL.

**The information-set problem you must fix.** The paper uses **daily log returns only** — no
intraday data. Compared head-to-head against your HAR-DRD (which consumes 5-min RCov), the DL model
would be starved, and the comparison would confound architecture with information. This is your
project's own critique running in reverse. **So run two variants:**

- **`LSTM-BEKK`** — faithful replication, daily returns only. Answers: does the published model,
  as published, beat the econometric frontier?
- **`LSTM-BEKK-RC`** — identical architecture, but the LSTM input is augmented with RCov summaries
  (log realized variances, average realized correlation, log realized quarticity). Answers: does
  the architecture win once given information parity?

The gap between these two is a direct measurement of **how much of any DL gain is information
versus architecture** — the single most valuable number your paper can report, and the axis your
factorial exists to isolate.

**Protocol deviation to declare.** The paper uses a fixed 70/15/15 split [V]. Your protocol requires
rolling re-estimation with fixed window `m` for GW validity. You will therefore re-estimate on the
common rolling schedule, which differs from the original. State this: it is a deliberate
protocol-equalization, not a replication failure.

**Cost:** ~1.5 days. The recursion is a custom differentiable RNN cell; the fiddly parts are
Cholesky-based NLL, truncated BPTT through the covariance recursion, `H_0` initialization, and
constraining `a+b<1` (parameterize via sigmoid on a logit pair).

## 2.3 XGBoost — the case for and against, decided

**The case against, stated honestly:**
1. **Trees cannot extrapolate.** They predict piecewise constants. Volatility in a crisis exceeds
   anything in the training window, and the tree returns its training maximum. HAR extrapolates
   linearly. For right-skewed, regime-shifting data this is a structural, not incidental, weakness.
2. **No joint structure.** Predicting `N(N+1)/2 = 465` entries with 465 independent models means
   nothing in the estimator knows those numbers jointly describe one matrix.
3. **Cost.** 465 models × rolling refits × tuning configs is not free.

**The case for, which wins:**
1. **It is the method that actually won in the closest comparable study.** Christensen, Siggaard &
   Veliyev find regularized regressions and tree-based methods outperform the HAR lineage for
   univariate RV. Excluding trees means excluding the incumbent ML winner and inviting the obvious
   referee question.
2. **It is the "did shallow ML already get the gain?" control.** If XGBoost captures most of any
   LSTM-BEKK advantage, the deep-learning story collapses — and that is a headline finding in your
   skeptical framing. You need this cell.
3. **Its predicted failure mode is a live test of your diagnostic machinery.** The extrapolation
   weakness should appear as **miscalibration in high-volatility regimes** in the augmented MZ-GLS
   test. Predicting a specific failure in advance and then detecting it validates the protocol.
   That is worth more than another model that merely performs.

**Verdict: keep, but restructure.** Do **not** run 465 models on log-matrix entries. Run it on the
DRD decomposition:
- **30 models** for log realized variances (one per asset, or pooled with asset-identity features).
- **1 pooled model** for correlation entries, with pair-identity and pair-summary features.
  Correlations are homogeneous across pairs in a way variances are not, so pooling is natural and
  cuts 435 models to one.

31 fits, not 465. And the variance/correlation split matches your DRD evaluation decomposition, so
you can report *where* trees help — which is the interesting question.

## 2.4 The roster (12 models)

**`M_Econ` (8)** — all cheap once the harness exists:
1. `RW` — trivial anchor
2. `EWMA` — lambda tuned
3. **`HAR-DRD`** — the workhorse benchmark and the SPA reference model
4. `HARQ-DRD` — quarticity-scaled daily coefficient (BPQ, generalized)
5. `LW-linear` — Ledoit-Wolf 2004 optimal delta
6. `LW-NL` — analytical nonlinear shrinkage (LW 2020, **not** QuEST)
7. `DCC` — sample-correlation targeting
8. **`DCC-NL`** — nonlinear-shrunk correlation targeting

Ladders inside the class: `HAR-DRD → HARQ-DRD` isolates measurement-error correction;
`LW-linear → LW-NL` isolates the nonlinear eigenvalue map; `DCC → DCC-NL` isolates shrinking the
targeting matrix.

**`M_DL` (4):**
9. `Ridge-DRD` — regularized linear on shared features
10. `XGBoost-DRD` — shallow-ML control (31 fits, see 1.3)
11. **`LSTM-BEKK`** — faithful, daily returns
12. **`LSTM-BEKK-RC`** — information parity

Ladder: `HAR-DRD → Ridge-DRD → XGBoost-DRD` isolates regularization then nonlinearity with
information fixed; `LSTM-BEKK → LSTM-BEKK-RC` isolates information with architecture fixed.

**Deferred to week 2-3:** `GHAR` (linear, ~1 day, belongs in `M_Econ`), `GPVar` (GluonTS, but the
MXNet backend is deprecated — budget a day for environment archaeology alone), `SpotV2Net`
(no confirmed public code), `ReSPDNet` (Riemannian, highest bar). Do not half-replicate a
competitor — that is precisely what you are criticizing others for.

## 2.5 The re-estimation cadence decision (this is what makes the week feasible)

GW validity requires a fixed finite window and **identical cadence across every model**. Retraining
an LSTM every 5 days over a 1000-day evaluation period is 200 fits and is not affordable.

**Set the common cadence to monthly (21 trading days) for all 12 models, with m = 250.**
Roughly 48 refits over the evaluation period. An LSTM-BEKK fit on 250 days of N=30 returns is small
— minutes, not hours. Everything else is faster. Parity preserved, budget survivable, and the
choice is defensible and pre-registered rather than convenient.

---

## 2.6 Shrinkage: use analytical nonlinear shrinkage, not QuEST

You asked about QuEST. Important practical correction:

- **QuEST** (Ledoit-Wolf 2015/2017) inverts the Marchenko-Pastur relation numerically —
  a nonconvex optimization per estimation date. Accurate, slow, fiddly.
- **Analytical nonlinear shrinkage** (Ledoit-Wolf 2020) is a **kernel estimator with a closed
  form** — orders of magnitude faster, essentially the same accuracy. [M]

**Use the analytical version.** Reference code is published by the authors (MATLAB/R, with Python
ports). Do not hand-write either one.

Also run **linear** shrinkage (Ledoit-Wolf 2004, optimal delta) as a separate model. The comparison
`LW-linear → LW-NL` isolates exactly what the nonlinear eigenvalue map buys, which is the same
question your "could AI learn the shrinkage map" agenda item asks.

**Validation before use:** simulate from a known Sigma, plot sample vs linear-shrunk vs
nonlinear-shrunk vs true eigenvalues. The characteristic picture — sample spectrum over-dispersed,
nonlinear shrinkage pulling the extremes toward truth non-uniformly — must appear. If it does not,
the implementation is wrong. This figure belongs in the paper.

## 2.7 DCC-NL specifics

The composition, stated as the thing to implement:
```
1. N univariate GARCH(1,1)          -> standardized residuals eps_t
2. C_hat = sample correlation of eps -> NONLINEAR-SHRINK IT -> C_NL
3. Q_t = (1-a-b) C_NL + a eps_{t-1} eps_{t-1}' + b Q_{t-1}
4. R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
5. Sigma_t = D_t R_t D_t
```
The contribution is step 2. Because `C_NL` enters with permanent weight `(1-a-b)/(1-b)`, its noise
would otherwise contaminate every forecast forever — which is why shrinking the *target* matters
more than shrinking each `H_t`.

Composite likelihood is unnecessary at N = 30-50; note in the paper that it is required at the
N = 1000 scale ELW demonstrate, and that your dimension therefore under-powers DCC-NL relative to
its designed regime. **This is the most important honesty statement in your model section.**

---

---

# PART 3 — EVALUATION PROTOCOL

## 3.1 Losses and tuning budget — ranking

**Primary — multivariate QLIKE / Stein:**
```
L_Q = tr(Sigma_hat^-1 @ Sigma_proxy) - logdet(Sigma_hat^-1 @ Sigma_proxy) - N
```
**Secondary — Frobenius:**
```
L_F = ||Sigma_proxy - Sigma_hat||_F^2
```
**Decomposition (report always):** split into a variance block and a correlation block via DRD.
Frobenius overweights large variances; a model that wins only on mega-cap variances is making a
different claim from one that improves dependence structure. No paper in your list separates these.

**Rule, pre-committed:** a conclusion counts only if it holds under **both losses across all three
proxies** (intersection, not union). Robust losses guarantee proxy-consistency, not agreement with
each other — disagreement is information about where errors concentrate, and must be reported, not
resolved by picking the favourable cell.

**Forbidden:** MAE, MAPE, HMSE, R^2 on log-vol, any loss in sigma-space, Cholesky-space, or
tangent-space coordinates. Patton (2011) and LRV (2013) are the authority; proxy unbiasedness dies
under the nonlinearity.

## 3.2 Calibration — the MZ family (your addition, and it is a good one)

**This is a separate question from ranking, and treating it separately is the right instinct.**
Ranking asks "who is closer"; calibration asks "is this forecast systematically biased, and where."
A model can win on average loss while being badly miscalibrated in a specific regime — which is
precisely the DL failure mode you suspect.

**Implementation for matrices** (per Patton-Sheppard [M]): do not run MZ element-by-element.
Draw K = 100 fixed random portfolios `w_k` (seeded, published) plus the GMV and equal-weight
portfolios, and run MZ on portfolio variances:

**Standard MZ:**
```
w_k' Sigma_proxy_t w_k  =  a + b * (w_k' Sigma_hat_t w_k) + e_t ,   H0: (a,b) = (0,1)
```

**Augmented MZ** (your calm-regime hypothesis, formalized):
```
w_k' Sigma_proxy_t w_k  =  a + b * (w_k' Sigma_hat_t w_k) + c' z_{t-1} + e_t ,   H0: (a,b,c) = (0,1,0)
```
with `z_{t-1}` the regime variables. Rejecting `c = 0` says: *known state information predicts this
model's forecast errors* — i.e. the model is miscalibrated conditional on regime. Run it with a
calm/turbulent indicator specifically, since that is your hypothesis.

**MZ-GLS:** the error variance scales strongly with the volatility level (heteroskedasticity is
severe, not incidental), so OLS MZ has inefficient estimates and misleading standard errors. Use
GLS weighting by an estimate of the conditional error variance, and report **HAC standard errors**
in both cases. Report the SEs explicitly — they are what licenses the calibration claims.

**Caveat to carry:** MZ has a known weakness under noisy proxies — the LHS noise inflates residual
variance and reduces power, and non-robust MZ variants (e.g. MZ on log-variances) inherit the
Patton problem. Keep MZ in variance space.

## 3.3 Correlation-change diagnostic (your idea — but as a diagnostic, not a ranking loss)

**The motivation is sharp:** a model that simply predicts the historical average correlation scores
respectably on levels while being useless for dynamics. Levels-based losses partly hide this.

**Why it cannot be a ranking loss:** differencing breaks the Patton robustness argument. The target
becomes a difference of latent quantities, and the proxy noise enters twice with amplified
variance, so a "change loss" is not proxy-consistent. Ranking models by it would reintroduce
exactly the failure mode the whole protocol exists to prevent.

**So run it as calibration, in the MZ frame:**
```
Delta R_proxy_t  =  a + b * Delta R_hat_t + e_t ,   H0: (a,b) = (0,1)
```
on the average pairwise correlation and on the vech of the correlation change. Report:
- MZ slope on correlation changes (a model predicting the average has slope ~ 0);
- directional accuracy of correlation-change sign;
- correlation between predicted and realized `Delta R`.

A model with strong level performance and near-zero change-slope is capturing the unconditional
correlation structure and nothing else. Exposing that is a genuine contribution — the DRD
decomposition already separates variance from correlation, and this separates correlation *level*
from correlation *dynamics*.

## 3.4 Screening: SPA and MCS (both, they answer different questions)

- **SPA (Hansen 2005)** with `HAR-DRD` as the fixed benchmark: *does anything in the universe beat
  the workhorse, accounting for having searched over all of them?* This is the direct descendant of
  Hansen-Lunde's "nothing beats GARCH(1,1)" and it is the single most quotable headline your paper
  can produce, in either direction.
- **MCS (Hansen-Lunde-Nason 2011)**: *which models cannot be distinguished from the best?* This is
  the screening device feeding Stage 2.

Use SPA's studentization and recentering — it is less conservative than White's Reality Check when
the universe contains poor models, which yours does by construction (`RW` is in there).

Anticipate that MCS may retain a large set at N = 30 with correlated forecasts. That is honest, not
a failure; say so in advance.

## 3.5 Full inference sequence

```
M_0 (11 models)
  --SPA(screen, benchmark=HAR-DRD)-->  "does anything beat the workhorse?"
  --MCS(screen)-->                     superior set
  --pre-specified rule-->              1 DL finalist + 1 Econ finalist
  --DM(confirm)-->                     average superiority
  --GW(confirm)-->                     state-dependent superiority
  --MZ / augmented MZ-GLS-->           calibration, overall and by regime
  --GMV + turnover + costs-->          economic value, same DM/GW structure
```

Screen and confirm blocks are chronologically separated; confirm is locked in code.

**GW instruments — three, pre-specified:**
```
z_{t-1} = [ 1 , log(average realized variance)_{t-1} , average realized correlation_{t-1} ]
```
A second, separately pre-registered GW test with `[1, log RQ_{t-1}, jump indicator_{t-1}]` targets
the measurement-quality channel, with Bonferroni across the two tests. Do not stack six collinear
instruments into one chi-square — the degrees of freedom cost exceeds the information gained.

**GW validity:** requires a fixed finite estimation window. `m` and the re-estimation cadence must
be **identical across every model**, including GPVar. If GPVar cannot be refit on a 250-day rolling
window at acceptable cost, either accept a coarser common cadence for everyone or drop the GW claim
for that model and report a descriptive fluctuation analysis instead.

## 3.6 Mandatory diagnostics

- Cumulative `sum d_t` plot for every headline pair (straight line vs staircase vs single cliff).
- ACF of `d_t` and the HAC inflation factor kappa, with `T_eff = T / kappa`.
- Per-asset and per-universe (variant C) distributions of loss ratios.
- Seed distributions for stochastic models, never a best seed.

---

---

# PART 4 — THE WEEK (Tier 1 must ship by Day 5)

| Day | Work | Gate — do not proceed until this passes |
|---|---|---|
| **0** | WRDS/Databento decision; read LPS (2015); skim LSTM-BEKK section 2.2 | Data source settled |
| **1** | Cleaning, synchronization, 2 proxies, PD/conditioning logs; stylized-fact + **Epps** validation | Epps curve rises and flattens |
| **2** | Splits + confirm lock; losses (Stein, Frobenius, DRD, GMV); simulated DCC DGP with known Sigma | **Non-robust loss demonstrably flips the ranking** on simulated data |
| **3** | Rolling driver; DM+HAC, MCS, GW, standard MZ | **Naive t over-rejects ~25% vs HAC ~5%; GW catches regime dependence DM misses** |
| **4** | `RW`, `EWMA`, `HAR-DRD`, `HARQ-DRD`, `Ridge-DRD` end-to-end | Full results table on the screening block |
| **5** | `LW-linear`, `LW-NL`, `DCC`, `DCC-NL`; eigenvalue figure; **unlock confirm once**; README + PREREGISTRATION | **TIER 1 SHIPPED — repo is public-ready** |
| **6-7** | Tier 2, in priority order: `LSTM-BEKK`, then `XGBoost-DRD`, SPA, augmented MZ-GLS | Whatever finishes, finishes |

## Slippage rules (decide now, not at 2am on Day 4)

- **Behind at end of Day 2** → drop the second proxy, keep `RCov_5min` only. Rank-stability moves
  to Tier 3.
- **Behind at end of Day 3** → drop GW and MZ from Tier 1, keep DM + MCS. GW moves to Day 6.
- **Behind at end of Day 4** → drop `HARQ-DRD` and `LW-linear`. The `DCC → DCC-NL` pair is the
  one that must survive; it is the frontier baseline and the reason the project is interesting.
- **Day 6 LSTM-BEKK not training by evening** → stop, commit, ship Tier 1, resume in week 2. Do not
  spend Day 7 debugging a model at the cost of the README.

## The two hours most people skip and shouldn't

The three synthetic demonstrations on Days 2-3 are unit tests *and* paper figures *and* the
strongest signal in the repo that you understand why the protocol exists rather than having copied
it. They are also how you find out your HAC and MCS wiring is wrong before real data hides it.

# PART 5 — REPO AND README

```
covbench/
  PREREGISTRATION.md        <- committed Day 2, never edited
  data/                     <- hashed rcov arrays + build script
  covbench/
    data/       loaders, synchronization, rcov estimators
    models/     one class per model, common fit/predict interface
    losses.py   stein, frobenius, drd decomposition, gmv
    inference/  dm.py, mcs.py, gw.py, bootstrap.py
    protocol.py splits, confirm-lock, rolling estimation driver
  tests/                    <- the synthetic-truth tests from Days 2-3
  results/
  notebooks/                <- diagnostics, stylized-fact validation, Epps curve
```

**README framing — this is what a quant reader actually evaluates.** Lead with the protocol, not
the models:

> A leak-free, pre-registered benchmark for multivariate covariance forecasting. Nine models
> (random walk through DCC-NL and neural networks) compared under proxy-robust matrix losses across
> three realized-covariance proxies, with Model Confidence Set screening, chronologically separated
> confirmatory Diebold-Mariano and Giacomini-White tests, and a global-minimum-variance economic
> channel net of costs. Includes the tests demonstrating why the common evaluation shortcuts in this
> literature — non-robust losses, seed-level significance, unconditional-average comparison — give
> the wrong answer.

Four baselines evaluated impeccably beat nine models evaluated casually. The three synthetic
demonstrations (non-robust loss flips ranking; naive t-test over-rejects at 25%; DM blind to regime
dependence where GW is not) are the most distinctive thing in the repo — they show you know why the
protocol exists.

---

---

# PART 6 — "WHERE SIMPLE AI IS WELL-POSED" (your last question — and it is the paper's future)

The framing that makes this publishable: rather than asking "can a big network beat econometrics,"
identify **narrow sub-problems where the econometric solution is an explicit analytical map, so a
learned map has a precise baseline and a precise constraint set.** Four candidates, ranked by how
well-posed they are:

**1. Learned nonlinear eigenvalue shrinkage. (Best candidate.)**
Ledoit-Wolf's nonlinear shrinkage is a map
`(lambda_1..lambda_N, N/T) -> (d_1..d_N)` derived under asymptotic assumptions (Marchenko-Pastur,
no time-series dynamics, no fat tails). A learned map can condition on things the analytical
derivation cannot: realized quarticity, recent turbulence, cross-sectional dispersion, sample
non-stationarity.
Why it is well-posed: acts only on eigenvalues, so it is **rotation-equivariant by construction**;
PD is guaranteed if outputs are positive; the analytical estimator is a strong, published baseline;
the training target is available (simulate from known Sigma, or use out-of-sample GMV variance as
the loss). The whole thing is a small MLP on N inputs.
Referee guard: must beat analytical NLS out-of-sample, under the same rolling-window discipline,
without peeking across the time split.

**2. Learned model switching (the GW decision rule, estimated).**
GW already yields `choose g if h_T' alpha > c`. Replace the linear rule with a learned classifier on
richer state features. This *directly operationalizes* the regime-dependence outcome: instead of
"results are mixed," you ship a switching rule with a formal test that switching adds value.

**3. Learned measurement-error correction (HARQ, generalized).**
HARQ's attenuation correction is `beta_1 + beta_1Q * sqrt(RQ)` — a hand-specified linear map from
one reliability statistic to one coefficient. Learn the map from a richer reliability vector
(RQ, jump statistics, liquidity, sampling-grid disagreement) to a full correction.

**4. Jump-day and event handling.**
The Rahimikia-Poon finding is that ML's edge concentrates on jump days where HAR is structurally
weakest. A model that is *only* invoked on flagged days, with HAR elsewhere, is a small, testable,
economically motivated hybrid.

Common discipline for all four: SPD constraints respected, permutation/rotation invariance where
required, strict temporal separation, evaluation under the same robust losses and the same
inference sequence. "Learned" is not automatically better; each must beat its analytical ancestor
under the protocol.

---

---

# PART 7 — CUT ORDER

1. Universe variant (C) → 2. `XGBoost-DRD` → 3. `LSTM-BEKK-RC` (keep the faithful one) →
4. third proxy → 5. `LSTM-BEKK` entirely (ship 10 econometric+linear models, add in week 2) →
6. `DCC-NL` (ship `LW-NL` and `DCC` separately, compose in week 2).

**Never cut:** the confirm lock, `PREREGISTRATION.md`, the Epps check, SPA/MCS, the
cumulative-`d_t` diagnostic, or the both-losses requirement.

---

# PART 8 — LIMITATIONS TO STATE

- N = 30, 1-day horizon. **DCC-NL is under-powered at this dimension** — its advantage grows with
  N, and ELW demonstrate at N = 1000. This is the most important honesty statement in the paper.
- `LSTM-BEKK` re-estimated on a rolling schedule, deviating from the paper's fixed 70/15/15 split.
- No `GHAR`, `SpotV2Net`, `GPVar`, or `ReSPDNet` yet.
- Composite likelihood omitted (unnecessary at N=30-50, required at N=1000).
- Single market, single sample period; regime coverage limited to what the sample contains.

