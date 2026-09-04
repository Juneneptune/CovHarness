# Project Ledger — Skeptical DL-vs-Econometrics Covariance Benchmark

Last updated: 2026-08-05. Re-upload this to the project to persist state.

---

# PART A — SAVED CONCEPTS (explicitly flagged as paper-relevant)

## SAVED CONCEPT 1 — Realized Quarticity (flagged during BPQ/HARQ)

**Definition.** `RQ_t = (M/3) * sum_j r_{t,j}^4` — scaled sum of *fourth* powers of intraday
returns. Consistently estimates integrated quarticity `IQ_t = ∫ σ⁴(s) ds`.

**Why it matters (one sentence).** RV is a noisy estimate of latent volatility and *how noisy
varies day to day*; RQ is the estimate of that noise. Barndorff-Nielsen–Shephard asymptotics give
`Var(RV_t - IV_t) ≈ (2/M) * IQ_t`, so `sqrt(RQ_t)` is (up to constants) the standard deviation of
RV's own measurement error on day t.

**Three roles, in order of importance:**
1. Makes measurement error *observable and time-varying*. Converts "RV is noisy" from a static
   caveat into a daily measurable quantity. The conceptual pivot of BPQ (2016). Analogue of the
   Andersen–Bollerslev move: as RV made latent volatility observable, RQ makes RV's *reliability*
   observable.
2. Engine of attenuation-aware forecasting (HARQ). Errors-in-variables attenuation is proportional
   to error variance, and error variance ∝ RQ, so undo it day by day:
   coefficient = `β1 + β1Q * sqrt(RQ_{t-1})`, with `β1Q < 0`. Still linear-in-parameters → plain OLS.
   Key result: the gain is not only on noisy days — segregating them frees the baseline coefficient
   to be more aggressive on the clean majority (HAR's single constant coefficient is a compromise
   dragged down by bad days).
3. General reliability weight, extensible beyond HAR — including, relevant to this project, the
   multivariate/realized-covariance analogues where entries have heterogeneous measurement errors.

**Referee cautions that MUST accompany any use:**
- RQ is a *fourth-moment* estimator → itself very noisy and heavy-tailed (Cont's moment ladder:
  near the tail-index boundary, fourth-and-higher sample moments barely converge). The correction
  introduces its own measurement error.
- RQ is corrupted by jumps and microstructure noise — fourth powers amplify both far more than
  squares. Jump-robust quarticity estimators (tri-power etc.) exist and matter for jumpy assets.
- Evaluation remains proxy-based. RQ improves how you *use* the RV proxy; it does not make the
  target non-latent.

**Intro-ready sentence.** *Realized quarticity is the high-frequency estimate of realized
volatility's own measurement-error variance; it turns the errors-in-variables problem in RV
forecasting from an unmodeled nuisance into a daily, exploitable signal — at the cost of being a
noisy, jump-sensitive fourth-moment estimator in its own right.*

---

## SAVED CONCEPT 2 — Three distinct failure modes when DL "beats" econometrics
### (flagged during Diebold–Mariano; core methodological contribution of this project)

**User's originating insight:** DL models may show longer *streaks* of beating conventional models,
which makes them look better than they are.

**The insight is right, but it fuses three separate problems that need three separate defenses.**

### (i) Streakiness — a HAC / effective-sample-size issue
Serially correlated loss differentials `d_t` mean your T out-of-sample days contain far fewer than
T independent verdicts. Long-run variance = `γ(0) + 2·Σ γ(τ)` = `2π f_d(0)`; ignoring it inflates
false positives dramatically (simulated: 25.7% rejection rate at a nominal 5%, AR(1) ρ=0.5).
- **Defense:** HAC-corrected DM / Giacomini–White.
- **Diagnostics to report:** ACF of `d_t`; **cumulative sum plot of `d_t`** over the out-of-sample
  window. Straight line = broad, stable win. Staircase = regime-dependent. Single steep drop =
  the entire "win" came from one episode (e.g. March 2020) — the most common and most damning case.

### (ii) Leakage — a PROTOCOL issue that no statistical test can detect
If a model has peeked at the future (lookahead in feature construction, normalization computed over
the full sample, hyperparameters tuned on the test period), it beats the benchmark *consistently
and non-streakily* — and DM will award it a beautifully significant p-value.
**A leaky model looks like a great model to every forecast-comparison test in existence.**
- **Defense:** strict temporal splits; out-of-sample normalization; validation block strictly
  preceding the test block. Protocol, not post-hoc statistics.

### (iii) Data snooping — a MULTIPLE-TESTING issue that HAC does not touch
Try twenty architectures, report the best one's DM test → inflated significance.
- **Defense:** White (2000) Reality Check, Hansen (2005) SPA, Hansen–Lunde–Nason MCS.

### The asymmetry that is genuinely publishable
DL models plausibly *do* produce blockier loss differentials than econometric ones: a high-capacity
model fit on a particular regime excels while that regime persists and degrades when it shifts —
its edge is regime-shaped by construction, where HAR's three coefficients are too rigid to
specialize. **So the same nominal average improvement carries less evidential weight for the DL
model.** This asymmetry is a reason the benchmark should report the temporal structure of every win,
not just its average.

**Design principle for the paper:** *report the temporal structure of the win, not just its average.*

---

# PART B — RELATED STANDING DECISIONS (from this session)

**On fairness of cross-paradigm comparison** (arose from "why don't HAR papers compare to ARCH?"):
The claim "RV beats GARCH" (ABDL 2003) is an **information-set** result, not an architecture
result — GARCH is fed daily squared returns, HAR is fed intraday-based RV. ABDL themselves write
that the findings "do not reflect a failure of the GARCH model per se, but rather the efficacy of
exploiting volatility measures based on intraday return observations." Feed GARCH the realized
measure (Realized GARCH, GARCH-X) and the gap largely closes.

Consequences for this project's design — the benchmark must declare which question it answers:
1. **Matched information sets.** Decide explicitly whether the comparison is "best model given each
   one's *standard* information set" (paradigm comparison) or "best model given a *common*
   information set" (architecture comparison). If the latter, vanilla GARCH is a strawman and
   Realized GARCH / GARCH-X are the fair ARCH-family representatives.
2. **Proxy-robust loss is mandatory, not optional.** Cross-paradigm comparison means the two model
   families have different native targets (`r_t²` vs RV) and *no neutral proxy exists*. Only a
   Patton-robust loss gives a ranking invariant to proxy choice. Report MSE **and** QLIKE.
3. **Matched tuning budgets.** Equalize hyperparameter search / early stopping, or the "win" is a
   tuning win misattributed to architecture.
4. **Note the QLIKE coherence wrinkle:** QLIKE = the Gaussian likelihood = the objective GARCH/DCC
   were *estimated* under. Scoring an MSE-trained network with QLIKE is not automatically neutral.
   Consider training DL under both objectives.

**Referee question this generalizes to:** not *which model won*, but *were the models given the same
information and the same tuning budget?*

---

**On why GW rather than DM/West for this project** (the argument that actually survives scrutiny):

The binding reason is **not** that nested-model DM has a large size distortion. Simulation this
session (AR(1) truth, AR(1) vs AR(2), squared loss) found the distortion is real but *moderate*:
Var(ΔL) degenerates as predicted (0.018 → 0.0006 as the estimation sample grows), and when the
out-of-sample fraction π = n/T_est is bounded away from zero the DM statistic is systematically
**shifted** (mean −0.46 at π=0.2 → −0.90 at π=3, sd 0.76 instead of 1), rejecting ~8.5% at a
nominal 5%. Direction is one-sided: it favors the *smaller* model, because the big model pays a
real estimation cost for a genuinely-zero parameter. Notably, when π → 0 (fixed small evaluation
window, growing estimation sample) standard DM is approximately fine even for nested pairs.

**The real reason is that DM/West's null hypothesis presupposes objects a neural network does not
possess.** West's null is stated in terms of probability limits β*, and the variance correction
needs `F = E[∂ΔL/∂β]` and `B = avar(√n(β̂−β*))`. Requirements: (1) a unique probability limit,
(2) √n-asymptotic normality, (3) differentiability in a finite-dimensional β. Where each fails:
- **Non-parametric:** converges at n^(−2/(4+d)), not √n → B infinite; and β is a function, not a
  vector → F undefined.
- **Bayesian shrinkage:** shrinkage is deliberately non-negligible; and two estimators with the
  same probability limit are *declared identical* by West's null — so the framework is structurally
  incapable of comparing estimation procedures (OLS vs ridge on the same model).
- **Neural nets:** non-convex objective, initialization/batch-order dependence, early stopping,
  permutation non-identifiability → **no β\* exists at all**. Every ingredient undefined.

GW never differentiates through the estimator and never takes a limit in the parameter direction:
with m fixed, the forecast is just `f̂_{m,t} = Φ(y_{t−m+1},…,y_t)`, some measurable function of the
last m observations. **All asymptotics move to the out-of-sample direction**, so constraints land
on the data (mixing, finite moments of ΔL) rather than on the estimator.

**Caveat recorded for honesty (raised by user, correct):** the classic nested-model degeneracy
argument (β̂_B → (β*_A, 0) ⇒ forecasts converge ⇒ ΔL degenerates) is a *theorem only under
identifiability + consistency*. It is a conditional result, not automatic. Three cases:
1. Identified & smooth (AR, HAR/HARQ) → degeneracy holds; nonstandard limit.
2. Unidentified but observationally equivalent (NN permutation symmetry) → parameters never
   converge, but forecasts do → degeneracy still holds.
3. Unidentified with genuinely distinct forecasts (multiple minima + early stopping) → nothing
   converges, no degeneracy — but then **West's null is not even well-defined**, which is worse.
DM is invalid in all three; the *mechanism* differs. Do not overclaim the degeneracy argument.

**Scope note — GW does not dominate West.** They answer different questions. West's unconditional
null ("which specification is closer to the DGP") is the right scientific question; GW's
("which procedure will forecast better next period") is the right engineering question. This
project asks the engineering question, so GW fits — that is a *matching* argument, not a
superiority claim. Also note the fixed-m device is an asymptotic fiction chosen for tractability
(nobody believes m stays at 500 as T→∞); its defense is that it is a *less unrealistic* fiction
than "estimation error vanishes." Rolling windows themselves are old (Fama–MacBeth 1973); what is
new is fixing m in the *limit theory* so theory matches practice. Describe it that way in the paper.

---

## SAVED CONCEPT 3 — Giacomini–White test functions (instruments) as regime probes
### (flagged during GW 2006; this is the intended primary evaluation tool for the benchmark)

**The null being tested.** `H0: E[ΔL_{m,t+1} | F_t] = 0 a.s.` — "given everything known today, you
cannot predict which method will forecast better tomorrow." Strictly stronger than DM's
unconditional `E[ΔL] = 0` ("equal on average").

**The theorem that makes it testable.** A conditional moment restriction is equivalent to
infinitely many unconditional ones:

    E[X | F_t] = 0  a.s.   <=>   E[h_t · X] = 0  for every F_t-measurable h_t

- (=>) tower property: `E[h_t X] = E[ h_t E[X|F_t] ] = 0`, since h_t is known at t and pulls out.
- (<=) take `h_t = E[X|F_t]`; then `E[(E[X|F_t])²] = 0`, forcing `E[X|F_t] = 0`.

Since infinitely many can't be tested, pick a finite q×1 vector `h_t` — the **test function**.
This weakens the null to only the directions probed. That is the entire art.

**The statistic.** With `Z_{m,t+1} = h_t ΔL_{m,t+1}`:

    T = n · Zbar' Ω̂⁻¹ Zbar   ~  χ²_q      where  Ω̂ = (1/n) Σ Z Z'

Equivalently (Corollary 3): **n·R² from regressing ΔL on h_t**, compared to χ²_q. Same auxiliary-
regression logic as Engle's ARCH LM test — run a regression under the null; if it explains
anything, the null is false.

**Why no HAC correction is needed (one-step).** Under the *conditional* null, `Z_{m,t+1}` is a
martingale difference sequence (`E[Z_{t+1}|F_t] = h_t · 0 = 0`), hence serially uncorrelated, so
the long-run variance collapses to the plain outer-product average. Stronger null ⇒ simpler
variance. (For τ-step forecasts ΔL is MA(τ−1) even under the null → reinstate weighted lags.)

**What actually converges.** ΔL itself never converges — it fluctuates forever, which is the point
of fixing m. What converges is the *distribution* of the scaled sample average (a CLT), requiring
only mixing + finite (2+δ) moments **of the loss series** — assumptions about data, not about the
estimator.

**Menu of instruments for this project** (each is a different question):

| h_t | Question |
|---|---|
| 1 | Is one better on average? — **this is exactly DM** (DM = GW with q=1, h=1) |
| (1, ΔL_t) | Is relative performance streaky / autocorrelated? |
| (1, RV_t) or (1, VIX_t) | Does the volatility regime predict the winner? ← regime-dependence outcome |
| (1, sqrt(RQ_t)) | Does the DL model win when RV is poorly measured? |
| (1, recession/crisis dummy) | Does relative performance flip across the cycle? |
| (1, cross-sectional dispersion) | Multivariate analogue: does correlation dispersion predict the winner? |

**The power trap this avoids — worked example.** If the network beats HAR by 0.5 in turbulence and
loses by 0.5 in calm, half the time each, the unconditional mean is exactly 0 and **DM concludes
"equally good."** Verified by simulation (n=1000, persistent regime, 2000 reps):
- DM (h=1): rejects 30% — and this is NOT power. It is detecting sample-specific regime *imbalance*,
  so **the sign of its verdict flips depending on which regime dominated the sample window.**
  Different sample periods give opposite "significant" answers.
- GW with h=(1, regime indicator): rejects **100%**.
Lesson: under regime-dependent relative performance an unconditional test doesn't merely lose power
— it yields sample-dependent, sign-unstable conclusions that *look* significant. Worse than no answer.

**Instrument selection discipline.**
- Too few / wrong ones → test blind to structure in plain sight ("incorrectly accepts a false null").
- Too many irrelevant ones → each adds a df, raising the χ² critical value (χ²₁ 95% = 3.84;
  χ²₁₀ = 18.3) with no added signal → power falls.
- **Pre-specify instruments before looking at results.** Fishing over instruments until something
  rejects is data snooping (failure mode (iii) in Saved Concept 2).

**Bonus: GW yields a decision rule, not just a test.** Regress `ΔL_{m,t+τ}` on `h_t` over the
out-of-sample period → coefficient `α̂`. Then at time T: choose method g if `h_T'α̂ > c`, else f.
Report `I_{n,c}` = fraction of dates the rule picks g. **This is the constructive version of the
regime-dependent outcome**: a real-time switching rule between HAR and the DL model, with a formal
test that switching adds value — considerably more publishable than "results are mixed."

**Design constraint to respect.** GW validity requires a **fixed, finite** estimation window m.
An expanding window (common in DL practice, "train on all history") is *not* GW-valid. Also, m is
part of the method — HAR with m=500 and HAR with m=2000 are different methods; report sensitivity.

---

## SAVED CONCEPT 4 — The staged inference design
### (developed in ChatGPT sessions; recorded here with referee critique from this session)

**The sequence:**
```
M_0 (pre-committed universe)
  --MCS(screen)-->        superior set
  --pre-specified rule--> 1 DL finalist + 1 Econ finalist
  --DM(confirm)-->        average superiority
  --GW(confirm)-->        state-dependent superiority
```
with `T_eval = T_screen ∪ T_confirm`, disjoint, all screen dates preceding all confirm dates.
Losses: multivariate QLIKE/Stein primary, Frobenius secondary. Seed variation treated as
algorithmic uncertainty; pre-specified seed ensemble `Σ̄ = (1/S)Σ_s Σ^(s)`, PD whenever components
are. GMV economic channel repeats the same DM/GW structure.

**The strongest idea in it:** the chronological screen/confirm split. Running MCS and then DM on the
*same* out-of-sample data is double-dipping — testing a data-selected pair as though specified ex
ante. Almost nobody in this literature separates the two.

**Referee critique (raised this session — fold these into the paper):**
1. **Stage 2 changes what the headline claim can be.** `argmin` over screening losses compares
   *the DL model that happened to look best in the screening period* against the econometric
   equivalent — not a class-level comparison. Asymmetric: if DL rankings are less stable across
   regimes (which the regime-blockiness argument predicts), the screen-selected DL finalist is more
   likely to be the wrong pick on confirm. **Fix adopted:** select by *median rank across four
   equal sub-blocks* of the screening period, not by mean loss; report mean-loss version as a
   robustness check.
2. **Power.** Halving the OOS sample then running χ²₆ on the second half is underpowered, and the
   confirm block may be a single regime. **Fix adopted:** three instruments max per GW test.
3. **Instrument collinearity.** High-vol, jump, dispersion, quarticity, and average-correlation
   indicators all spike together in crises — they add df (χ²₁ = 3.84 vs χ²₆ = 12.6) without adding
   independent directions. **Fix adopted:** two separately pre-registered GW tests (regime channel;
   measurement-quality channel) with Bonferroni across them.
4. **GW requires fixed finite m.** Expanding-window training (standard DL practice) is not
   GW-valid. m and re-estimation cadence must be identical across every model.
5. **Stein loss needs a nonsingular proxy.** `logdet(Σ̂⁻¹Σ_proxy)` = `logdet Σ_proxy − logdet Σ̂`,
   so a singular RCov gives −∞. Constraint: RCov rank ≤ M (intraday returns/day). Decide the
   dimension cap in advance; do not patch with eigenvalue clipping later.
6. **"Robust across proxies" is rank-stability evidence, not satisfaction of the theorem's
   condition** (which requires conditional unbiasedness — Epps bias breaks it upstream).
   Pre-commit to the intersection rule: conclusions must hold across all losses × proxies.
7. **Nesting.** If finalists are nested, DM is invalid → Clark–West adjustment. Verify and record.
8. **Anticipate a near-vacuous MCS** at moderate N with correlated forecasts. Honest, but say so
   in advance rather than being surprised.

---

## SAVED CONCEPT 5 — DCC-NL and nonlinear shrinkage
### (developed in ChatGPT sessions; implementation guidance added this session)

**Why sample eigenvalues are the problem.** For `S = UΛU'`, when `c = N/T` is not tiny, finite-
sample noise disperses the spectrum even if the true covariance is simple: large sample eigenvalues
too large, small ones too small. Portfolio optimization then exploits those errors through `S⁻¹` —
"error maximization."

**Linear vs nonlinear.** Linear: `Σ̂ = δF + (1−δ)S`, one common shrinkage structure.
Nonlinear works **spectrally**: `Σ̂_NL = U diag(d_1,…,d_N) U'` with `d_i = f(λ_i, spectral info, N/T)`.
**"Nonlinear" refers to the map from sample eigenvalues to cleaned eigenvalues — NOT to the DCC
time-series dynamics.** Different eigenvalues get different corrections: large noisy ones shrunk one
amount, middle ones adjusted differently, badly underestimated small ones raised substantially.

**Why DCC-NL shrinks the *targeting* matrix.** Not "forecast DCC then shrink every H_t." In
`Q_t = (1−a−b)C + a·u_{t−1}u_{t−1}' + b·Q_{t−1}`, C is the attractor. It re-enters **every period**
with permanent weight `(1−a−b)/(1−b)` — so its noise never decays out (unlike the seed `Q_0`, whose
influence decays as `b^t`). Improving C improves the entire forecast path.

**Composite likelihood** solves a *different* problem: computational scalability of the dynamic
parameters at large N, via lower-dimensional (pairwise) pieces rather than a full N-dimensional
likelihood. DCC-NL combines the two. Not needed at N = 30–50; required at ELW's N = 1000.

**Implementation correction (this session): use analytical nonlinear shrinkage, not QuEST.**
QuEST (Ledoit–Wolf 2015/2017) inverts the Marchenko–Pastur relation numerically — a nonconvex
optimization per date, accurate but slow and fiddly. **Ledoit–Wolf (2020) analytical nonlinear
shrinkage is a closed-form kernel estimator**, orders of magnitude faster, essentially the same
accuracy [M]. Use published reference code; do not hand-write either.

**Validation before use:** simulate from a known Σ, plot sample vs linear-shrunk vs nonlinear-shrunk
vs true eigenvalues. Sample spectrum over-dispersed; nonlinear shrinkage pulling extremes toward
truth non-uniformly. If that picture does not appear, the implementation is wrong. Figure belongs
in the paper.

**Could AI learn the shrinkage map?** Conceptually yes — learn
`(λ_1,…,λ_N, features) → (λ̃_1,…,λ̃_N)`. Well-posed because it acts only on eigenvalues (rotation-
equivariant by construction), PD guaranteed with positive outputs, and the analytical estimator is a
strong published baseline. Must beat analytical NLS out-of-sample under the same rolling-window
discipline, respecting SPD constraints, permutation/eigenbasis invariance, and temporal separation.
"Learned" is not automatically better. **This is the strongest candidate for the project's
AI-extension section.**

---

# PART C — PAPER LEDGER

Status key: **Guided** = walked through section-by-section this session, grounded in the PDF.
**Interrogated** = user answered checkpoint questions and passed. **Feynman** = user explained it
closed-book and it was graded. Per project protocol a paper is not "strong understanding" until
Feynman is passed.

| ID | Paper | Guided | Interrogated | Feynman | Notes |
|---|---|---|---|---|---|
| F01 | Cont (2001), Stylized Facts | ✅ | ✅ | ⬜ | 11 facts; ergodicity; moment ladder; time deformation |
| H01 | Andersen & Bollerslev (1998), Answering the Skeptics | ✅ | ✅ | ⬜ | R² ceiling `1/(κ−1)`; RV fix; proxy-noise founding insight |
| F02 | Engle (1982), ARCH | ✅ | partial | ⬜ | Theorem 1 moments; LM test; UK inflation. **PDF is a scan — OCR'd this session** |
| M03 | Engle (2002), DCC | ✅ | ✅ | ⬜ | Deep: recursion, normalization, PSD proof, full training pipeline |
| H10 | Corsi (2009), HAR | ✅ | ⬜ | ⬜ | 6 interrogation questions delivered, **user has not yet answered** |
| H11 | Bollerslev–Patton–Quaedvlieg (2016), HARQ | ✅ | ⬜ | ⬜ | Attenuation + realized quarticity (see Saved Concept 1) |
| E12 | Patton (2011), Robust Loss | ✅ | ⬜ | ⬜ | Robustness ⟺ curvature-in-proxy independent of forecast; MSE & QLIKE unique |
| E02 | Diebold–Mariano (1995) | ✅ | ⬜ | ⬜ | Loss differential; HAC long-run variance. **PDF is a scan — OCR'd this session** |
| E06 | Giacomini–White (2006) | ✅ | ⬜ | ⬜ | Conditional predictive ability; fixed-m; test functions (Saved Concept 3) |

**Discussed but not read as primary sources:**
- ABDL (2003), Modeling & Forecasting RV — understood via consequences (Corsi's intro + web-verified
  abstract/text). Uses **30-minute** returns, DM/$ and Yen/$, plain sum-of-squares RV, no
  microstructure correction. Not read in full; not in project files (see file issues below).
- Bollerslev (1986) GARCH — equations verified via AB98 eqs. 2–3, not from the primary PDF.
- Engle–Kroner (1995) BEKK — read this session for the DCC pre-history (quadratic-form PSD trick,
  Prop 2.5). Not separately interrogated.

**Next in queue (evaluation spine):**
- E06 Giacomini–White (2006) — conditional predictive ability; fixes DM for *estimated* models and
  nested comparisons. Highest leverage next, and directly serves the regime-dependence hypothesis.
- E07 White (2000) Reality Check → E08 Hansen (2005) SPA → E09 Hansen–Lunde–Nason MCS —
  the multiple-model / data-snooping arc (defense (iii) above).
- E13 Patton–Sheppard (2009) — multivariate robust losses. Required, since this project is
  multivariate and Patton (2011) is univariate.
- E15 Laurent–Rombouts–Violante — multivariate loss analogue.

---

# PART D — FILE HEALTH ISSUES (found by diagnostics this session)

- **F03_Bollerslev_1986_GARCH.pdf — BROKEN.** Contains only the cover page of an EERI reprint;
  the paper itself is missing. Needs re-download.
- **H02_ABDL_2003_ModelingRV.pdf — WRONG PAPER + unreadable.** It is a Kalman-filter NBER working
  paper, not the JBES "Modeling and Forecasting Realized Volatility," and its text layer is
  encrypted-font gibberish. Needs replacement if the primary source is wanted.
- **F02 (Engle 1982) and E02 (Diebold–Mariano 1995) are image-only scans** with no text layer.
  Readable via OCR (done this session); quality good, equations mangled in places.
- **Christensen, Siggaard & Veliyev (2023)** — not in project files at all (manifest says LIB/AUT).
- All other files verified readable.

---

# PART E — RUNNING GLOSSARY OF FRAMING DECISIONS

- **Proxy-aware evaluation:** robust-proxy (MSE/QLIKE/Frobenius) vs proxy-free (likelihood,
  portfolio) must be kept distinct throughout.
- **Predictability is horizon-relative.** RV's advantage over daily-data models is largest at short
  horizons; everything mean-reverts at long horizons.
- **All three outcomes equally publishable:** DL wins / econometrics wins / regime-dependent.
  Regime-dependence has a specific statistical signature (blocky `d_t`, high `f_d(0)`) and a
  specific correct tool (conditional predictive ability, not unconditional DM).
