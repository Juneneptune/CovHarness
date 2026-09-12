# Project state

Last updated 2026-09-12.

Project conda environment is `covharness` (Python 3.11). Recreate with `conda env create -f environment.yml` from the repository root.

## Current milestone

Block 2A. Covariance-space losses and the proxy-robustness demonstration. Block 1 is fully closed. The JPM identity reconciliation confirmed that the historical root-only 5-minute grid selected zero noncommon suffixes, which is why 5-minute RV was unchanged. No further TAQ or JPM work is in this block.

## Completed

Block 1 measurement, identity repair, and root-only reconciliation are closed.

Losses implemented in `covharness.losses`.

Squared Frobenius.

```math
L_F(S,H)=\|S-H\|_F^2=\operatorname{tr}((S-H)^{\top}(S-H))=\sum_{ij}(S_{ij}-H_{ij})^2
```

Equal square dimensions, finite entries, symmetry within `SYMMETRY_ATOL=1e-10`. PSD is not required to compute it. No scaling or annualization. Scalar float. Inputs are not mutated.

Reduced multivariate QLIKE, the primary ranking loss.

```math
L_Q(S,H)=\log\det(H)+\operatorname{tr}(H^{-1}S)
```

`S` is square, finite, symmetric, and PSD, and it may be singular. `H` is strictly PD. Cholesky `H=LL^{\top}` supplies both `logdet(H)=2\sum\log\operatorname{diag}(L)` and the triangular solves for `tr(H^{-1}S)`. `inv(H)` is not formed. A failed Cholesky is the PD failure. No jitter, diagonal loading, clipping, or silent fallback.

Full Stein, when both arguments are SPD.

```math
L_S(S,H)=L_Q(S,H)-\log\det(S)-N
```

Reduced QLIKE and full Stein are not numerically equal. For a common SPD target they differ by a target-only term, so rankings and pairwise differentials agree. Singular `S` is rejected by full Stein.

No matrix repair inside evaluation. A non-PD forecast raises `ForecastNotPositiveDefiniteError`.

Descriptive localization. `variance_squared_error` is $\sum_i(S_{ii}-H_{ii})^2$. `correlation_frobenius_squared` is $\|R(S)-R(H)\|_F^2$ with $R(A)=D(A)^{-1}AD(A)^{-1}$. These are not additive pieces of the ranking losses and do not inherit Patton / LRV ranking consistency. Nonpositive diagonals are rejected.

Singular-proxy contract. Rank-1 `S=[[1,1],[1,1]]` with SPD `H` yields finite squared Frobenius and finite reduced QLIKE. Full Stein fails clearly. Rank deficiency does not by itself prevent evaluation. It can still increase proxy noise and reduce power.

Proxy-robustness demonstration. `S=u\Sigma` with `u\sim\mathrm{Exp}(1)`, so $\mathrm{E}[S]=\Sigma$. `H_A=\Sigma`. `H_B=\log(2)\Sigma`. Analytic expected squared Frobenius is 2.18 versus 2.385. Analytic expected reduced QLIKE is 1.906 versus 2.058. Unsquared Frobenius is 1.086 versus 1.023, so the non-robust criterion ranks `H_B` first. Monte Carlo, seed 20260212, 25,000 draws. Mean squared Frobenius 2.227 versus 2.431. Mean reduced QLIKE 1.903 versus 2.055. Mean full Stein 1.185 versus 1.336. Mean unsquared Frobenius 1.098 versus 1.031. Single-draw flip rates. squared Frobenius 0.576, reduced QLIKE 0.568, full Stein 0.568, unsquared Frobenius 0.576. For this scale family the per-draw Frobenius ranking coincides while expected rankings differ.

`M/N` is reported by `matrix_eigen_diagnostics` when `n_returns` is supplied, and as `FrequencyEppsResult.m_over_n`. The TAQ measurement investigation was not rerun.

`BENCHMARK_IMPLEMENTATION_PLAN.md` no longer states that `M>N` is required for QLIKE evaluation. Kernels remain in the broader plan as alternative proxies.

## Files that own the implementation

- `src/covharness/losses/contracts.py`
- `src/covharness/losses/frobenius.py`
- `src/covharness/losses/qlike.py`
- `src/covharness/losses/localization.py`
- `src/covharness/losses/robustness.py`
- `src/covharness/losses/__init__.py`
- `src/covharness/diagnostics/epps.py` (`m_over_n` added)
- `tests/unit/test_losses.py`
- `tests/unit/test_loss_robustness.py`
- `tests/unit/test_epps.py`
- `notebooks/proxy_robust_losses.ipynb`
- `results/proxy_robust_losses.png`
- `README.md`
- `docs/PROJECT_STATE.md`
- `BENCHMARK_IMPLEMENTATION_PLAN.md`

## Tests run

Command `pytest -q` on 2026-09-12 after Block 2A.

```
........................................................................ [ 77%]
.....................                                                    [100%]
93 passed in 13.93s
```

## Methodological decisions already in code

- Reduced QLIKE is the primary ranking loss. Full Stein is the SPD-proxy form.
- Squared Frobenius is the complementary robust loss. Ordinary unsquared Frobenius is a labeled non-robust contrast only.
- Localization diagnostics are descriptive. They are not ranking losses.
- Forecasts that fail PD are exposed, not repaired.
- `M/N` is a proxy-quality diagnostic, not a gate on reduced QLIKE.

## Known problems or limitations

- Opening and overnight treatment remain unresolved.
- Realized kernels are not implemented. The long historical extract is not solved.
- The leak-proof temporal protocol is not implemented.
- Forecasting models, inference tests, and portfolio evaluation are not implemented.

## Next recommended task

Block 2B — leak-proof temporal protocol
