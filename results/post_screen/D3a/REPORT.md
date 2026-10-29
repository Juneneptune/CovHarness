# Task D3a report

## 1. Outcome

Task D3a. Executed. Shifted-day realized-covariance, quarticity, and return panels were built for phases 00, 03, 06, 09, 12, 15, 18, and 21 on HISTORY_A through SCREEN. Phase 00 matches the production panel bitwise and has 1250 rows. Each later phase has 1249 rows and drops 2023-03-25. No model was trained. CONFIRM labels were not built. The panels are training material. Evaluation targets remain the UTC-day realized covariance.

## 2. Starting state

Branch `main` at spec commit `d45a5bce027b50be38bc5aaf722ddbd183392d10`. The working tree was clean. Documents and code read were the Post-SCREEN register, `docs/BINANCE_OPEN_DATA_MEASUREMENT_SPEC.md`, `src/covharness/data/binance_klines.py`, `binance_calendar.py`, and `src/covharness/realized/binance_panel.py`, including the interval parameter added for A3.

## 3. Spec commit

`d45a5bce027b50be38bc5aaf722ddbd183392d10`, recorded before any phase panel was built.

## 4. Changes

`measure_phase_day` and `build_preconfirm_phase_panels` were added in `covharness.realized.binance_panel`. Hour 0 calls the existing five-minute measurement. A shifted hour stacks the same five-minute endpoints across the phase boundary and passes them to the existing log-return, Gram-matrix, and quarticity functions. The entry script is `scripts/post_screen_phase_panels.py`.

The label convention was fixed in the spec. Phase 00 is the production UTC day, because that is the object the bitwise test names. For \(h\in\{3,6,9,12,15,18,21\}\), the day labeled \(d\) is \([h{:}00\) on \(d-1,\ h{:}00\) on \(d)\), anchored on the previous one-minute close. A window that intersects 2023-03-24 is dropped. Under that rule phase 00 keeps 2023-03-25 and every shifted phase drops it. A kept row ends by 24:00 on its label, so it is first usable at that label and not on the previous day. `contiguous_with_previous` is false for the first row and for the row after the halt gap. Phase offsets three hours apart share 21 hours. Same-phase successive days share only the boundary. No effective-sample-size figure was computed.

## 5. Problems

None observed.

## 6. Verification

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_post_screen_d3a.py -k "not preconfirm"`

Exit code 0. 4 passed, 1 deselected.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -m pytest -q tests/unit/test_post_screen_d3a.py::test_phase_zero_preconfirm_matches_production_panel`

Exit code 0. 1 passed in 15.43 seconds. `np.array_equal` against the production realized-covariance rows through 2025-04-12.

`/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python -u scripts/post_screen_phase_panels.py`

Exit code 0. Elapsed 18.6 seconds. The script repeated the bitwise phase-00 check, the halt-drop check, and the eligibility check, then rechecked the production-panel hash. No CONFIRM date is a row. No repair was applied. A missing close would have raised.

## 7. Artifacts

The phase arrays are local and match `results/**/*.npz` in `.gitignore`, so they are not in the result commit. The committed summary stores their hashes.

| Path | SHA-256 |
| --- | --- |
| `results/post_screen/D3a/d3a_summary.json` | `b036385e0ac4cb745a68ede43a68ccc7d649e8dbb5e3022e6b8fb8a49c872201` |
| `results/post_screen/D3a/phase_00.npz` | `b81caf31c40fe1f12ad462db535160ca5d05fcaa39df911eef5b1c99b10bbbc3` |
| `results/post_screen/D3a/phase_03.npz` | `f7bbfd5ce4ace7430546e270e80a17aaca312e16862784f4e6439fabaa8ad68f` |
| `results/post_screen/D3a/phase_06.npz` | `64cd60a218d0ac5c7498bba4d13256658d620a78858c2e1ce30787f03b22bf5d` |
| `results/post_screen/D3a/phase_09.npz` | `80cceb383cb8dc16431c53b0fbfb6ae5d2f7b8627b8b8dbc102a1d98e84b5e5b` |
| `results/post_screen/D3a/phase_12.npz` | `1075f9061b20ecdbed0816382babebf11f22a6958a4735beeb24cee446f4ecf3` |
| `results/post_screen/D3a/phase_15.npz` | `f6b8f655ef818f90b6ab08b9fc7c674695cce11cb74c489a5737c4abb7a5fc5f` |
| `results/post_screen/D3a/phase_18.npz` | `94a8475832d67cc0ed6ab5ce70c505f311c261c4bfdc350ee05150ba6798771a` |
| `results/post_screen/D3a/phase_21.npz` | `18706e47788a6e9183d7778278a0f07940cc5a58f8e5bdcef21d4927016ace1e` |

Spec commit `d45a5bce027b50be38bc5aaf722ddbd183392d10`. Result commit `b3cc780d971ec76844b6bf1f77a899a6632616e1`. The production panel hash before and after is `2b7358107a10f77c3879524e1d9b1389c2c66db4a444af89c2cdf59aa186640a`. `configs/binance_open_data_core.yaml` was not modified.

## 8. Docs

The register status for D3a is completed. The result commit cell holds `b3cc780d971ec76844b6bf1f77a899a6632616e1`. `docs/PROJECT_STATE.md` records the row counts and the halt drop. `README.md` states that the shifted phases are training panels, that they stop at the last selection date, and that they do not replace the evaluation target.

## 9. Next task

T1, the editorial pass on public documents. Not started.
