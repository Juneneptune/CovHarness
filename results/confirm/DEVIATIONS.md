# Deviations

V01 uses the return-only input. The shared fit signature still requires a feature array, so the runner passed a column of zeros that V01 does not read. The forecast is unchanged by that array. The headline definition is unchanged.

The five LSTMBEKK-noC seed cubes are identical. The ablation column is that common cube. No seed was selected.

BEKKS01 carry counts were not written to the job summary. The saved repair count is 0 and the saved policy-event list is empty. The model was not refit to recover a counter.

## Addendum after the run

Commit 4c2a879 corrected a printed Mincer-Zarnowitz p-value in the report text. The stored statistics and the headline were unaffected.
