#!/bin/bash
# Launch the 15 D2 jobs. This script does not run git and does not wait.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="/local/scratch/a/lim316/miniconda3/envs/covharness/bin/python"
NPROC="$(nproc)"
CAP="$((NPROC - 4))"
if [ "$CAP" -lt 1 ]; then
  CAP=1
fi
mkdir -p results/post_screen/D2
printf '%s\n' "$NPROC" "$CAP" > results/post_screen/D2/parallelism.txt
for variant in V10 V01 V11; do
  for seed in 0 1 2 3 4; do
    directory="results/post_screen/D2/${variant}/seed_${seed}"
    mkdir -p "$directory"
    while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do
      sleep 2
    done
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PY" scripts/post_screen_d2_job.py --variant "$variant" --seed "$seed" \
      > "${directory}/stdout.log" 2>&1 &
  done
done
# The jobs stay in this shell. The caller backgrounds the shell.
wait
