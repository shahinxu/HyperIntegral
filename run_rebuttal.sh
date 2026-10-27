#!/usr/bin/env bash
# Full rebuttal experiment matrix. Safe to re-run: every sweep skips jobs that
# already produced a summary.json, so an interrupted run resumes where it left off.
set -u
cd "$(dirname "$0")"

PY=$HOME/miniconda3/envs/scalepinn/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$HOME/.local/julia-1.10.5/bin:$PATH
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
JPG="${JPG:-2}"   # 8 GPUs x 2 = 16 concurrent slots
SEEDS_MAIN="${SEEDS_MAIN:-11 12 13 14 15 16 17 18 19 20}"
SEEDS_ABL="${SEEDS_ABL:-11 12 13}"
EPOCHS="${EPOCHS:-8000}"
STAGE1="${STAGE1:-1500}"
RAMP="${RAMP:-1000}"

mkdir -p results/paper

stage () {  # stage <name> <suite> <scenes...> -- <seeds...>
  local name=$1 suite=$2; shift 2
  local scenes=() seeds=()
  while [ "$1" != "--" ]; do scenes+=("$1"); shift; done; shift
  seeds=("$@")
  echo "=== [$(date +%H:%M:%S)] stage $name ==="
  $PY -m experiments.paper_sweep --suite "$suite" \
      --scenes "${scenes[@]}" --seeds "${seeds[@]}" --noises 0.02 \
      --n_epochs "$EPOCHS" --stage1 "$STAGE1" --ramp "$RAMP" \
      --gpus $GPUS --jobs_per_gpu "$JPG" --out "results/paper/$name" \
      2>&1 | tee -a "results/paper/${name}.log"
}

# Fastest stages first: every container restart costs only the jobs in flight,
# and each completed stage immediately fills another response table.
stage main            main     ecological social -- $SEEDS_MAIN
stage sparsity        sparsity ecological social -- $SEEDS_ABL
stage ablation        ablation ecological social -- $SEEDS_ABL
stage sigma_h         sigma_h  ecological social -- $SEEDS_ABL
stage noise           noise    ecological social -- $SEEDS_ABL
stage tuning          tuning   ecological social -- $SEEDS_ABL
# Neuronal last: n=30 makes every run far slower, and HyperPINN there needs a
# reduced epoch budget to finish inside the per-job timeout at all.
stage main_neuronal   main     neuronal -- $SEEDS_MAIN
stage sparsity_neuronal  sparsity  neuronal -- $SEEDS_ABL
stage ablation_neuronal  ablation  neuronal -- $SEEDS_ABL
stage sigma_h_neuronal   sigma_h   neuronal -- $SEEDS_ABL
stage noise_neuronal     noise     neuronal -- $SEEDS_ABL
stage tuning_neuronal    tuning    neuronal -- $SEEDS_ABL

echo "=== [$(date +%H:%M:%S)] all stages finished ==="
for d in results/paper/*/; do d=$(basename "$d");
  f="results/paper/$d/results.jsonl"
  [ -f "$f" ] && echo "$d: $(grep -c . "$f") records"
done
