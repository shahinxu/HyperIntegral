#!/usr/bin/env bash

set -e

SCENE=${1:-ecological}
N_SAMPLES=${2:-300}
NOISE=${3:-0.0}
GPU_ID=${4:-0}
N_NODES=${5:-60}
MAX_ORDER=${6:-5}
RESULTS_ROOT=${7:-results/hyperpinn}

echo "======================================="
echo "Running HyperPINN"
echo "scene=${SCENE}, n_samples=${N_SAMPLES}, noise=${NOISE}, gpu_id=${GPU_ID}, n_nodes=${N_NODES}, max_order=${MAX_ORDER}, results_root=${RESULTS_ROOT}"
echo "======================================="

python -m models.hyperpinn.run \
  --scene "${SCENE}" \
  --n_samples "${N_SAMPLES}" \
  --noise "${NOISE}" \
  --gpu_id "${GPU_ID}" \
  --n_nodes "${N_NODES}" \
  --max_order "${MAX_ORDER}" \
  --results_root "${RESULTS_ROOT}"