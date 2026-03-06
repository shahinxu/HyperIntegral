#!/usr/bin/env bash

set -e

SCENE=${1:-rossler}
N_SAMPLES=${2:-300}
NOISE=${3:-0.0}
GPU_ID=${4:-0}
N_EPOCHS=${5:-20000}
LR=${6:-0.001}
MAX_ORDER=${7:-7}
N_NODES=${8:-60}
RESULTS_ROOT=${9:-results/integral}

echo "======================================="
echo "Running Integral"
echo "scene=${SCENE}, n_samples=${N_SAMPLES}, noise=${NOISE}, gpu_id=${GPU_ID}, n_epochs=${N_EPOCHS}, lr=${LR}, max_order=${MAX_ORDER}, n_nodes=${N_NODES}, results_root=${RESULTS_ROOT}"
echo "======================================="

python -m models.integral.run \
    --scene "${SCENE}" \
    --n_samples "${N_SAMPLES}" \
    --noise "${NOISE}" \
    --gpu_id "${GPU_ID}" \
    --n_epochs "${N_EPOCHS}" \
    --lr "${LR}" \
    --max_order "${MAX_ORDER}" \
    --n_nodes "${N_NODES}" \
    --results_root "${RESULTS_ROOT}"