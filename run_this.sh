#!/usr/bin/env bash

set -e

SCENE=${1:-ecological}
N_SAMPLES=${2:-300}
NOISE=${3:-0.0}
MAX_ORDER=${4:-7}
BIN_THRESH=${5:-1e-4}
N_NODES=${6:-60}
RESULTS_ROOT=${7:-results/this}

echo "======================================="
echo "Running THIS"
echo "scene=${SCENE}, n_samples=${N_SAMPLES}, noise=${NOISE}, bin_thresh=${BIN_THRESH}, results_root=${RESULTS_ROOT}"
echo "======================================="

python -m models.this.run \
    --scene "${SCENE}" \
    --n_samples "${N_SAMPLES}" \
    --noise "${NOISE}" \
    --bin_thresh "${BIN_THRESH}" \
    --results_root "${RESULTS_ROOT}"