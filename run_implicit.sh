#!/usr/bin/env bash

set -e

SCENE=${1:-ecological}
N_SAMPLES=${2:-300}
NOISE=${3:-0.0}
GPU_ID=${4:-0}
N_EPOCHS=${5:-40000}
LR=${6:-0.001}
MAX_ORDER=${7:-7}
N_NODES=${8:-60}
EMBED_DIM=${9:-32}
HEAD_HIDDEN=${10:-64}
EVAL_EVERY=${11:-5000}
MAX_CANDIDATES_PER_ORDER=${12:-4096}
RESULTS_ROOT=${13:-results/implicit}

echo "======================================="
echo "Running Implicit Hypergraph"
if [ "${SCENE}" = "rossler" ]; then
    echo "scene=${SCENE}, n_samples=${N_SAMPLES}, noise=${NOISE}, gpu_id=${GPU_ID}, n_epochs=${N_EPOCHS}, lr=${LR}, max_order=${MAX_ORDER}, n_nodes=${N_NODES}, embed_dim=${EMBED_DIM}, head_hidden=${HEAD_HIDDEN}, eval_every=${EVAL_EVERY}, max_candidates_per_order=${MAX_CANDIDATES_PER_ORDER}, results_root=${RESULTS_ROOT}"
else
    echo "scene=${SCENE}, n_samples=${N_SAMPLES}, noise=${NOISE}, gpu_id=${GPU_ID}, n_epochs=${N_EPOCHS}, lr=${LR}, embed_dim=${EMBED_DIM}, head_hidden=${HEAD_HIDDEN}, eval_every=${EVAL_EVERY}, max_candidates_per_order=${MAX_CANDIDATES_PER_ORDER}, results_root=${RESULTS_ROOT}"
fi
echo "======================================="

CMD=(python -m models.Implicit.run \
        --scene "${SCENE}" \
        --n_samples "${N_SAMPLES}" \
        --noise "${NOISE}" \
        --gpu_id "${GPU_ID}" \
        --n_epochs "${N_EPOCHS}" \
        --lr "${LR}" \
        --embed_dim "${EMBED_DIM}" \
        --head_hidden "${HEAD_HIDDEN}" \
        --eval_every "${EVAL_EVERY}" \
        --max_candidates_per_order "${MAX_CANDIDATES_PER_ORDER}" \
        --results_root "${RESULTS_ROOT}")

if [ "${SCENE}" = "rossler" ]; then
    CMD+=(--max_order "${MAX_ORDER}" --n_nodes "${N_NODES}")
fi

"${CMD[@]}"