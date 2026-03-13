set -e

SCENE=${1:-ecological}
N_SAMPLES=${2:-300}
NOISE=${3:-0.0}
GPU_ID=${4:-0}
N_EPOCHS=${5:-20000}
LR=${6:-0.001}
MAX_ORDER=${7:-7}
N_NODES=${8:-60}
RANK=${9:-20}
MAX_CANDIDATES_PER_ORDER=${10:-4096}
RESULTS_ROOT=${11:-results/low_rank}

echo "======================================="
echo "Running Orderwise Sparse Tensors"
if [ "${SCENE}" = "rossler" ]; then
    echo "scene=${SCENE}, n_samples=${N_SAMPLES}, noise=${NOISE}, gpu_id=${GPU_ID}, n_epochs=${N_EPOCHS}, lr=${LR}, max_order=${MAX_ORDER}, n_nodes=${N_NODES}, rank_ignored=${RANK}, max_candidates_per_order=${MAX_CANDIDATES_PER_ORDER}, results_root=${RESULTS_ROOT}"
else
    echo "scene=${SCENE}, n_samples=${N_SAMPLES}, noise=${NOISE}, gpu_id=${GPU_ID}, n_epochs=${N_EPOCHS}, lr=${LR}, rank_ignored=${RANK}, max_candidates_per_order=${MAX_CANDIDATES_PER_ORDER}, results_root=${RESULTS_ROOT}"
fi
echo "======================================="

CMD=(python -m models.low_rank.run \
        --scene "${SCENE}" \
        --n_samples "${N_SAMPLES}" \
        --noise "${NOISE}" \
        --gpu_id "${GPU_ID}" \
        --n_epochs "${N_EPOCHS}" \
        --lr "${LR}" \
        --rank "${RANK}" \
        --max_candidates_per_order "${MAX_CANDIDATES_PER_ORDER}" \
        --results_root "${RESULTS_ROOT}")

if [ "${SCENE}" = "rossler" ]; then
    CMD+=(--max_order "${MAX_ORDER}" --n_nodes "${N_NODES}")
fi

"${CMD[@]}"