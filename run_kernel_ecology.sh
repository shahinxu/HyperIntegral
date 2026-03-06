#!/usr/bin/env bash

set -e

N_SAMPLES=${1:-300}
NOISE=${2:-0.0}

python models/kernel/run.py \
  --scene ecological \
  --n_samples "${N_SAMPLES}" \
  --noise "${NOISE}" \
  --n_epochs 3000
