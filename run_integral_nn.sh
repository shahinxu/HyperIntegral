#!/bin/bash

n_samples=300

for i in {1..10}
do
    echo "Run $i/10"
    python Rossler_Integral_nn.py \
        --n_samples $n_samples \
        --gpu_id 0
done