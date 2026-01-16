#!/bin/bash

# Activate conda environment
source /home/rzh/miniconda3/bin/activate resume_attack

for noise in 0.05
do
    echo "======================================="
    echo "Running with noise = $noise"
    echo "======================================="
    
    for i in {1..3}
    do
        echo "Run $i/3 with noise=$noise"
        python Rossler_Integral_nn.py --noise $noise --gpu_id 1 --n_samples 300
    done
    
    echo ""
done

echo "All runs completed!"
