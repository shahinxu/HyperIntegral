#!/bin/bash

# Bash script to run Rossler_Oscillators.py multiple times sequentially
# Usage: ./run_rossler_multiple.sh [num_runs]

# Default parameters
M=300
tmax=20
N=8
max_order=7
gpu_id=1
noise=0.05

# Number of runs (default to 10 if not specified)
NUM_RUNS=${1:-10}

# Create logs directory
LOGS_DIR="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOGS_DIR
LOG_FILE="$LOGS_DIR/all_runs.log"

echo "======================================="
echo "Starting Rossler Oscillators sequential run"
echo "======================================="

failed=0
for i in $(seq 1 $NUM_RUNS); do
    echo "----------------------------------------"
    echo "Starting run $i of $NUM_RUNS"
    echo "Time: $(date)"
    echo "----------------------------------------"
    
    echo "=== Run $i started at $(date) ===" >> $LOG_FILE
    python -u Rossler_Oscillators.py \
        --M $M \
        --tmax $tmax \
        --N $N \
        --max_order $max_order \
        --gpu_id $gpu_id \
        --noise $noise \
        2>&1 | tee -a $LOG_FILE
    
    EXIT_CODE=$?
    echo "=== Run $i finished at $(date) with exit code $EXIT_CODE ===" >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Run $i completed successfully"
    else
        echo "✗ Run $i failed with exit code $EXIT_CODE"
        failed=$((failed + 1))
        echo "ERROR: Run $i failed. Continuing to next run..."
    fi
    echo ""
done
if [ $failed -gt 0 ]; then
    exit 1
fi
