source /home/rzh/miniconda3/bin/activate resume_attack

M=300
gpu_id=0

for noise in 0.05
do
    echo "======================================="
    echo "Running with noise = $noise"
    echo "======================================="
    
    for i in {1..3}
    do
        python Rossler_Integral_linear.py \
            --noise $noise \
            --gpu_id $gpu_id \
            --n_samples $M
    done
done