source /home/rzh/miniconda3/bin/activate resume_attack

M=300
gpu_id=0

for noise in 0.02 0.07 0.1 0.2
do
    echo "======================================="
    echo "Running with noise = $noise"
    echo "======================================="
    
    for i in {1..3}
    do
        python models/integral/run.py \
            --scene rossler \
            --noise $noise \
            --gpu_id $gpu_id \
            --n_samples $M
    done
done