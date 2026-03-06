source /home/rzh/miniconda3/bin/activate resume_attack

M=300
tmax=20
N=8
max_order=7
gpu_id=1

for noise in 0.2
do
    echo "======================================="
    echo "Running HyperPINN with noise = $noise"
    echo "======================================="
    
    for i in {1..3}
    do
        python models/baseline/HyperPINN/HyperPINN_Rossler.py \
            --M $M \
            --tmax $tmax \
            --N $N \
            --max_order $max_order \
            --gpu_id $gpu_id \
            --noise $noise
    done
done