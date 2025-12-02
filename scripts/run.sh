srun --partition=msigpu \
     --nodes=1 \
     --ntasks=24 \
     --cpus-per-task=16 \
     --mem=187200M \
     --time=24:00:00 \
     --gres=scratch:850 \
     --job-name=myjob \
     --output=logs/%x_%j.out

     interactive-gpu
#SBATCH --gres=gpu:a40:1

srun -p interactive-gpu --gres=gpu:a40:1 --time=02:00:00 --pty bash -i 
 
# A100 GPU
srun -p msigpu --gres=gpu:a100:1 --time=00:30:00 --pty bash -i
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_COMPILER_WORKS=1 -DCMAKE_CUDA_COMPILER=/common/software/install/manual/cuda/12.0/bin/nvcc

# V100 GPU
srun -p msigpu --gres=gpu:v100:1 --time=00:30:00 --pty bash -i
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_COMPILER_WORKS=1 -DCMAKE_CUDA_COMPILER=/common/software/install/manual/cuda/12.0/bin/nvcc

# Compile with 2PC and float precision
# make PIRANHA_FLAGS="-DFLOAT_PRECISION=4 -DTWOPC"
make -j PIRANHA_FLAGS="-DTWOPC"

./piranha-debug -p 0 -c files/samples/localhost_config.json "--gtest_filter=EvalTest*2PC*" >/dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=2 ./piranha -p 2 -c files/samples/localhost_config.json >/dev/null &
./piranha-debug -p 1 -c files/samples/localhost_config.json "--gtest_filter=EvalTest*2PC*"

