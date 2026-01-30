#!/bin/zsh

export GOOGLETEST=/project/sartori00/googletest
export LD_LIBRARY_PATH=${GOOGLETEST}/lib64:$LD_LIBRARY_PATH

module load gcc/12.2.0
module load cuda/12.3
module load apps/cmake/3.29

export LD_LIBRARY_PATH=/apps/common/gcc/12.2.0/lib64:$LD_LIBRARY_PATH
