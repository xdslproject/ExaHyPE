#!/bin/bash

#SBATCH -N 1
#SBATCH -p shared
#SBATCH -t 0:01:00
#SBATCH -o correct.out
#SBATCH -e errors.out
#SBATCH -n 1
#SBATCH --mem 4GB 

module purge
module load gcc
# module load openmpi

# module load intel/2023.2
module load intelmpi/2021.10

module load python/3.10.8

python ../DSL/ExaHyPE/examples/Batched_stateless.py

# icpx correctness_test.cpp test.cpp Functions.cpp ../Peano/Peano/src/exahype2/fv/PatchUtils.h ../Peano/Peano/src/exahype2/CellData.cpp ../Peano/Peano/src/exahype2/fv/rusanov/rusanov.cpp -o crct_test -DDimensions=2 -I ../Peano/Peano/src -fopenmp -std=c++17 -w

g++ correctness_test.cpp test.cpp Functions.cpp ../Peano/Peano/src/exahype2/CellData.cpp -o crct_test  -I ../Peano/Peano/src -fopenmp -fpermissive -std=c++17
#-DDimensions 2
# ./crct_test

rm crct_test
