#!/bin/bash -x
#SBATCH --time=40:40:00
#SBATCH --mem=12gb
#SBATCH --export=ALL
#SBATCH --partition=multiple

echo "Loading Python module and mpi module"
module load devel/python/3.10.0_gnu_11.1
module load compiler/gnu/11.1
module load mpi/openmpi/4.1

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores each."
echo "Each node has ${SLURM_MEM_PER_NODE} of memory allocated to this job."
echo "Grid size $3 x $4"
echo "Discretization $1 x $2"
time mpirun python sliding_lid_parallel.py -d $1 $2 -g $3 $4 -b
