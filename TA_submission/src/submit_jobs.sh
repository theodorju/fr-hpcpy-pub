#!/bin/bash
echo "Submitting jobs"

# Cases for the 300x300 grid
sbatch --nodes=2 --ntasks-per-node=2 --job-name="4" run_sbatch_job.sh 2 2 300 300
sbatch --nodes=2 --ntasks-per-node=4 --job-name="8" run_sbatch_job.sh 4 2 300 300
sbatch --nodes=2 --ntasks-per-node=8 --job-name="16" run_sbatch_job.sh 4 4 300 300
sbatch --nodes=2 --ntasks-per-node=16 --job-name="32" run_sbatch_job.sh 8 4 300 300
sbatch --nodes=2 --ntasks-per-node=32 --job-name="64" run_sbatch_job.sh 8 8 300 300
sbatch --nodes=4 --ntasks-per-node=32 --job-name="128" run_sbatch_job.sh 16 8 300 300
sbatch --nodes=8 --ntasks-per-node=32 --job-name="256" run_sbatch_job.sh 16 16 300 300
sbatch --nodes=16 --ntasks-per-node=32 --job-name="512" run_sbatch_job.sh 32 16 300 300
sbatch --nodes=32 --ntasks-per-node=32 --job-name="1024" run_sbatch_job.sh 32 32 300 300
sbatch --nodes=64 --ntasks-per-node=32 --job-name="2048" run_sbatch_job.sh 64 32 300 300
sbatch --nodes=128 --ntasks-per-node=32 --job-name="4096" run_sbatch_job.sh 64 64 300 300
sbatch --nodes=128 --ntasks-per-node=40 --job-name="5120" run_sbatch_job.sh 128 40 300 300

# Cases for the 1000x1000 grid
sbatch --nodes=2 --ntasks-per-node=2 --job-name="1_4" run_sbatch_job.sh 2 2 1000 1000
sbatch --nodes=2 --ntasks-per-node=4 --job-name="1_8" run_sbatch_job.sh 4 2 1000 1000
sbatch --nodes=2 --ntasks-per-node=8 --job-name="1_16" run_sbatch_job.sh 4 4 1000 1000
sbatch --nodes=2 --ntasks-per-node=16 --job-name="1_32" run_sbatch_job.sh 8 4 1000 1000
sbatch --nodes=2 --ntasks-per-node=32 --job-name="1_64" run_sbatch_job.sh 8 8 1000 1000
sbatch --nodes=4 --ntasks-per-node=32 --job-name="1_128" run_sbatch_job.sh 16 8 1000 1000
sbatch --nodes=8 --ntasks-per-node=32 --job-name="1_256" run_sbatch_job.sh 16 16 1000 1000
sbatch --nodes=16 --ntasks-per-node=32 --job-name="1_512" run_sbatch_job.sh 32 16 1000 1000
sbatch --nodes=32 --ntasks-per-node=32 --job-name="1_1024" run_sbatch_job.sh 32 32 1000 1000
sbatch --nodes=64 --ntasks-per-node=32 --job-name="1_2048" run_sbatch_job.sh 64 32 1000 1000
sbatch --nodes=128 --ntasks-per-node=32 --job-name="1_4096" run_sbatch_job.sh 64 64 1000 1000
sbatch --nodes=128 --ntasks-per-node=40 --job-name="1_5120" run_sbatch_job.sh 128 40 1000 1000

# Cases for the 3000x3000 grid
sbatch --nodes=2 --ntasks-per-node=2 --job-name="3_4" run_sbatch_job.sh 2 2 3000 3000
sbatch --nodes=2 --ntasks-per-node=4 --job-name="3_8" run_sbatch_job.sh 4 2 3000 3000
sbatch --nodes=2 --ntasks-per-node=8 --job-name="3_6" run_sbatch_job.sh 4 4 3000 3000
sbatch --nodes=2 --ntasks-per-node=16 --job-name="3_32" run_sbatch_job.sh 8 4 3000 3000
sbatch --nodes=2 --ntasks-per-node=32 --job-name="3_64" run_sbatch_job.sh 8 8 3000 3000
sbatch --nodes=4 --ntasks-per-node=32 --job-name="3_128" run_sbatch_job.sh 16 8 3000 3000
sbatch --nodes=8 --ntasks-per-node=32 --job-name="3_256" run_sbatch_job.sh 16 16 3000 3000
sbatch --nodes=16 --ntasks-per-node=32 --job-name="3_512" run_sbatch_job.sh 32 16 3000 3000
sbatch --nodes=32 --ntasks-per-node=32 --job-name="3_1024" run_sbatch_job.sh 32 32 3000 3000
sbatch --nodes=64 --ntasks-per-node=32 --job-name="3_2048" run_sbatch_job.sh 64 32 3000 3000
sbatch --nodes=128 --ntasks-per-node=32 --job-name="3_4096" run_sbatch_job.sh 64 64 3000 3000
sbatch --nodes=128 --ntasks-per-node=40 --job-name="3_5120" run_sbatch_job.sh 128 40 3000 3000