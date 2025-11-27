#!/bin/sh -l
  
#SBATCH -A dmr170002p 
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --export    NONE
#SBATCH --time=01:30:00
#SBATCH --job-name ddmd_gpu
#SBATCH --mail-user=mariya.goliyad@rutgers.edu 
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)

module load anaconda3
conda activate /ocean/projects/dmr170002p/goliyad/conda_env/test_dragon 
dragon-network-config --output-to-yaml 

dragon -w ssh --network-config slurm.yaml run_dragon.py 
