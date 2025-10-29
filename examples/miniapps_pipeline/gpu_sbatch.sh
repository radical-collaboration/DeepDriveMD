#!/bin/sh -l
  
#SBATCH -A XXX 
#SBATCH --partition=gpu   #-debug
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --export    NONE
#SBATCH --time=00:30:00
#SBATCH --job-name ddmd_gpu
#SBATCH --mail-user=XXX 
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)

module load anaconda
source activate base
conda activate XXX 

python run_miniapps_pipeline.py
