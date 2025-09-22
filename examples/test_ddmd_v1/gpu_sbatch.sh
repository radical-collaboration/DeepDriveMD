#!/bin/sh -l
  
#SBATCH -A <***>
#SBATCH --partition=gpu  #-debug
#SBATCH --nodes=3
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --export    NONE
#SBATCH --time=00:30:00
#SBATCH --job-name ddmd_gpu
#SBATCH --mail-user=<***>
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)

unset SLURM_EXPORT_ENV
module load anaconda
source activate base
conda activate   /anvil/scratch/$USER/conda_env/deepdrivemd

python -m deepdrivemd.deepdrivemd -c test/bba/lassen-keras-dbscan.yaml
