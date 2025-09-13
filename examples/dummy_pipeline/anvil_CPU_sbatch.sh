#!/bin/sh -l
  
#SBATCH -A *** 
#SBATCH --partition debug   #wholenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --job-name ddmd_cpu
#SBATCH --mail-user=***  
#SBATCH --mail-type=ALL    # When to send emails (BEGIN, END, FAIL, ALL)

module load anaconda
source activate base
conda activate *** 

python run_dummy_pipeline.py
