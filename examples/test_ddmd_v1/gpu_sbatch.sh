#!/bin/sh -l
  
#SBATCH -A < >
#SBATCH --partition=gpu  #-debug
#SBATCH --nodes=3
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --export    NONE
#SBATCH --time=00:30:00
#SBATCH --job-name ddmd_gpu
#SBATCH --mail-user=< >
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)


export BASE_DIR=/anvil/scratch/$USER/test_ddmd
export WORK_DIR=$BASE_DIR/DeepDriveMD/examples/test_ddmd_v1

#WARNING: this directory has to be empty before running new experiment!
export EXPRMNT_DIR=$WORK_DIR/ddmd_test_experiments
# Remove the following line if you want to keep data from previous experiments.
rm -rf $EXPRMNT_DIR

export CONDA_ENV=$WORK_DIR/conda_env
unset SLURM_EXPORT_ENV
module load anaconda
source activate base
conda activate   $CONDA_ENV/deepdrivemd


cp  $WORK_DIR/test/bba/lassen-keras-dbscan.yaml $WORK_DIR/test/bba/new_lassen-keras-dbscan.yaml
sed -i "s|\${EXPRMNT_DIR}|$EXPRMNT_DIR|g" $WORK_DIR/test/bba/new_lassen-keras-dbscan.yaml 
sed -i "s|\${CONDA_ENV}|$CONDA_ENV|g" $WORK_DIR/test/bba/new_assen-keras-dbscan.yaml
sed -i "s|\${WORK_DIR}|$WORK_DIR|g" $WORK_DIR/test/bba/new_lassen-keras-dbscan.yaml

python -m deepdrivemd.deepdrivemd -c $WORK_DIR/test/bba/new_lassen-keras-dbscan.yaml
