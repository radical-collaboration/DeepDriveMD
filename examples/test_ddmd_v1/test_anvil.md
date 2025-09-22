# DeepDriveMD-F (DeepDriveMD-pipeline)

## Environment Setup

```shell

    # Create a working directory for code and experiments
    mkdir -p /anvil/scratch/$USER/test_ddmd
    export BASE_DIR=/anvil/scratch/$USER/test_ddmd
    cd $BASE_DIR

    # Clone the DeepDriveMD repository
    git clone git@github.com:radical-collaboration/DeepDriveMD.git
    cd DeepDriveMD
    git checkout try/test_MDpipeline

    # Prepare experiment workspace
    cd examples/test_ddmd_v1
    export WORK_DIR=$BASE_DIR/DeepDriveMD/examples/test_ddmd_v1
    mkdir -p $WORK_DIR/ddmd_test_experiments
    mkdir -p $WORK_DIR/conda_env
    export CONDA_ENV=$WORK_DIR/conda_env

    # Clone the MD_tools repository and copy fixes 
    git clone https://github.com/braceal/MD-tools.git
    cp -r MD-tools_fix/* MD-tools

    # Create all required conda environments
    . ./env_setup.sh


```

## Running an Experiment on a GPU Node

```shell
    
    export BASE_DIR=/anvil/scratch/$USER/test_ddmd
    export WORK_DIR=$BASE_DIR/DeepDriveMD/examples/test_ddmd_v1

    # IMPORTANT: Ensure this directory is empty before starting a new experiment
    export EXPRMNT_DIR=$WORK_DIR/ddmd_test_experiments

    # Submit experiment job via SLURM
    cd $WORK_DIR
    sbatch gpu_sbatch.sh

```

## Checking Results

```shell

    cd $PROJECT/radical.pilot.sandbox/<your-session-id>/pilot.0000

    # NOTE: The current test experiment is known to fail in Stage 2 at task.000015

```


