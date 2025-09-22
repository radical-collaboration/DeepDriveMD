#!/bin/bash
set -euo pipefail

module load anaconda

### Base directories
export BASE_DIR=/anvil/scratch/$USER
mkdir -p $BASE_DIR/mdtools $BASE_DIR/ddmd

export MDT_DIR=$BASE_DIR/mdtools
export WORK_DIR=$BASE_DIR/ddmd

### Clone MD-tools
if [ ! -d "$MDT_DIR/MD-tools" ]; then
    git clone https://github.com/braceal/MD-tools.git $MDT_DIR/MD-tools
fi

### Clone DeepDriveMD-pipeline
if [ ! -d "$WORK_DIR/DeepDriveMD-pipeline" ]; then
    git clone https://github.com/DeepDriveMD/DeepDriveMD-pipeline.git $WORK_DIR/DeepDriveMD-pipeline
    cd $WORK_DIR/DeepDriveMD-pipeline
    #git checkout test_anvil
fi

##############################################
# 1. DeepDriveMD env
##############################################
conda create -y -p /anvil/scratch/$USER/conda_env/deepdrivemd python=3.8
conda activate /anvil/scratch/$USER/conda_env/deepdrivemd
pip install --upgrade pip setuptools wheel
cd $WORK_DIR/DeepDriveMD-pipeline
pwd
pip install -e .

conda deactivate

##############################################
# 2. OpenMM env
##############################################
conda create -y -p /anvil/scratch/$USER/conda_env/conda-openmm python=3.9
conda activate /anvil/scratch/$USER/conda_env/conda-openmm
conda install -y -c conda-forge "openmm>=8.0" "cudatoolkit=11.8"
pip install --upgrade pip setuptools wheel
cd $WORK_DIR/DeepDriveMD-pipeline
pwd
pip install -e .
cd $MDT_DIR/MD-tools
pwd
pip install -e .

conda deactivate

##############################################
# 3. Keras / TensorFlow env
##############################################
conda create -y -p /anvil/scratch/$USER/conda_env/conda-keras python=3.6.12
conda activate /anvil/scratch/$USER/conda_env/conda-keras
conda install -y scikit-learn
pip install --upgrade pip setuptools wheel
pip install tensorflow-gpu==2.6.2 pandas
cd $WORK_DIR/DeepDriveMD-pipeline
pwd
pip install -e .

conda deactivate
