#!/bin/bash
set -euo pipefail

module load anaconda


##############################################
# 1. DeepDriveMD env
##############################################
conda create -y -p $CONDA_ENV/deepdrivemd python=3.8
conda activate $CONDA_ENV/deepdrivemd
pip install --upgrade pip setuptools wheel
cd $WORK_DIR
pwd
pip install -e .

conda deactivate

##############################################
# 2. OpenMM env
##############################################
conda create -y -p $CONDA_ENV/conda-openmm python=3.9
conda activate $CONDA_ENV/conda-openmm
conda install -y -c conda-forge "openmm>=8.0" "cudatoolkit=11.8"
pip install --upgrade pip setuptools wheel
cd $WORK_DIR
pwd
pip install -e .
cd $WORK_DIR/MD-tools
pwd
pip install -e .

conda deactivate

##############################################
# 3. Keras / TensorFlow env
##############################################
conda create -y -p $CONDA_ENV/conda-keras python=3.6.12
conda activate $CONDA_ENV/conda-keras
conda install -y scikit-learn
pip install --upgrade pip setuptools wheel
pip install tensorflow-gpu==2.6.2 pandas
cd $WORK_DIR
pwd
pip install -e .

conda deactivate
