# hyperspace
Distributed Bayesian Hyperparameter Optimization using RCT 


### Installation on `xsede.bridges` via miniconda2

```
* wget <miniconda2> https://conda.io/en/latest/miniconda.html 
* bash Miniconda2-latest-Linux-x86_64.sh
* module load python2/2.7.11_gcc
* cd /pylon5/mc3bggp/dakka
* conda create --name <ve_name> python=2.7 
* source activate <ve_name>
* module load cuda/9.0
* module load keras/2.2.0_tf1.7_py2_gpu tensorflow/1.7_py2_gpu
```

### Installation on `xsede.bridges` via miniconda3

```
* wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
* bash Miniconda3-latest-Linux-x86_64.sh
* conda create --name <VE name> python=3.6
* source activate /home/dakka/.conda/envs/<VE name> or source activate <VE name>
* conda install scikit-learn
* conda install -c omnia/label/cuda92 -c conda-forge openmm
* git clone git@github.com:yngtodd/hyperspace.git
* cd hyperspace; python setup.py install 
* Check for existing conda envs: `conda info --envs`
```

### `OpenMM` specific installations for `CUDA`

```
* module load cuda/9.0
* conda install -c omnia/label/cuda90 -c conda-forge openmm
* export OPENMM_CUDA_COMPILER=`which nvcc`
* python -m simtk.testInstallation
```



### Installation of `mpi4py` on XSEDE Bridges using GCC compiler 

* wget `mpi4py` from bitbucket
* modify `mpi.cfg` as shown [here](https://github.com/jdakka/hyperspace-RCT/hyperspace_workload) and on [mpi4py installation](https://mpi4py.readthedocs.io/en/stable/install.html#using-pip-or-easy-install)
* python setup.py build
* python setup.py install