### RCT installation on `Summit` with anaconda

#### Summit Modules: 

```
module load python/2.7.15-anaconda2-5.3.0
module load cuda/9.1.85
module load gcc/6.4.0
conda create --name <venv>
source activate or conda activate <venv>
```

#### Working Stack:

```
radical.entk         : 0.7.13
radical.pilot        : 0.50.19-v0.50.19-101-g936422e@feature-summit_am
radical.utils        : 0.50.3
saga                 : 0.50.0-v0.50.0-22-g8c260f8@feature-summit

```
RCT Stack:

```
pip install radical.utils
git clone https://github.com/radical-cybertools/saga-python.git
cd saga-python;git reset --hard g8c260f8;pip install .;cd ..
git clone https://github.com/radical-cybertools/radical.pilot.git
cd radical.pilot;git reset --hard g936422e;pip install .;cd ..
pip install radical.entk 

```

* Run `radical-stack` to check against the working stack

#### Variables to export before executing

```
export RADICAL_PILOT_DBURL=mongodb://user:user@ds223760.mlab.com:23760/adaptivity
export SAGA_PTY_SSH_TIMEOUT=2000
export RADICAL_PILOT_PROFILE=True
export RP_ENABLE_OLD_DEFINES=True
# export PATH=/usr/sbin:$PATH
export RADICAL_VERBOSE="DEBUG"
export RADICAL_ENTK_PROFILE=True
export RADICAL_ENTK_REPORT=True
```

`python <microscope.py> >>debug.log 2>>debug.log` saves the logs for later 
debugging

#### OpenMM installation

* Download the package from GitHub 

   ```bash 
   git clone https://github.com/pandegroup/openmm.git 
   ```

* Compile the code using `cmake` (version >= 3.12.0) 

   ```bash 
   cd openmm
   
   mkdir -p build_openmm
   
   cd build_openmm
   
   >>> Summitdev
   cmake .. -DCUDA_HOST_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc -DCUDA_SDK_ROOT_DIR=/sw/summitdev/cuda/9.0.69/samples -DCUDA_TOOLKIT_ROOT_DIR=/sw/summitdev/cuda/9.0.69 -DCMAKE_CXX_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/g++ -DCMAKE_C_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc -DCMAKE_INSTALL_PREFIX=${openmm_install_path (/ccs/home/hm0/anaconda2_ppc)} 
   
   >>>Summit
   cmake .. -DCUDA_HOST_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc -DCUDA_SDK_ROOT_DIR=/sw/summit/cuda/9.1.85/samples -DCUDA_TOOLKIT_ROOT_DIR=/sw/summit/cuda/9.1.85/ -DCMAKE_CXX_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/g++ -DCMAKE_C_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc -DCMAKE_INSTALL_PREFIX=/ccs/home/hm0/.conda/envs/hm0
   ```

   SUMMIT is using different CUDA from SummitDev. The gcc from SD can still be used to compile the code. 

* Build the package 

   ```bash 
   make -j 40 
   
   make install 
   
   make PythonInstall
   ```

* Test the installation 

   ```bash 
   python -m simtk.testInstallation 
   make test
   ```
Collapse




