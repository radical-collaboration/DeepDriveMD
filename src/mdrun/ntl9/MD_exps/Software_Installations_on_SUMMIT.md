# Softwares Installation on SUMMIT 

## OpenMM 

The latest version of OpenMM can be compile without issue on Summit, thanks to the developers. 

* Installation 

  1. Download the package from GitHub 

     ```bash 
     git clone https://github.com/pandegroup/openmm.git 
     ```

  2. Load Anaconda module and create your own conda environment 

     ```bash
     module load python/2.7.15-anaconda2-5.3.0 
     module load gcc/7.4.0  
     module load cuda/9.2.148 
     module load cmake/3.15.2 
     ```

     NOTE: 
        1. The current running version is still on `Python 2.7` and there shouldn't be any trouble with `Python 3`. 
        2. OpenMM can be installed with any `CUDA` version on Summit. Using 9.2 was to accommodate TF.  
     
3. Install the dependencies, `swig`, `numpy` and `cython` via `conda`  
  
4. Compile the code using `cmake` (version >= 3.12.0) 
  
   ```bash 
     ccmake ..
     ```
     Changes to make: 
           1. `CUDA_HOST_COMPILER` set to `gcc 7.4` 
           2. `CUDA_TOOLKIT_ROOT_DIR` set as `${CUDA_ROOT}/samples` 
           3. (Recommended) Install OpenMM in your conda path, set `CMAKE_INSTALL_PREFIX`. 
  
5. Build the package 
  
   ```bash 
     make -j 40 
     
     make install 
     
     make PythonInstall
     ```
  
6. Test the installation 
  
   ```bash 
     python -m simtk.testInstallation 
     make test
     ```
  
* Known issues
    1. A lower version (6.4.0) of `gcc` may result in `no module found: simtk`.
    2. A higher version `gcc/8.1.0` may result in large/wrong `Median difference` from CPU platform from `python -m simtk.testInstallation`.  


## MDAnalysis

* Installation 

  1. Install `scipy` from `conda`

     ```bash
     conda install scipy
     ```

     This will install `scipy (1.1.0-py27h9c1e066_0)` and `numpy (1.14.5) `. 

  2. Update `numpy` with `pip` to the newest version

     ```
     pip install --upgrade numpy
     ```

     Update `numpy` to the newest version to avoid issue 2. 

  3. Install `MDAnalysis`

     ```
     pip install MDAnalysis==0.18  MDAnalysisTests	
     ```

  4. Test your installation by import `MDAnalysis`

     ```
     python -c 'import MDAnalysis as mda'
     ```

  * NOTE: Don't forget `h5py`, install it from `conda` 

* Known issues 
    Let me know if you have issue. Email: heng.ma@anl.gov 
    
