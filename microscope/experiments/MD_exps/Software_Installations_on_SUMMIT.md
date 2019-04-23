# Softwares Installation on SUMMIT 

## OpenMM 

* Installation 

  1. Download the package from GitHub 

     ```bash 
     git clone https://github.com/pandegroup/openmm.git 
     ```

  2. Load Anaconda module and create your own conda environment 

  3. Install the dependencies, `swig`, `numpy` and `cython` 

  4. Compile the code using `cmake` (version >= 3.12.0) 

     ```bash 
     cd openmm
     
     mkdir -p build_openmm
     
     cd build_openmm
     
     >>> Summitdev
     cmake .. -DCUDA_HOST_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc -DCUDA_SDK_ROOT_DIR=/sw/summitdev/cuda/9.0.69/samples -DCUDA_TOOLKIT_ROOT_DIR=/sw/summitdev/cuda/9.0.69 -DCMAKE_CXX_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/g++ -DCMAKE_C_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc -DCMAKE_INSTALL_PREFIX=${openmm_install_path (/ccs/home/hm0/anaconda2_ppc)} 
     
     >>>Summit
     cmake .. -DCUDA_HOST_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc -DCUDA_SDK_ROOT_DIR=/sw/summit/cuda/9.1.85/samples -DCUDA_TOOLKIT_ROOT_DIR=/sw/summit/cuda/9.1.85/ -DCMAKE_CXX_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/g++ -DCMAKE_C_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc -DCMAKE_INSTALL_PREFIX=${openmm_install_path (/ccs/home/hm0/.conda/envs/hm0)} 
     ```

     SUMMIT is using different CUDA from SummitDev. The gcc from SummitDev can still be used to compile the code. Interesingly, the `OpenMM` only work being compiled with certain version of `gcc`, `7.1.1` as tested on `Summitdev`. A different version of `gcc` will cause issues stated in the *known issues* section. 

     UPDATE: `gcc 7.4` is now available on Summit. 

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
     pip install MDAnalysis MDAnalysisTests	
     ```

  4. Test your installation by import `MDAnalysis`

     ```
     python -c 'import MDAnalysis as mda'
     ```

  * NOTE: Don't forget `h5py`

* Known issues

  1. `scipy` from `pip` 

     ```
     (hm0) [hm0@login5.summit fs-pep]$ python -c 'import MDAnalysis as mda'                                                                                                                                                                                    
     Traceback (most recent call last):
       File "<string>", line 1, in <module>
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/MDAnalysis/__init__.py", line 183, in <module>
         from .lib import log
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/MDAnalysis/lib/__init__.py", line 41, in <module>
         from . import pkdtree
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/MDAnalysis/lib/pkdtree.py", line 34, in <module>
         from scipy.spatial import cKDTree
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/scipy/spatial/__init__.py", line 101, in <module>
         from ._spherical_voronoi import SphericalVoronoi
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/scipy/spatial/_spherical_voronoi.py", line 18, in <module>
         from scipy.spatial.distance import pdist
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/scipy/spatial/distance.py", line 126, in <module>
         from ..special import rel_entr
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/scipy/special/__init__.py", line 641, in <module>
         from ._ufuncs import *
     ImportError: /ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/scipy/special/_ufuncs.so: undefined symbol: _gfortran_stop_numeric_f08
     ```

     

  2. Older version of `numpy (<1.16)`

     ```
     (hm0) [hm0@login5.summit fs-pep]$ python -c 'import MDAnalysis as mda'                                                                                                                                                                                    
     Traceback (most recent call last):
       File "<string>", line 1, in <module>
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/MDAnalysis/__init__.py", line 183, in <module>
         from .lib import log
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/MDAnalysis/lib/__init__.py", line 35, in <module>
         from . import transformations
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/MDAnalysis/lib/transformations.py", line 202, in <module>
         from .mdamath import angle as vecangle
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/MDAnalysis/lib/mdamath.py", line 47, in <module>
         from . import util
       File "/ccs/home/hm0/.conda/envs/hm0/lib/python2.7/site-packages/MDAnalysis/lib/util.py", line 210, in <module>
         from ._cutil import unique_int_1d
       File "__init__.pxd", line 872, in init MDAnalysis.lib._cutil
     ValueError: numpy.ufunc has the wrong size, try recompiling. Expected 192, got 216
     ```

