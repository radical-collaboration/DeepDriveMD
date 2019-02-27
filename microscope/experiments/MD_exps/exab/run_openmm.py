import simtk.unit as u
import sys, os, shutil 
import errno

sys.path.append('../')

from MD_utils.openmm_simulation import openmm_simulate_charmm_npt_z 

top_file = os.path.abspath('./input_data/exab.top')
pdb_file = os.path.abspath('./input_data/exab.gro')

gpu_index = 0
check_point = None


openmm_simulate_charmm_npt_z(top_file, pdb_file,
                           check_point = check_point,
                           GPU_index=gpu_index,
                           output_traj="output.dcd",
                           output_log="output.log",
                           output_cm='output_cm.h5',
                           report_time=50*u.picoseconds,
                           sim_time=10000*u.nanoseconds)


