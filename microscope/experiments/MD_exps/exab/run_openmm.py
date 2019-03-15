import simtk.unit as u
import sys, os, shutil 
import argparse 

sys.path.append('../')
from MD_utils.openmm_simulation import openmm_simulate_charmm_npt_z 

# top_file = os.path.abspath('./input_data/exab.top')
# pdb_file = os.path.abspath('./input_data/exab.gro')

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="pdb file")
parser.add_argument("-p", help="topology file")
parser.add_argument("-c", help="check point file to restart simulation")

args = parser.parse_args()

if args.f:
    pdb_file = os.path.abspath(args.f)
else:
    raise IOError("No pdb file assigned")

if args.p:
    top_file = os.path.abspath(args.p)
else:
    top_file = None

if args.c:
    check_point = os.path.abspath(args.c)
else:
    check_point = None

gpu_index = 0

openmm_simulate_charmm_npt_z(top_file, pdb_file,
                           check_point = check_point,
                           GPU_index=gpu_index,
                           output_traj="output.dcd",
                           output_log="output.log",
                           output_cm='output_cm.h5',
                           report_time=50*u.picoseconds,
                           sim_time=10000*u.nanoseconds)


