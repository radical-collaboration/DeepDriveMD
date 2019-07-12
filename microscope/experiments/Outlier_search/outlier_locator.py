from glob import glob
from utils import read_h5py_file 

# Find the trajectories and contact maps 
cm_files_list = sorted(glob('omm_runs_*/*_cm.h5')) 
traj_file_list = sorted(glob('omm_runs_*/*.dcd'))

if cm_files == []: 
    raise IOError("No h5/traj file found, recheck your input filepath") 

# Convert everything to cvae input 
cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files_list] 
cvae_input = cm_to_cvae(cm_data_lists)

# A record of every trajectory length
train_data_length = [cm_data.shape[1] for cm_data in cm_data_lists]
traj_dict = dict(zip(traj_file_list, train_data_length)) 

# Outlier search 

