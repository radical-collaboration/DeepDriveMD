import numpy as np 
from glob import glob

# Let's say I have a list of h5 file names 
cm_files = sorted(glob('omm*/*_cm.h5')) 

# Get a list of opened h5 files 
cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files] 

# # A function to count number of frames 
# frame_number = lambda lists: sum([cm.shape[1] for cm in lists]) 
# 
# frame_marker = 0
# # number of training frames for cvae, change to 1e5 later, HM 
# while frame_number(cm_data_lists) < 100000:
#     for cm in cm_data_lists:
#         cm.refresh()
#     if frame_number(cm_data_lists) >= frame_marker:
#         print('Current number of frames from OpenMM:', frame_number(cm_data_lists))
#         frame_marker = int((10000 + frame_marker) / 10000) * 10000
#         print('    Next report at frame', frame_marker)
# 
# print('Ready for CAVE with total number of frames:', frame_number(cm_data_lists))

# Compress all .h5 files into one in cvae format 
cvae_input = cm_to_cvae(cm_data_lists)
train_data_length = [cm_data.shape[1] for cm_data in cm_data_lists]
cvae_data_length = len(cvae_input)

# # Write the traj info 
omm_log = 'openmm_log.txt' 
log = open(omm_log, 'w')
for i, n_frame in enumerate(train_data_length):
    log.writelines("{} {}\n".format(cm_files[i], n_frame))
log.close()

# Create .h5 as cvae input
cvae_input_file = 'cvae_input.h5'
cvae_input_save = h5py.File(cvae_input_file, 'w')
cvae_input_save.create_dataset('contact_maps', data=cvae_input)
cvae_input_save.close()
