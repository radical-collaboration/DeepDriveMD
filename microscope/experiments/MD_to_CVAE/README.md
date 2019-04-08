# Depends on how this steps will be handled, the script needs to be handled differently. 

1. Running on the master node. 
    No more action should be required other than specify the directories of input (MD outputs), 
    and output (OpenMM trajectory length file and combined contact map h5 file). 

    ```python
    import numpy as np
    from glob import glob
    
    # Let's say I have a list of h5 file names 
    cm_files = sorted(glob('omm*/*_cm.h5'))
    
    # Get a list of opened h5 files 
    cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files]
    
    # Compress all .h5 files into one in cvae format 
    cvae_input = cm_to_cvae(cm_data_lists)
    train_data_length = [cm_data.shape[1] for cm_data in cm_data_lists]
    cvae_data_length = len(cvae_input)
    
    # Write the traj info 
    omm_log = 'openmm_log.txt' 
    log = open(omm_log, 'w')
    for i, n_frame in enumerate(train_data_length):
        log.writelines("{} {}\n".format(cm_files[i], n_frame))
    log.close()
    
    # Create .h5 as cvae input
    cvae_input_file = 'cvae_input.h5'
    cvae_input_save = h5py.File(cvae_input_file, 'w')
    cvae_input_save.create_dataset('contact_maps', data=cvae_input)
    cvae_input_save.close():
    ```

2. A inference node like the MD and CVAE nodes. 
    It will require the all there dir path be in the `args`. 
