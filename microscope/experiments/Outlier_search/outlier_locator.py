import os 
import argparse 
import numpy as np 
from glob import glob
from utils import read_h5py_file, outliers_from_cvae, cm_to_cvae  
from utils import predict_from_cvae, outliers_from_latent
from utils import find_frame, write_pdb_frame, make_dir_p 

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
DEBUG = 1 

# Inputs 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--md", help="Input: MD simulation directory")
# parser.add_argument("-o", help="output: cvae weight file. (Keras cannot load model directly, will check again...)")
parser.add_argument("-c", "--cvae", help="Input: CVAE model directory")
parser.add_argument("-p", "--pdb", help="Input: pdb file") 

args = parser.parse_args()

# Pdb file for MDAnalysis 
pdb_file = os.path.abspath(args.pdb) 

# Find the trajectories and contact maps 
cm_files_list = sorted(glob(os.path.join(args.md, 'omm_runs_*/*_cm.h5')))
traj_file_list = sorted(glob(os.path.join(args.md, 'omm_runs_*/*.dcd'))) 
check_pnt_list = sorted(glob(os.path.join(args.md, 'omm_runs_*/checkpnt.chk'))) 

if cm_files_list == []: 
    raise IOError("No h5/traj file found, recheck your input filepath") 

# Find all the trained model weights 
model_weights = sorted(glob(os.path.join(args.cvae, 'cvae_runs_*/cvae_weight.h5'))) 

# Convert everything to cvae input 
cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files_list] 
cvae_input = cm_to_cvae(cm_data_lists)

# A record of every trajectory length
train_data_length = [cm_data.shape[1] for cm_data in cm_data_lists]
traj_dict = dict(zip(traj_file_list, train_data_length)) 

# Outlier search 
outlier_list = [] 
eps_record = {} 

for model_weight in model_weights: 
    # Identify the latent dimensions 
    model_dim = int(os.path.basename(os.path.dirname(model_weight))[10:12]) 
    print 'Model latent dimension: %d' % model_dim  
    # Get the predicted embeddings 
    cm_predict = predict_from_cvae(model_weight, cvae_input, hyper_dim=model_dim) 
    # initialize eps if empty 
    if str(model_weight) in eps_record.keys(): 
        eps = eps_record[model_weight] 
    else: 
        eps = 0.2 

    # Search the right eps for DBSCAN 
    while True: 
        outliers = np.squeeze(outliers_from_latent(cm_predict, eps=eps)) 
        n_outlier = len(outliers) 
        print('dimension = {0}, eps = {1:.2f}, number of outlier found: {2}'.format(
            model_dim, eps, n_outlier))
        if n_outlier > 50: 
            eps = eps + 0.05 
        else: 
            eps_record[model_weight] = eps 
            outlier_list.append(outliers) 
            break 


outlier_list_uni, outlier_count = np.unique(np.hstack(outlier_list), return_counts=True) 

if DEBUG: 
    print outlier_list_uni
    
outliers_pdb_path = os.path.abspath('./outlier_pdbs') 
make_dir_p(outliers_pdb_path) 
print 'Writing outliers in %s' % outliers_pdb_path  

new_outlier_list = [] 
for outlier in outlier_list_uni: 
    traj_file, num_frame = find_frame(traj_dict, outlier)  
    outlier_pdb_file = os.path.join(outliers_pdb_path, '{}_{:06d}.pdb'.format(os.path.basename(os.path.dirname(traj_file)), num_frame)) 
    if not os.path.exists(outlier_pdb_file): 
        print 'Found a new outlier# {} at frame {} of {}'.format(outlier, num_frame, traj_file)
        outlier_pdb = write_pdb_frame(traj_file, pdb_file, num_frame, outlier_pdb_file)  
        print '     Written as {}'.format(outlier_pdb_file)
    new_outlier_list.append(outlier_pdb_file) 


