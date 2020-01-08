import os, sys 
import glob, errno
import argparse, h5py  
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.externals import joblib 


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--md", dest="f", help="Input: Directory that stores MD simulations")
# parser.add_argument("-o", help="output: cvae weight file. (Keras cannot load model directly, will check again...)")
parser.add_argument("-d", "--dim", default=12, help="Number of dimensions in latent space")
# parser.add_argument("-gpu", default=0, help="gpu_id")

args = parser.parse_args()

MD_dir = args.f
hyper_dim = int(args.dim) 
# gpu_id = args.gpu

omm_dirs = sorted(glob.glob(os.path.join(MD_dir, 'omm_runs_*')))

if omm_dirs == []:
    raise IOError('Input file doesn\'t exist...')

cm_data_lists = [] 
for omm in omm_dirs: 
    cm_file = os.path.join(omm, 'output_cm.h5') 
    cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True) 
    cm_data_lists.append(cm_h5[u'contact_maps'].value) 
    cm_h5.close()  

pca_input = np.hstack(cm_data_lists) 
print pca_input.shape 

if __name__ == '__main__': 
    pca = PCA(n_components = hyper_dim) 
    pca.fit(pca_input) 

    print pca.transform(pca_input).shape

    model_file = 'tica_model.pkl' 
    joblib.dump(pca, model_file) 
