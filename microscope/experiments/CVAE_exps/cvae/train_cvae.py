import os, sys, errno
import argparse 
from CVAE import run_cvae  



parser = argparse.ArgumentParser()
parser.add_argument("-f", help="Input: contact map h5 file")
# parser.add_argument("-o", help="output: cvae weight file. (Keras cannot load model directly, will check again...)")
parser.add_argument("-d", "--dim", help="Number of dimensions in latent space") 
parser.add_argument("-gpu", help="gpu_id")

args = parser.parse_args()

if args.f: 
    cvae_input = os.path.abspath(args.f) 
else: 
    raise IOError('No input file...') 

if args.dim: 
    hyper_dim = args.dim 
else: 
    hyper_dim = 3

if args.gpu: 
    gpu_id = args.gpu 
else: 
    gpu_id = 0 


if __name__ == '__main__': 
    cvae = run_cvae(gpu_id, cvae_input, hyper_dim=hyper_dim)

    model_weight = os.path.join(model_dir, 'cvae_weight.h5')
    model_file = os.path.join(model_dir, 'cvae_model.h5')

    cvae.model.save_weights(model_weight)
    cvae.save(model_file)
