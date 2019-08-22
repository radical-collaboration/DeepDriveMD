import os, glob 
import shutil


omm_dirs = glob.glob('MD_exps/fs-pep/omm_runs*') 
cvae_dirs = glob.glob('CVAE_exps/cvae_runs_*') 
jsons = glob.glob('Outlier_search/*json') 

for omm_dir in omm_dirs: 
    shutil.move(omm_dir, 'MD_exps/fs-pep/old_MDs/') 

for cvae_dir in cvae_dirs: 
    shutil.rmtree(cvae_dir) 

for json in jsons: 
    os.remove(json) 

if os.path.isdir('Outlier_search/outlier_pdbs'): 
    shutil.rmtree('Outlier_search/outlier_pdbs') 


