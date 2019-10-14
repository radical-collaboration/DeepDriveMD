import os, glob 
import shutil
import time

omm_dirs = glob.glob('MD_exps/fs-pep/omm_runs*') 
cvae_dirs = glob.glob('CVAE_exps/cvae_runs_*') 
jsons = glob.glob('Outlier_search/*json') 

result_save = os.path.join('./results', 'result_%d' % int(time.time())) 
os.makedirs(result_save) 

for omm_dir in omm_dirs: 
    shutil.move(omm_dir, result_save) 

for cvae_dir in cvae_dirs: 
    shutil.move(cvae_dir, result_save) 

for json in jsons: 
    shutil.move(json, result_save) 

if os.path.isdir('Outlier_search/outlier_pdbs'): 
    shutil.move('Outlier_search/outlier_pdbs', result_save) 


