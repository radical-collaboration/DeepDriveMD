import os, glob 
import sys 
import shutil
import time

if len(sys.argv) > 1: 
    status = sys.argv[1] 
else: 
    status = 'fail'

print status 
omm_dirs = glob.glob('MD_exps/fs-pep/omm_runs*') 
cvae_dirs = glob.glob('CVAE_exps/cvae_runs_*') 
tica_dirs = glob.glob('TICA_exps/tica_runs_*') 
jsons = glob.glob('Outlier_search/*json') 

result_save = os.path.join('./results', 'result_%d_%s' % (int(time.time()), status)) 
os.makedirs(result_save) 

omm_save = os.path.join(result_save, 'omm_results') 
os.makedirs(omm_save) 
for omm_dir in omm_dirs: 
    shutil.move(omm_dir, omm_save) 

cvae_save = os.path.join(result_save, 'cvae_results') 
os.makedirs(cvae_save) 
for cvae_dir in cvae_dirs: 
    shutil.move(cvae_dir, cvae_save) 

tica_save = os.path.join(result_save, 'tica_results') 
os.makedirs(tica_save) 
for tica_dir in tica_dirs: 
    shutil.move(tica_dir, tica_save) 


outlier_save = os.path.join(result_save, 'outlier_save/') 
os.makedirs(outlier_save) 
for json in jsons:  
    shutil.move(json, outlier_save) 

if os.path.isdir('Outlier_search/outlier_pdbs'): 
    shutil.move('Outlier_search/outlier_pdbs', outlier_save) 

sandbox_path = '/gpfs/alpine/bip179/scratch/hm0/radical.pilot.sandbox' 
local_entk_path = sorted(glob.glob('re.session.*'))[-1] 
shutil.move(local_entk_path, result_save) 
sandbox_src = os.path.join(sandbox_path, local_entk_path) 
sandbox_dst = os.path.join(result_save, local_entk_path + '_sandbox') 
shutil.copytree(sandbox_src, sandbox_dst) 


