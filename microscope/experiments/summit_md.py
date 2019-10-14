import os, json, time 
from radical.entk import Pipeline, Stage, Task, AppManager

# ------------------------------------------------------------------------------
# Set default verbosity

if os.environ.get('RADICAL_ENTK_VERBOSE') is None:
    os.environ['RADICAL_ENTK_REPORT'] = 'True'

# Assumptions:
# - # of MD steps: 2
# - Each MD step runtime: 15 minutes
# - Summit's scheduling policy [1]
#
# Resource rquest:
# - 4 <= nodes with 2h walltime.
#
# Workflow [2]
#
# [1] https://www.olcf.ornl.gov/for-users/system-user-guides/summit/summit-user-guide/scheduling-policy
# [2] https://docs.google.com/document/d/1XFgg4rlh7Y2nckH0fkiZTxfauadZn_zSn3sh51kNyKE/
#
'''
export RMQ_HOSTNAME=two.radical-project.org 
export RMQ_PORT=33235 
export RADICAL_PILOT_DBURL=mongodb://hyperrct:h1p3rrc7@two.radical-project.org:27017/hyperrct 
export RADICAL_PILOT_PROFILE=True
export RADICAL_ENTK_PROFILE=True
'''
#

base_path = os.path.abspath('.') # '/gpfs/alpine/proj-shared/bip179/entk/hyperspace/microscope/experiments/'
conda_path = '/ccs/home/hm0/.conda/envs/omm'

CUR_STAGE=0
MAX_STAGE=10
RETRAIN_FREQ = 5

LEN_initial = 100
LEN_iter = 10 

def generate_training_pipeline():
    """
    Function to generate the CVAE_MD pipeline
    """

    def generate_MD_stage(num_MD=1): 
        """
        Function to generate MD stage. 
        """
        s1 = Stage()
        s1.name = 'MD'
        initial_MD = True 
        outlier_filepath = '%s/Outlier_search/restart_points.json' % base_path

        if os.path.exists(outlier_filepath): 
            initial_MD = False 
            outlier_file = open(outlier_filepath, 'r') 
            outlier_list = json.load(outlier_file) 
            outlier_file.close() 

        # MD tasks
        time_stamp = int(time.time())
        for i in range(num_MD):
            t1 = Task()
            # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/MD_exps/fs-pep/run_openmm.py
            t1.pre_exec = ['. /sw/summit/python/2.7/anaconda2/5.3.0/etc/profile.d/conda.sh']
            t1.pre_exec += ['module load cuda/9.1.85']
            t1.pre_exec += ['conda activate %s' % conda_path] 
            t1.pre_exec += ['export PYTHONPATH=%s/MD_exps:$PYTHONPATH' % base_path] 
            t1.pre_exec += ['cd %s/MD_exps/fs-pep' % base_path] 
            t1.pre_exec += ['mkdir -p omm_runs_%d && cd omm_runs_%d' % (time_stamp+i, time_stamp+i)]
            t1.executable = ['%s/bin/python' % conda_path]  # run_openmm.py
            t1.arguments = ['%s/MD_exps/fs-pep/run_openmm.py' % base_path]
          #   t1.arguments += ['--topol', '%s/MD_exps/fs-pep/pdb/topol.top' % base_path]


            # pick initial point of simulation 
            if initial_MD or i >= len(outlier_list): 
                t1.arguments += ['--pdb_file',
                        '%s/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb' % base_path]
#                 t1.arguments += ['--length', LEN_initial] 
#                print "Running from initial frame for %d ns. " % LEN_initial
            elif outlier_list[i].endswith('pdb'): 
                t1.arguments += ['--pdb_file', outlier_list[i]] 
#                 t1.arguments += ['--length', LEN_iter] 
                t1.pre_exec += ['cp %s ./' % outlier_list[i]]  
#                print "Running from outlier %s for %d ns" % (outlier_list[i], LEN_iter) 
            elif outlier_list[i].endswith('chk'): 
                t1.arguments += ['--pdb_file',
                        '%s/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb' % base_path,
                        '-c', outlier_list[i]] 
#                 t1.arguments += ['--length', LEN_iter]
                t1.pre_exec += ['cp %s ./' % outlier_list[i]]
#                print "Running from checkpoint %s for %d ns" % (outlier_list[i], LEN_iter) 

            # how long to run the simulation 
            if initial_MD: 
                t1.arguments += ['--length', LEN_initial] 
            else: 
                t1.arguments += ['--length', LEN_iter]

            # assign hardware the task 
            t1.cpu_reqs = {'processes': 1,
                           'process_type': None,
                              'threads_per_process': 4,
                              'thread_type': 'OpenMP'
                              }
            t1.gpu_reqs = {'processes': 1,
                           'process_type': None,
                              'threads_per_process': 1,
                              'thread_type': 'CUDA'
                             }
                              
            # Add the MD task to the simulating stage
            s1.add_tasks(t1)
        return s1 


    def generate_aggregating_stage(): 
        """ 
        Function to concatenate the MD trajectory (h5 contact map) 
        """ 
        s2 = Stage()
        s2.name = 'aggregating'

        # Aggregation task
        t2 = Task()
        # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/MD_to_CVAE/MD_to_CVAE.py
        t2.pre_exec = [] 
        t2.pre_exec += ['. /sw/summit/python/2.7/anaconda2/5.3.0/etc/profile.d/conda.sh']
        t2.pre_exec += ['conda activate %s' % conda_path] 
        t2.pre_exec += ['cd %s/MD_to_CVAE' % base_path]
        t2.executable = ['%s/bin/python' % conda_path]  # MD_to_CVAE.py
        t2.arguments = ['%s/MD_to_CVAE/MD_to_CVAE.py' % base_path, 
                '--sim_path', '%s/MD_exps/fs-pep' % base_path]

        # Add the aggregation task to the aggreagating stage
        s2.add_tasks(t2)
        return s2 


    def generate_ML_stage(num_ML=1): 
        """
        Function to generate the learning stage
        """
        s3 = Stage()
        s3.name = 'learning'

        # learn task
        time_stamp = int(time.time())
        for i in range(num_ML): 
            t3 = Task()
            # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/CVAE_exps/train_cvae.py
            t3.pre_exec = []
            t3.pre_exec += ['. /sw/summit/python/2.7/anaconda2/5.3.0/etc/profile.d/conda.sh']
            t3.pre_exec += ['module load cuda/9.1.85']
            t3.pre_exec += ['conda activate %s' % conda_path] 

            t3.pre_exec += ['export PYTHONPATH=%s/CVAE_exps:$PYTHONPATH' % base_path]
            t3.pre_exec += ['cd %s/CVAE_exps' % base_path]
            dim = i + 3 
            cvae_dir = 'cvae_runs_%.2d_%d' % (dim, time_stamp+i) 
            t3.pre_exec += ['mkdir -p {0} && cd {0}'.format(cvae_dir)]
            t3.executable = ['%s/bin/python' % conda_path]  # train_cvae.py
            t3.arguments = ['%s/CVAE_exps/train_cvae.py' % base_path, 
                    '--h5_file', '%s/MD_to_CVAE/cvae_input.h5' % base_path, 
                    '--dim', dim] 
            
            t3.cpu_reqs = {'processes': 1,
                           'process_type': None,
                    'threads_per_process': 4,
                    'thread_type': 'OpenMP'
                    }
            t3.gpu_reqs = {'processes': 1,
                           'process_type': None,
                    'threads_per_process': 1,
                    'thread_type': 'CUDA'
                    }
        
            # Add the learn task to the learning stage
            s3.add_tasks(t3)
        return s3 


    def generate_interfacing_stage(): 
        s4 = Stage()
        s4.name = 'scanning'

        # Scaning for outliers and prepare the next stage of MDs 
        t4 = Task() 
        t4.pre_exec = [] 
        t4.pre_exec += ['. /sw/summit/python/2.7/anaconda2/5.3.0/etc/profile.d/conda.sh']
        t4.pre_exec += ['module load cuda/9.1.85']
        t4.pre_exec += ['conda activate %s' % conda_path] 

        t4.pre_exec += ['export PYTHONPATH=%s/CVAE_exps:$PYTHONPATH' % base_path] 
        t4.pre_exec += ['cd %s/Outlier_search' % base_path] 
        t4.executable = ['%s/bin/python' % conda_path] 
        t4.arguments = ['outlier_locator.py', '--md', '../MD_exps/fs-pep', '--cvae', '../CVAE_exps', '
                --pdb', '../MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb', 
                '--ref', '../MD_exps/fs-pep/pdb/fs-peptide.pdb']

        t4.cpu_reqs = {'processes': 1,
                           'process_type': None,
                'threads_per_process': 12,
                'thread_type': 'OpenMP'
                }
        t4.gpu_reqs = {'processes': 1,
                           'process_type': None,
                'threads_per_process': 1,
                'thread_type': 'CUDA'
                }
        s4.add_tasks(t4) 
        s4.post_exec = func_condition 
        
        return s4


    def func_condition(): 
        global CUR_STAGE, MAX_STAGE 
        if CUR_STAGE < MAX_STAGE: 
            func_on_true()
        else:
            func_on_false()

    def func_on_true(): 
        global CUR_STAGE, MAX_STAGE
        print 'finishing stage %d of %d' % (CUR_STAGE, MAX_STAGE) 
        
        # --------------------------
        # MD stage
        s1 = generate_MD_stage(num_MD=6 * 20)
        # Add simulating stage to the training pipeline
        p.add_stages(s1)

        if CUR_STAGE % RETRAIN_FREQ == 0: 
            # --------------------------
            # Aggregate stage
            s2 = generate_aggregating_stage() 
            # Add the aggregating stage to the training pipeline
            p.add_stages(s2)

            # --------------------------
            # Learning stage
            s3 = generate_ML_stage(num_ML=10) 
            # Add the learning stage to the pipeline
            p.add_stages(s3)

        # --------------------------
        # Outlier identification stage
        s4 = generate_interfacing_stage() 
        p.add_stages(s4) 
        
        CUR_STAGE += 1

    def func_on_false(): 
        print 'Done' 



    p = Pipeline()
    p.name = 'MD_ML'

    # --------------------------
    # MD stage
    s1 = generate_MD_stage(num_MD=6 * 20)
    # Add simulating stage to the training pipeline
    p.add_stages(s1)

    # --------------------------
    # Aggregate stage
    s2 = generate_aggregating_stage() 
    # Add the aggregating stage to the training pipeline
    p.add_stages(s2)

    # --------------------------
    # Learning stage
    s3 = generate_ML_stage(num_ML=10) 
    # Add the learning stage to the pipeline
    p.add_stages(s3)

    # --------------------------
    # Outlier identification stage
    s4 = generate_interfacing_stage() 
    p.add_stages(s4) 

    CUR_STAGE += 1

    return p




if __name__ == '__main__':

    # Create a dictionary to describe four mandatory keys:
    # resource, walltime, cores and project
    # resource is 'local.localhost' to execute locally
    res_dict = {
            'resource': 'ornl.summit',
            'queue'   : 'killable',
            'schema'  : 'local',
            'walltime': 60 * 12,
            'cpus'    : 42 * 20,
            'gpus'    : 6 * 20,#6*2 ,
            'project' : 'BIP179'
    }

    # Create Application Manager
    # appman = AppManager()
    appman = AppManager(hostname=os.environ.get('RMQ_HOSTNAME'), port=int(os.environ.get('RMQ_PORT')))
    appman.resource_desc = res_dict

    p1 = generate_training_pipeline()
    # p2 = generate_MDML_pipeline()

    pipelines = []
    pipelines.append(p1)
    # pipelines.append(p2)

    # Assign the workflow as a list of Pipelines to the Application Manager. In
    # this way, all the pipelines in the list will execute concurrently.
    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
