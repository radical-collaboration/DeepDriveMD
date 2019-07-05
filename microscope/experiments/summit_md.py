import os, time 
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
# export RMQ_HOSTNAME=two.radical-project.org 
# export RMQ_PORT=33235 
# export RADICAL_PILOT_DBURL=mongodb://user:user@ds223760.mlab.com:23760/adaptivity 
#
#


def generate_training_pipeline():

    p = Pipeline()
    p.name = 'MD_ML'

    # --------------------------
    # MD stage
    s1 = Stage()
    s1.name = 'MD'

    # MD tasks
    for i in range(2):
        t1 = Task()
        # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/MD_exps/fs-pep/run_openmm.py
        t1.pre_exec = [] 
        t1.pre_exec += ['module load cuda/9.1.85']
        t1.pre_exec += ['source activate omm'] 
        t1.pre_exec += ['export PYTHONPATH=/gpfs/alpine/scratch/hm0/bip179/entk_test/hyperspace/microscope/experiments/MD_exps:$PYTHONPATH'] 
        t1.pre_exec += ['export CUDA_VISIBLE_DEVICES=0'] 
        t1.pre_exec += ['cd /gpfs/alpine/bip179/scratch/hm0/entk_test/hyperspace/microscope/experiments/MD_exps/fs-pep'] 
        time_stampe = int(time.time())
        t1.pre_exec += ['mkdir -p omm_runs_%d && cd omm_runs_%d' % (time_stampe, time_stampe)]
#         t1.pre_exec += ['which python']
#         t1.executable = ['/ccs/home/hm0/Research/CUDA/device'] 
#         t1.arguments = ['python'] 
        t1.executable = ['/ccs/home/hm0/.conda/envs/omm/bin/python']  # run_openmm.py
        t1.arguments = ['/gpfs/alpine/bip179/scratch/hm0/entk_test/hyperspace/microscope/experiments/MD_exps/fs-pep/run_openmm.py', 
                '-f', '/gpfs/alpine/bip179/scratch/hm0/entk_test/hyperspace/microscope/experiments/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb']

        t1.cpu_reqs = {'processes': 1,
                          'threads_per_process': 1,
                          'thread_type': 'OpenMP'
                          }
        t1.gpu_reqs = {'processes': 1,
                          'threads_per_process': 1,
                          'thread_type': 'CUDA'
                         }
                          
        # Add the MD task to the simulating stage
        s1.add_tasks(t1)

    # Add simulating stage to the training pipeline
    p.add_stages(s1)

    # --------------------------
    # Aggregate stage
    s2 = Stage()
    s2.name = 'aggregating'

    # Aggregation task
    t2 = Task()
    # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/MD_to_CVAE/MD_to_CVAE.py
    t2.pre_exec = [] 
    t2.pre_exec += ['source activate omm'] 
    t2.pre_exec += ['cd /gpfs/alpine/bip179/scratch/hm0/entk_test/hyperspace/microscope/experiments/MD_to_CVAE']
    t2.executable = ['/ccs/home/hm0/.conda/envs/omm/bin/python']  # MD_to_CVAE.py
    t2.arguments = ['/gpfs/alpine/bip179/scratch/hm0/entk_test/hyperspace/microscope/experiments/MD_to_CVAE/MD_to_CVAE.py', 
            '-f', '/gpfs/alpine/bip179/scratch/hm0/entk_test/hyperspace/microscope/experiments/MD_exps/fs-pep']

    # Add the aggregation task to the aggreagating stage
    s2.add_tasks(t2)

    # Add the aggregating stage to the training pipeline
    p.add_stages(s2)

    # --------------------------
    # Learning stage
    s3 = Stage()
    s3.name = 'learning'

    # learn task
    t3 = Task()
    # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/CVAE_exps/train_cvae.py
    t3.pre_exec = []
    t3.pre_exec += ['module load cuda/9.1.85']
    t3.pre_exec += ['source activate omm']
    t3.pre_exec += ['cd /gpfs/alpine/bip179/scratch/hm0/entk_test/hyperspace/microscope/experiments/CVAE_exps']
    t3.executable = ['/ccs/home/hm0/.conda/envs/omm/bin/python']  # train_cvae.py
    t3.arguments = ['/gpfs/alpine/bip179/scratch/hm0/entk_test/hyperspace/microscope/experiments/CVAE_exps/train_cvae.py', 
            '-f', '/gpfs/alpine/bip179/scratch/hm0/entk_test/hyperspace/microscope/experiments/MD_to_CVAE/cvae_input.h5', 
            '-d', '3'] 
# 
#     # Add the learn task to the learning stage
#     s3.add_tasks(t3)
# 
#     # Add the learning stage to the pipeline
#     p.add_stages(s3)

    return p


def generate_MDML_pipeline():

    p = Pipeline()
    p.name = 'MDML'

    # --------------------------
    # MD stage
    s1 = Stage()
    s1.name = 'simulating'

    # MD tasks
    for i in range(4):
        t1 = Task()
        t1.executable = ['sleep']  # MD executable
        t1.arguments = ['60']

        # Add the MD task to the Docking Stage
        s1.add_tasks(t1)

    # Add simulating stage to the pipeline
    p.add_stages(s1)

    # --------------------------
    # Aggregate stage
    s2 = Stage()
    s2.name = 'aggregating'

    # Aggregation task
    t2 = Task()
    t2.executable = ['sleep']  # Executable to aggregate Trajectories +
                               # Contact maps

    t2.arguments = ['30']

    # Add the aggregating task to the aggreagating stage
    s2.add_tasks(t2)

    # Add aggregation stage to the to the pipeline
    p.add_stages(s2)

    # --------------------------
    # Learning stage
    s3 = Stage()
    s3.name = 'inferring'

    # Inferring task
    t3 = Task()
    t3.executable = ['sleep']  # CVAE executable
    t3.arguments = ['30']

    # Add the infer task to the learning stage
    s3.add_tasks(t3)

    # Add the learning stage to the pipeline
    p.add_stages(s3)

    return p


if __name__ == '__main__':

    # Create a dictionary to describe four mandatory keys:
    # resource, walltime, cores and project
    # resource is 'local.localhost' to execute locally
    res_dict = {
            'resource': 'ornl.summit',
            'queue'   : 'batch',
            'schema'  : 'local',
            'walltime': 120,
            'cpus'    : 42,
            'gpus'    : 2,
            'project' : 'BIP179'
    }

    # Create Application Manager
    # appman = AppManager()
    appman = AppManager(hostname=os.environ.get('RMQ_HOSTNAME'), port=os.environ.get('RMQ_PORT'))
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
