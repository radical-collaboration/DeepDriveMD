import os
from radical.entk import Pipeline, Stage, Task, AppManager

# ------------------------------------------------------------------------------
# Set default verbosity

if os.environ.get('RADICAL_ENTK_VERBOSE') is None:
    os.environ['RADICAL_ENTK_REPORT'] = 'True'

# Assumptions:
# - Each MD step runs for ~2h
# - Overheads+other workflow steps 0.5h
# - <= 300 concurrent tasks
# - Summit's scheduling policy [1]
#
# Resource rquest:
# - 46 <= nodes < 92 with 6h walltime.
#
# Workflow [2]:
# - 46 <= pipelines < 91
# - 2*45 <= Docking/MD stages < 2*90 (2 * (45 concurrent 2 hours-long stages)
#   limited by 6h walltime)
#
# The workflow has two types of pipelines: Head and MD. The Head Pipeline
# consists of 1 Stage with 2 Tasks: Generator and ML/AL. The MD Pipeline
# consists of 2 stages: the 1st stage has 1 Docking task; the 2nd stage has 6
# OpenMM tasks, each using 1 GPU.
#
# [1] https://www.olcf.ornl.gov/for-users/system-user-guides/summit/summit-
#     user-guide/#scheduling-policy
# [2] https://docs.google.com/drawings/d/1vxudWZtKrF6-
#     O_eGLuQkmzMC9T8HbEJCpYbRFZ3ipnw/


CUR_NEW_STAGE = 0
# MAX_NEW_STAGE = 90

# For local testing
MAX_NEW_STAGE = 3


def generate_MD_pipeline():

    def describe_MD_pipline():
        p = Pipeline()
        p.name = 'MD'

        # MD stage
        s = Stage()
        s.name = 'Simulation'

        # Each Task() is an OpenMM executable that will run on a single GPU.
        # Set sleep time for local testing
        for i in range(18):

            task = Task()
            task.name = 'md_{}'.format(i)
            task.pre_exec = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
            task.executable = ['stress-ng']
            task.arguments = ['-c', '32', '-t', '600']
            task.cpu_reqs = {'processes': 1,
                             'process_type': None,
                             'threads_per_process': 32,
                             'thread_type': None
                             }

            task.gpu_reqs = {'processes': 1,
                             'process_type': None,
                             'threads_per_process': 1,
                             'thread_type': None
                             }

            # Add the MD task to the Docking Stage
            s.add_tasks(task)

        # Add post-exec to the Stage
        s.post_exec = {
                            'condition': func_condition,
                            'on_true': func_on_true,
                            'on_false': func_on_false
                        }

        # Add MD stage to the MD Pipeline
        p.add_stages(s)

        return p

    def func_condition():
        '''
        Adaptive condition

        Returns true ultil MAX_NEW_STAGE is reached. MAX_NEW_STAGE is
        calculated to be achievable within the available walltime.

        Note: walltime is known but runtime is assumed. MD pipelines might be
        truncated when walltime limit is reached and the whole workflow is
        terminated by the HPC machine.
        '''
        global CUR_NEW_STAGE, MAX_NEW_STAGE

        if CUR_NEW_STAGE <= MAX_NEW_STAGE:
            return True

        return False

    def func_on_true():

        global CUR_NEW_STAGE

        CUR_NEW_STAGE += 1

        describe_MD_pipline()

    def func_on_false():
        print 'Done'

    p = describe_MD_pipline()

    return p


def generate_ML_pipeline():

    # Create a Pipeline object
    p = Pipeline()
    p.name = 'ML'

    # Create a Stage object
    s = Stage()
    s.name = 'CVAE'

    for i in range(12):

        task = Task()
        task.name = 'cvae_train_task_{}'.format(i)

        task.pre_exec = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        task.executable = ['stress-ng'] 
        task.arguments = ['-c', '32', '-t', '1200']
        task.cpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 32,
                         'thread_type': None
                         }

        task.gpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 1,
                         'thread_type': None
                         }

        s.add_tasks(task)

    # Add Stage to the Pipeline
    p.add_stages(s)

    return p


if __name__ == '__main__':

    cores = 12*32 + 18*32 
    res_dict = {

            'resource': 'xsede.bridges',
            'project' : 'mc3bggp',
            'queue' : 'RM',
            'walltime': 30,
            'walltime': 15,
            'cpus': cores,
            'access_schema': 'gsissh'
    }

    # Create Application Manager
    appman = AppManager()
    appman.resource_desc = res_dict

    p1 = generate_MD_pipeline()
    p2 = generate_ML_pipeline()

    pipelines = []
    pipelines.append(p1)
    pipelines.append(p2)

    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()