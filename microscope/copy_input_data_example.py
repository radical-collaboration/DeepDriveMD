import os
from radical.entk import Pipeline, Stage, Task, AppManager

# ------------------------------------------------------------------------------
# Set default verbosity

if os.environ.get('RADICAL_ENTK_VERBOSE') is None:
    os.environ['RADICAL_ENTK_REPORT'] = 'True'


CUR_NEW_STAGE = 0

# For local testing
MAX_NEW_STAGE = 3


def generate_MD_pipeline():

    def describe_MD_pipline():
        p = Pipeline()
        p.name = 'MD'

        # MD stage
        s1 = Stage()
        s1.name = 'OpenMM'

        # Each Task() is an OpenMM executable that will run on a single GPU.
        # Set sleep time for local testing

        t1 = Task()
        t1.name = 'fs_pep'

        t1.pre_exec = []
        t1.pre_exec   += ['module load gcc/5.3.0']
        t1.pre_exec   += ['module load cuda/9.0']
        t1.pre_exec   += ['export PATH="/pylon5/mc3bggp/dakka/miniconda2/bin:$PATH"']
        t1.pre_exec   += ['source activate cvae']
        t1.executable = ['python']
        t1.arguments = ['/pylon5/mc3bggp/dakka/openmm_benchmark/fs-pep/simulation.py']
        t1.cpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 1,
                         'thread_type': None
                         }

        t1.gpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 1,
                         'thread_type': None
                         }

        # Add the MD task to the Docking Stage
        s1.add_tasks(t1)

        # Add MD stage to the MD Pipeline
        p.add_stages(s1)

        s2 = Stage()
        s2.name = 'collection_stage'

        t2 = Task()
        t2.name = 'collection_traj'

        t2.executable = ['/bin/bash']    
        t2.arguments = ['-l', '-c', 'grep -o . output.txt | sort | uniq -c > ccount.txt']  
        t2.copy_input_data = ['$Pipeline_%s_Stage_%s_Task_%s/output.log'%(p.uid, s1.uid, t1.uid)]
        t2.cpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 1,
                         'thread_type': None
                         }

        s2.add_tasks(t2) 

        p.add_stages(s2)

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


if __name__ == '__main__':

    
    res_dict = {

            'resource': 'xsede.bridges',
            'project' : 'mc3bggp',
            'queue' : 'GPU',
            'walltime': 60,
            'cpus': 32,
            'gpus': 1,
            'access_schema': 'gsissh'
    }

    # Create Application Manager
    appman = AppManager()
    appman.resource_desc = res_dict

    p = generate_MD_pipeline()

    pipelines = []
    pipelines.append(p)

    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
