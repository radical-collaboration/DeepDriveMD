import os
from radical.entk import Pipeline, Stage, Task, AppManager

# ------------------------------------------------------------------------------
# Set default verbosity

if os.environ.get('RADICAL_ENTK_VERBOSE') is None:
    os.environ['RADICAL_ENTK_REPORT'] = 'True'


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
        s.name = 'OpenMM'

        # Each Task() is an OpenMM executable that will run on a single GPU.
        # Set sleep time for local testing
        # for i in range(3):

        task = Task()
        task.name = 'md'

        task.pre_exec = []
        task.pre_exec   += ['export PATH="/pylon5/mc3bggp/dakka/miniconda2/bin:$PATH"']
        task.pre_exec   += ['module load cuda/9.0']
        # task.pre_exec   += ['export OPENMM_CUDA_COMPILER=`which nvcc`']
        task.pre_exec   += ['source activate cvae']
        

        task.executable = ['python']
        task.arguments = ['-m', 'simtk.testInstallation']
        task.cpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 4,
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
        # s.post_exec = {
        #                     'condition': func_condition,
        #                     'on_true': func_on_true,
        #                     'on_false': func_on_false
        #                 }

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

    # for i in range(3):

    task = Task()
    task.name = 'cvae_train_task_'

    task.pre_exec = []
    # task.pre_exec += ['module purge']
    # task.pre_exec += ['module load tensorflow/1.5_gpu']
    # task.pre_exec += ['source activate']
    
    task.pre_exec   += ['module load keras/2.2.0_tf1.7_py2_gpu tensorflow/1.7_py2_gpu']
    task.pre_exec   += ['module load cuda/9.0']
    task.pre_exec   += ['export PATH="/pylon5/mc3bggp/dakka/miniconda2/bin:$PATH"']
    task.pre_exec   += ['source activate cvae']
    task.executable = ['python'] 
    task.arguments  = ['/pylon5/mc3bggp/dakka/test_tf.py']
    task.cpu_reqs   = {'processes': 1,
                     'process_type': None,
                     'threads_per_process': 4,
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

    
    res_dict = {

            'resource': 'xsede.bridges',
            'project' : 'mc3bggp',
            'queue' : 'GPU',
            'walltime': 20,
            'cpus': 32,
            'gpus': 2,
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
