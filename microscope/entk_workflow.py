from radical.entk import Pipeline, Stage, Task, AppManager
import os
import traceback
import sys
from glob import glob
import radical.utils as ru
import shutil
from time import sleep


# suspend pipelines:  
# https://github.com/radical-cybertools/radical.entk/blob/master/examples/misc/suspend_pipelines.py

# ------------------------------------------------------------------------------
# Set default verbosity

if os.environ.get('RADICAL_ENTK_VERBOSE') == None:
    os.environ['RADICAL_ENTK_VERBOSE'] = 'INFO'

cur_dir = os.path.dirname(os.path.abspath(__file__))
hostname = os.environ.get('RMQ_HOSTNAME','localhost')
port = int(os.environ.get('RMQ_PORT',5672))

logger = ru.Logger(__name__, level='DEBUG')


class MD_pipeline:

    def __init__(self, name = 'MD_pipeline'):

        self.name = name
        self.engine = None # OpenMM
        self.system = None 

        # benchmark example for OpenMM: 
        # https://github.com/radical-cybertools/htbac/blob/master/examples/inputs/benchmark.py#L17-L18 

        self.processes = None
        self.gpu_processes = None
        self.threads_per_process = None
        self.gpu_threads_per_process = None
        self.duration = None 
        self.p = Pipeline()
        self.p.name = name
        

    def generate_task(self):
        
        task = Task()
        task.name = 'md_task'

        task.pre_exec = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        task.executable = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        task.arguments = ['-c', '32', '-t', '{}'.format(self.duration)]
        task.cpu_reqs = {'processes': self.processes,
                         'process_type': 'MPI',
                         'threads_per_process': self.threads_per_process,
                         'thread_type': None
                         }

        task.gpu_reqs = {'processes': self.gpu_processes,
                         'process_type': 'MPI',
                         'threads_per_process': self.gpu_threads_per_process,
                         'thread_type': None
                         }

        return task

    def generate_stage(self):
        s = Stage()
        s.name = 'MD_stage'
        tasks = [s.add_tasks(self.generate_task() for i in range(28))] 
        s.post_exec = {
            'condition': self.func_condition(),
            'on_true': self.func_on_true(),
            'on_false': self.func_on_false()
        }
        return s

    def func_condition(self):

        self.p.suspend()
        print 'Suspending pipeline {} for 10 seconds'.format(self.p.name)  
        sleep(10)
        return True

    def func_on_true(self):

        print 'Resuming pipeline {}'.format(self.p.name) 
        self.p.resume()

    def func_on_false(self):
        pass

    def generate_pipeline(self):

        self.p.add_stages(self.generate_stage())

        return self.p

class CVAE_pipeline:

    def __init__(self, name = 'CVAE_pipeline'):

        self.name = name
        self.engine = None
        self.system = None

        self.processes = None
        self.gpu_processes = None
        self.threads_per_process = None
        self.gpu_threads_per_process = None
        self.duration = None 
        self.p = Pipeline()
        self.p.name = name
        

    def generate_task(self, task_no):
        
        task = Task()
        task.name = 'cvae_train_task_{}'.format(task_no)

        task.pre_exec = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        task.executable = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        task.arguments = ['-c', '32', '-t', '{}'.format(self.duration)]
        task.cpu_reqs = {'processes': self.processes,
                         'process_type': 'MPI',
                         'threads_per_process': self.threads_per_process,
                         'thread_type': None
                         }

        task.gpu_reqs = {'processes': self.gpu_processes,
                         'process_type': 'MPI',
                         'threads_per_process': self.gpu_threads_per_process,
                         'thread_type': None
                         }

        return task

    def generate_stage(self):
        s = Stage()
        s.name = 'MD_stage'
        tasks = [s.add_tasks(self.generate_task(task_no = "i") for i in range(28))] 
        s.post_exec = {
            'condition': self.func_condition(),
            'on_true': self.func_on_true(),
            'on_false': self.func_on_false()
        }
        return s

    def func_condition(self):

        self.p.suspend()
        print 'Suspending pipeline {} for 10 seconds'.format(self.p.name)  
        sleep(10)
        return True

    def func_on_true(self):

        print 'Resuming pipeline {}'.format(self.p.name) 
        self.p.resume()

    def func_on_false(self):
        pass

    def generate_pipeline(self):

        self.p.add_stages(self.generate_stage())
        return self.p



if __name__ == '__main__':

    # Create a dictionary describe four mandatory keys:
    # resource, walltime, cores and project
    # resource is 'local.localhost' to execute locally
    # res_dict = {

    #     'resource': 'local.localhost',
    #     'walltime': 15,
    #     'cpus': 2,
    # }

    hyperparams = 5 # typically 5
    walltime = 30 
    cpus_for_hyperspace = 2**len(hyperparams)*32 
    cpus_for_md = 28*32
    total_cpus = cpus_for_hyperspace + cpus_for_md

    res_dict = {

        'resource': 'xsede.bridges',
        'project' : 'mc3bggp',
        'queue' : 'GPU',
        'walltime': walltime,
        'cpus': total_cpus,
        'access_schema': 'gsissh'
    }


    # Create Application Manager
    appman = AppManager(hostname=hostname, port=port)
    appman.resource_desc = res_dict

    md_p = MD_pipeline() 
    p1 = md_p.generate_pipeline()


    cvae_p = CVAE_pipeline() 
    p2 = cvae_p.generate_pipeline()


    # Assign the workflow as a set of Pipelines to the Application Manager
    appman.workflow = [p]

    # Run the Application Manager
    appman.run()


