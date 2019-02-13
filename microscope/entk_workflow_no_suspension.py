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

        self.processes = 1
        self.gpu_processes = None
        self.threads_per_process = 28
        self.gpu_threads_per_process = None
        self.duration = 120 
        self.p = Pipeline()
        self.p.name = name
        self.md_tasks = 3
        
    @property
    def md_cpus(self):
        return self.processes * self.threads_per_process * self.md_tasks

    def generate_task(self, task_no):
        
        task = Task()
        task.name = 'md_task_{}'.format(task_no)

        task.pre_exec = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        task.executable = ['stress-ng']
        task.arguments = ['-c', '{}'.format(self.threads_per_process), '-t', '{}'.format(self.duration)]
        task.cpu_reqs = {'processes': self.processes,
                         'process_type': None,
                         'threads_per_process': self.threads_per_process,
                         'thread_type': None
                         }

        task.gpu_reqs = {'processes': self.gpu_processes,
                         'process_type': None,
                         'threads_per_process': self.gpu_threads_per_process,
                         'thread_type': None
                         }

        return task

    def generate_stage(self):
        s = Stage()
        s.name = 'MD_stage'
        # s.add_tasks(self.generate_task())
        for i in range(self.md_tasks):
            s.add_tasks(self.generate_task(i)) 
        return s

    def generate_pipeline(self):
        self.p.add_stages(self.generate_stage())

        return self.p

class CVAE_pipeline:

    def __init__(self, name = 'CVAE_pipeline'):

        self.name = name
        self.engine = None

        self.processes = 1
        self.gpu_processes = None
        self.threads_per_process = 28
        self.gpu_threads_per_process = None
        self.duration = 120 
        self.p = Pipeline()
        self.p.name = name
        self.cvae_tasks = 3
    
    @property
    def cvae_cpus(self):
        return self.processes * self.threads_per_process * self.cvae_tasks  

    def generate_task(self, task_no):
        
        task = Task()
        # task.name = 'cvae_train_task'
        task.name = 'cvae_train_task_{}'.format(task_no)

        task.pre_exec = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        task.executable = ['stress-ng'] 
        task.arguments = ['-c', '{}'.format(self.threads_per_process), '-t', '{}'.format(self.duration)]
        task.cpu_reqs = {'processes': self.processes,
                         'process_type': None,
                         'threads_per_process': self.threads_per_process,
                         'thread_type': None
                         }

        task.gpu_reqs = {'processes': self.gpu_processes,
                         'process_type': None,
                         'threads_per_process': self.gpu_threads_per_process,
                         'thread_type': None
                         }

        return task

    def generate_stage(self):
        s = Stage()
        s.name = 'MD_stage'
        for i in range(self.cvae_tasks):
            s.add_tasks(self.generate_task(i))
        # s.add_tasks(self.generate_task())
        return s

    def generate_pipeline(self):

        self.p.add_stages(self.generate_stage())
        return self.p


if __name__ == '__main__':

    # Create Application Manager
    appman = AppManager(hostname=hostname, port=port)
    

    # Create MD pipeline, specify number of simulations (md_tasks)
    md_p = MD_pipeline() 
    p1 = md_p.generate_pipeline()

    # Create CVAE pipeline, specify number of hyperparameters 
    cvae_p = CVAE_pipeline() 
    p2 = cvae_p.generate_pipeline()

    # Specify the resource request
    total_cpus = md_p.md_cpus + cvae_p.cvae_cpus
    # On Bridges every 2 GPUs (P100) will land on 32 CPUs
    # total_cpus = (total_gpus/2)*32

    pipelines = []
    pipelines.append(p1)
    pipelines.append(p2)

    # Assign the workflow as a set of Pipelines to the Application Manager
    appman.workflow = set(pipelines)

    # Create a dictionary describe four mandatory keys:
    # resource, walltime, cores and project
    # resource is 'local.localhost' to execute locally
    # res_dict = {

    #     'resource': 'local.localhost',
    #     'walltime': 15,
    #     'cpus': 2,
    # }

    res_dict = {

        'resource': 'xsede.bridges',
        'project' : 'mc3bggp',
        'queue' : 'RM',
        'walltime': md_p.duration + 10,
        'cpus': total_cpus,
        'access_schema': 'gsissh'
    }

    appman.resource_desc = res_dict

    # Run the Application Manager
    appman.run()


