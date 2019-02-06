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

    def __init__(self, name = 'MD_pipeline', md_tasks = None):

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
        self.md_sim_tasks = md_tasks
        
    @property
    def md_gpus(self):
        return self.gpu_processes * self.gpu_threads_per_process * self._md_sim_tasks

    def generate_task(self, md_task):
        
        task = Task()
        task.name = 'md_task_{}'.format(md_task)

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
        s.add_tasks(self.generate_task(i) for i in range(self.md_sim_tasks)) 
        s.post_exec = {
            'condition': self.func_condition(),
            'on_true': self.func_on_true(),
            'on_false': self.func_on_false()
        }
        return s

    def func_condition(self):

        # self.p.suspend()
        print 'Suspending pipeline {} for 5 seconds'.format(self.p.name)  
        sleep(5)
        return True

    def func_on_true(self):

        print 'Resuming pipeline {}'.format(self.p.name) 
        # self.p.resume()

    def func_on_false(self):
        pass

    def generate_pipeline(self):

        self.p.add_stages(self.generate_stage())

        return self.p

class CVAE_pipeline:

    def __init__(self, name = 'CVAE_pipeline', no_hyperparameters = None):

        self.name = name
        self.engine = None

        self.processes = None
        self.gpu_processes = None
        self.threads_per_process = None
        self.gpu_threads_per_process = None
        self.duration = None 
        self.p = Pipeline()
        self.p.name = name
        self.cvae_tasks = 2**no_hyperparameters
    
    @property
    def cvae_gpus(self):
        return self.gpu_processes * self.gpu_threads_per_process * self.cvae_tasks  

    def generate_task(self, task_no):
        
        task = Task()
        task.name = 'cvae_train_task'

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
        s.add_tasks(self.generate_task())
        s.post_exec = {
            'condition': self.func_condition(),
            'on_true': self.func_on_true(),
            'on_false': self.func_on_false()
        }
        return s

    def func_condition(self):

        # self.p.suspend()
        print 'Suspending pipeline {} for 10 seconds'.format(self.p.name)  
        sleep(5)
        return True

    def func_on_true(self):

        print 'Resuming pipeline {}'.format(self.p.name) 
        # self.p.resume()

    def func_on_false(self):
        pass

    def generate_pipeline(self):

        self.p.add_stages(self.generate_stage())
        return self.p



if __name__ == '__main__':

    # Create Application Manager
    appman = AppManager(hostname=hostname, port=port)
    appman.resource_desc = res_dict

    # Create MD pipeline, specify number of simulations (md_tasks)
    md_p = MD_pipeline() 
    p1 = md_p.generate_pipeline(md_tasks = 4)

    # Create CVAE pipeline, specify number of hyperparameters 
    cvae_p = CVAE_pipeline() 
    p2 = cvae_p.generate_pipeline(no_hyperparameters = 5)

    # Specify the resource request
    total_gpus = md_gpus + cvae_gpus

    # On Bridges every 2 GPUs (P100) will land on 32 CPUs
    total_cpus = (total_gpus/2)*32

    pipelines = set()
    pipelines.add(p1,p2)

    # Assign the workflow as a set of Pipelines to the Application Manager
    appman.workflow = [pipelines]

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
        'queue' : 'GPU',
        'walltime': walltime,
        'cpus': total_cpus,
        'gpus': total_gpus, 
        'access_schema': 'gsissh'
    }


    # Run the Application Manager
    appman.run()


