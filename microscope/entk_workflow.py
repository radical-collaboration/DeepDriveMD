from radical.entk import Pipeline, Stage, Task, AppManager
import os
import traceback
import sys
from glob import glob
import radical.utils as ru
import shutil


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

class md_pipeline(Pipeline):
    def __init__(self, name):
        super(md_pipeline, self).__init__()
        self.name = name 



class cvae_pipeline(Pipeline):
    def __init__(self, name):
        super(cvae_pipelinevae, self).__init__()
        self.name = name 


class md_stage(Stage):
    def __init__(self, name):
        super(md_stage, self).__init__()
        self.name = name
        self.post_exec = {
        'condition': self.func_condition(),
        'on_true': self.func_on_true(),
        'on_false': self.func_on_false()
        }

    def func_condition(self):

        self.parent_pipeline.suspend()
        print 'Suspending pipeline %s for 10 seconds' %self.parent_pipeline.uid
        sleep(10)
        return True

    def func_on_true(self):

        print 'Resuming pipeline %s' %self.parent_pipeline.uid
        self.parent_pipeline.resume()

    def func_on_false(self):
        pass

        
class cvae_train_stage(Stage):
    def __init__(self, name):
        super(cvae_stage, self).__init__()
        self.name = name

class cvae_inference_stage(Stage):
    def __init__(self, name):
        super(cvae_stage, self).__init__()
        self.name = name
        self.post_exec = {
        'condition': self.func_condition(),
        'on_true': self.func_on_true(),
        'on_false': self.func_on_false()
    }

    def func_condition(self):

        self.parent_pipeline.suspend()
        print 'Suspending pipeline %s for 10 seconds' %self.parent_pipeline.uid
        sleep(10)
        return True

    def func_on_true(self):

        print 'Resuming pipeline %s' %self.parent_pipeline.uid
        self.parent_pipeline.resume()

    def func_on_false(self):
        pass



class md_task(Task):
    def __init__(self, name, duration):
        super(md_task, self).__init__()
        self.name = name   
        self.pre_exec = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        self.executable = ['stress-ng']
        self.arguments = ['-c', '28', '-t', '{}'.format(duration)]
        self.cpu_reqs = {'processes': 1, 'thread_type': None, 'threads_per_process': 32, 'process_type': None}  
        self.gpu_reqs = {'processes': 1}  
        self.copy_input_data = []


class cvae_training_task(Task):
    def __init__(self, name, hyperspace_index):
        super(cvae_task, self).__init__()
        self.name = name   
        self.pre_exec = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        self.executable = ['stress-ng']
        self.arguments = ['-c', '24', '-t', '6000'] 
        self.cpu_reqs = {'processes': 1, 'thread_type': None, 'threads_per_process': 32, 'process_type': None}
        self.gpu_reqs = {'processes': 1}
        self.copy_input_data = []


class cvae_inference_task(Task):
    def __init__(self, name):
        super(cvae_task, self).__init__()
        self.name = name   
        self.pre_exec = ['export PATH=/home/dakka/stress-ng-0.09.40:$PATH']
        self.executable = ['stress-ng']
        self.arguments = ['-c', '24', '-t', '6000'] 
        self.cpu_reqs = {'processes': 1, 'thread_type': None, 'threads_per_process': 32, 'process_type': None}
        self.copy_input_data = []


if __name__ == '__main__':

    # arguments for AppManager


    walltime = 30
    num_hyperparameters = 5
    initial_hparams = [(0,7)] 
    final_hparams = list()
    final_hparams += number_of_hyperparameters * [initial_hparams[0]]

    # Pipelines, Stages, Tasks 

    md_p = md_pipeline(name = 'MD_simulation_pipeline')
    cvae_p = cvae_pipe(name = 'CVAE_pipeline')

    md_s1 = md_stage(name = 'MD_simulation_long_dur')
    md_s2 = md_stage(name = 'MD_simulation_short_dur')

    cvae_train_s = cvae_train_stage(name = 'cvae_training_stage')
    cvae_inference_s = cvae_inference_stage(name = 'cvae_inference_stage')


    for i in range(28):

        # execute BoT simulations for 6 ns (long duration)
        # this will generate enough initial trajectory data for cvae training

        t = md_task(name 'md_long_task_{}'.format(i), duration = 6000)

        # Add the Task to the Stage

        md_s1.add_tasks(t)


    # Add Stage to the Pipeline

    md_p.add_stages(md_s1)

    logger.info('adding stage {} with {} tasks'.format(md_s1.name, md_s1._task_count))


    for i in range(28):

        # execute BoT simulations for 1 ns (short duration)
        

        t = md_task(name 'md_short_task_{}'.format(i), duration = 600)

        # Add the Task to the Stage

        md_s2.add_tasks(t)


    # Add Stage to the Pipeline

    md_p.add_stages(md_s2)

    logger.info('adding stage {} with {} tasks'.format(md_s2.name, md_s2._task_count))


    for i in range(len(hyperparameters)**2):
        t = cvae_training_task(name = 'cvae_training_task', hyperparameters = final_hparams)  
        


    for i in range(2**len(final_hparams)): 
    
        # run Bayesian optimization in parallel
        # each optimization runs for n_iterations

        t = cvae_training_task(name = 'optimization_{}'.format(i), hyperspace_index = i)
        
        cvae_train_s.add_tasks(t)
        
    cvae_p.add_stages(cvae_train_s2)


    # Create Application Manager
    appman = AppManager(hostname=hostname, port=port)

    res_dict = {

            'resource': 'local.localhost',
            'walltime': 1,
            'cpus': 1
    }


    # Assign resource manager to the Application Manager
    appman.resource_desc = res_dict
    appman.shared_data = []

    
    # Assign the workflow as a set of Pipelines to the Application Manager
    appman.workflow = [md_p,cvae_p]

    # Run the Application Manager
    appman.run()

    