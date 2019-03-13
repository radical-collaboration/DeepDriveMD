import os
from radical.entk import Pipeline, Stage, Task, AppManager

# ------------------------------------------------------------------------------
# Set default verbosity
hostname = os.environ.get('RMQ_HOSTNAME', 'csc190specfem.marble.ccs.ornl.gov')
port = int(os.environ.get('RMQ_PORT', 30672))

if os.environ.get('RADICAL_ENTK_VERBOSE') is None:
    os.environ['RADICAL_ENTK_REPORT'] = 'True'


def generate_MD_pipeline():

    def describe_MD_pipline():
        p = Pipeline()
        p.name = 'MD'

        # MD stage
        s1 = Stage()
        s1.name = 'OpenMM'

        # Each Task() is an OpenMM executable that will run on a single GPU.
        # Set sleep time for local testing
        for i in range(18):

            task = Task()
            task.name = 'md_{}'.format(i) 

            task.pre_exec    = []

            task.pre_exec   += ['export MINICONDA=/gpfs/alpine/scratch/jdakka/bip178/miniconda']
            task.pre_exec   += ['export PATH=$MINICONDA/bin:$PATH']
            task.pre_exec   += ['export LD_LIBRARY_PATH=$MINICONDA/lib:$LD_LIBRARY_PATH']
            task.pre_exec   += ['module load python/2.7.15-anaconda2-5.3.0']
            task.pre_exec   += ['module load cuda/9.1.85']
            task.pre_exec   += ['module load gcc/6.4.0']
            task.pre_exec   += ['source activate openmm']
            task.pre_exec   += ['which python']
            # task.pre_exec   += ['cd /gpfs/alpine/scratch/jdakka/bip178/benchmarks/MD_exps/fs-pep']
            task.executable  = 'python'
            task.arguments   = ['hello_world.py']
            # task.arguments = ['run_openmm.py', '-f', '/gpfs/alpine/scratch/jdakka/bip178/benchmarks/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb']
            task.cpu_reqs = {'processes': 1,
                             'process_type': None,
                             'threads_per_process': 1,
                             'thread_type': None
                             }

            task.gpu_reqs = {'processes': 1,
                             'process_type': None,
                             'threads_per_process': 1,
                             'thread_type': None
                             }

            # Add the MD task to the Docking Stage
            s1.add_tasks(task)

        # Add MD stage to the MD Pipeline
        p.add_stages(s1)


        return p

    
    p = describe_MD_pipline()
    return p



if __name__ == '__main__':

    
    res_dict = {

            'resource': 'ornl.summit',
            'project' : 'BIP178',
            'queue' : 'batch',
            'walltime': 60,
            'cpus': 126,
            # 'gpus': 18,
            'access_schema': 'local'
    }

    # Create Application Manager
    appman = AppManager(hostname=hostname, port=port)
    appman.resource_desc = res_dict

    p1 = generate_MD_pipeline()

    pipelines = []
    pipelines.append(p1)

    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
