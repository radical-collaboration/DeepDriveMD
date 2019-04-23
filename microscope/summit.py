import os
from radical.entk import Pipeline, Stage, Task, AppManager

# ------------------------------------------------------------------------------
# Set default verbosity
hostname = os.environ.get('RMQ_HOSTNAME', 'csc190specfem.marble.ccs.ornl.gov')
port = int(os.environ.get('RMQ_PORT', 30672))

if os.environ.get('RADICAL_ENTK_VERBOSE') is None:
    os.environ['RADICAL_ENTK_REPORT'] = 'True'


def generate_MD_pipeline():

    def describe_MD_pipeline():
        p = Pipeline()
        p.name = 'MD'

        # MD stage
        s1 = Stage()
        s1.name = 'OpenMM'

        t1 = Task()
        t1.name = 'md' 
        
        t1.pre_exec    = []

        t1.pre_exec   += ['module load python/2.7.15-anaconda2-5.3.0']
        t1.pre_exec   += ['module load cuda/9.1.85']
        t1.pre_exec   += ['module load gcc/6.4.0']
        t1.pre_exec   += ['source activate openmm']
        t1.pre_exec   += ['cd /gpfs/alpine/scratch/jdakka/bip178/benchmarks/MD_exps/fs-pep/']
        t1.executable  = '/ccs/home/jdakka/.conda/envs/openmm/bin/python'
        t1.arguments = ['run_openmm.py', '-f', 
        '/gpfs/alpine/scratch/jdakka/bip178/benchmarks/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb']
        t1.cpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 1,
                         'thread_type': None
                         }

        t1.gpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 1,
                         'thread_type': 'CUDA'
                        }

        # Add the MD task to the Docking Stage
        s1.add_tasks(t1)

        # Add MD stage to the MD Pipeline
        p.add_stages(s1)


        # TF stage
        s2 = Stage()
        s2.name = 'tf'

        # Task 2 is TF benchmark

        t2 = Task()
        t2.name = 'tf' 
        
        t2.pre_exec    = []


        t2.pre_exec   += ['module load python/2.7.15-anaconda2-5.3.0']
        t2.pre_exec   += ['module load cuda/9.1.85']
        t2.pre_exec   += ['module load gcc/6.4.0']
        t2.pre_exec   += ['source activate openmm']
        t2.pre_exec   += ['cd /gpfs/alpine/scratch/jdakka/bip178']
        t2.executable  = '/ccs/home/jdakka/.conda/envs/openmm/bin/python'
        t2.arguments = ['tf.py']
        t2.cpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 1,
                         'thread_type': None
                         }

        t2.gpu_reqs = {'processes': 1,
                         'process_type': None,
                         'threads_per_process': 1,
                         'thread_type': 'CUDA'
                        }

        # Add the MD task to the Docking Stage
        s2.add_tasks(t2)

        # Add MD stage to the MD Pipeline
        p.add_stages(s2)
        return p

    
    p = describe_MD_pipeline()
    return p



if __name__ == '__main__':

    
    res_dict = {

            'resource': 'ornl.summit',
            'project' : 'BIP178',
            'queue' : 'batch',
            'walltime': 120,
            'cpus': 42,
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
