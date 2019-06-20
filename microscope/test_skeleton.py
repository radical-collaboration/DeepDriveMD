import os
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


def generate_training_pipeline():

    p = Pipeline()
    p.name = 'training'

    # --------------------------
    # MD stage
    s1 = Stage()
    s1.name = 'simulating'

    # MD tasks
    for i in range(2):
        t1 = Task()
        # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/MD_exps/fs-pep/run_openmm.py
        t1.executable = ['sleep']  # run_openmm.py
        t1.arguments = ['60']

        # Add the MD task to the Docking Stage
        s1.add_tasks(t1)

    # Add Docking stage to the pipeline
    p.add_stages(s1)

    # --------------------------
    # Aggregate stage
    s2 = Stage()
    s2.name = 'aggregating'

    # Aggregation task
    t2 = Task()
    # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/MD_to_CVAE/MD_to_CVAE.py
    t2.executable = ['sleep']  # MD_to_CVAE.py

    t2.arguments = ['30']

    # Add MD stage to the MD Pipeline
    p.add_stages(s2)

    # --------------------------
    # Learning stage
    s3 = Stage()
    s3.name = 'learning'

    # Aggregation task
    t3 = Task()
    # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/CVAE_exps/train_cvae.py
    t3.executable = ['sleep']  # train_cvae.py
    t3.arguments = ['30']

    # Add MD stage to the MD Pipeline
    p.add_stages(s3)

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

    # Add Docking stage to the pipeline
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

    # Add MD stage to the MD Pipeline
    p.add_stages(s2)

    # --------------------------
    # Learning stage
    s3 = Stage()
    s3.name = 'inferring'

    # Aggregation task
    t3 = Task()
    t3.executable = ['sleep']  # CVAE executable
    t3.arguments = ['30']

    # Add MD stage to the MD Pipeline
    p.add_stages(s3)

    return p


if __name__ == '__main__':

    # Create a dictionary to describe four mandatory keys:
    # resource, walltime, cores and project
    # resource is 'local.localhost' to execute locally
    res_dict = {

            'resource': 'local.summit',
            'queue'   : 'batch',
            'schema'  : 'local',
            'walltime': 15,
            'cpus'    : 48,
            'gpus'    : 4
    }

    # Create Application Manager
    appman = AppManager()
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
