# DeepDriveMD v1

## Run

```
$ python summit_md.py
```

## Prerequisites

Configurations in the script (`summit_md.py`)
- `conda_path`: a location of the conda environment for openMM e.g. '$HOME/.conda/envs/openmm'
- `MAX_STAGE`: a number of iterations to stop
- `RETRAIN_FREQ`: an interval to trigger CVAE stage
- `LEN_initial`: a timesteps of the first MD run (nanosecond)
- `LEN_iter`: a timespan of the rest simulation trajectory (nanosecond)
- `generate_MD_stage(num_MD=N)`: a method with a N number of simulations
- `generate_ML_stage(num_ML=M)`: a method with a M number of training

Resource configurations in the script (`summit_md.py`)
- `res_dict` would have:
   - 'resource': HPC resource e.g. 'ornl.summit'
   - 'queue':
      - 'batch' for upto 2 hours TTX at `ornl.summit`
      - 'killable' for upto 1 day(24hours) TTX at `ornl.summit`
   - 'gpus': a number of GPU cores equal to N simulations (1 MD/ML run: 1GPU)

Bash environment variable for EnTK
- RMQ_HOSTNAME/RMQ_PORT (mandatory)
- export RADICAL_LOG_TGT=radical.log; export RADICAL_LOG_LVL=DEBUG (optional for extra log messages)
- export RADICAL_PILOT_PROFILE=TRUE; export RADICAL_ENTK_PROFILE=TRUE (optional for profiling)


## Clean up

Intermediate files/folders are generated over the workflow execution. These need to be deleted before the next run.
Currently, manual deletion is required like:
```
rm -rf CVAE_exps/cvae_runs_*
rm -rf VHP_exp/omm_runs_*
rm -rf fs-pep/omm_runs_*
rm -rf Outlier_search/outlier_pdbs/*
```


## Diagram
![alt text](https://github.com/radical-collaboration/DeepDriveMD/raw/master/papers/MicroScope_Workflow_Diagram.png "Logo Title Text 1")
