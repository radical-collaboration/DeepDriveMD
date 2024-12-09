#!/usr/bin/env python3

# - initial ML force field exists
# - while iteration < X (configurable):
#   - start DTF ( Ab-initio MD simulation ) (with all reasources) CPU only
#   - start force field training task (FFTrain) (with all resources) CPU only
#   - if DFT partially satisfy the uncertainty
#     - Kill Half of the Ab-initio Tasks
#     - Start DDMD with %50 CPU and %100 GPU
#   - If DFT fully satisfy:
#       - run 2nd DDMD loop (divide available resources between bot loop)
#   - If DDMD1 finish run DDMD 2 with full resoureces

# lower / upper bound on active num of simulations
# ddmd.get_last_n_sims ...

# ------------------------------------------------------------------------------
#

# This one will run synchronously
import os
import math, sys, argparse
import json
import time
import random
import signal
import threading as mt

from collections import defaultdict

import radical.pilot as rp
import radical.utils as ru

import itertools
import shutil
from pathlib import Path
from typing import List, Optional


from deepdrivemd.config import BaseStageConfig, ExperimentConfig
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.utils import parse_args


# ------------------------------------------------------------------------------
# This is the main class
# TODO: Maybe we need a base class and multiple classes for DDMD and AB-INITIO
class DDMD(object):

    # define task types (used as prefix on task-uid)
    # AB-INITIO TASKS
    TASK_TRAIN_FF   = 'task_train_ff'   # AB-initio-FF-training
    TASK_MD         = 'task_md'         # AB-initio MD-simulation
    TASK_DFT1       = 'task_dft1'       # Ab-inito DFT prep
    TASK_DFT2       = 'task_dft'        # Ab-inito DFT calculation
    TASK_DFT3       = 'task_dft'        # Ab-inito DFT finalize
    # DDMD TASKS
    TASK_DDMD_MD        = 'task_ddmd_md'        # DDMD MD-Simulation
    TASK_DDMD_TRAIN     = 'task_ddmd_train'     # DDMD Training
    TASK_DDMD_SELECTION = 'task_ddmd_selection' # DDMD Selection
    TASK_DDMD_AGENT     = 'task_ddmd_agent'     # DDMD Agent

    TASK_TYPES       = [TASK_TRAIN_FF,
                        TASK_DDMD,
                        TASK_MD,
                        TASK_DFT1,
                        TASK_DFT2,
                        TASK_DFT3,
                        TASK_DDMD_MD,
                        TASK_DDMD_TRAIN,
                        TASK_DDMD_SELECTION,
                        TASK_DDMD_AGENT]

    # these alues fall from heaven....
    # We need to have a swich condition here.
    ITER_AB_INITIO = 6
    ITER_DDMD   = 6
    ITER_DDMD_1 = int(math.floor(ITER_AB_INITIO / 2))
    ITER_DDMD_2 = ITER_AB_INITIO

    # keep track of core usage
    cores_used     = 0
    gpus_used      = 0
    avail_cores    = 0
    avail_gpus     = 0

    # keep track the stage
    stage = 0 # 0 no tasks started
              # 1 only ab-initio
              # 2 ab-initio + DDM1
              # 3 DDMD1 + DDMD2
              # 4 only DDMD2
              # 5 all done

    # --------------------------------------------------------------------------
    #
    def __init__(self):

        # control flow table
        self._protocol = {self.TASK_TRAIN_FF        : self._control_train_ff        ,
                          self.TASK_MD              : self._control_md              ,
                          self.TASK_DFT1            : self._control_dft1            ,
                          self.TASK_DFT2            : self._control_dft2            ,
                          self.TASK_DFT3            : self._control_dft3            ,
                          self.TASK_DDMD_MD         : self._control_ddmd_md         ,
                          self.TASK_DDMD_TRAIN      : self._control_ddmd_train      ,
                          self.TASK_DDMD_SELECTION  : self._control_ddmd_selection  ,
                          self.TASK_DDMD_AGENT      : self._control_ddmd_agent      }

        self._glyphs   = {self.TASK_TRAIN_FF        : 't',
                          self.TASK_MD              : 'm',
                          self.TASK_DFT1            : 'i',
                          self.TASK_DFT2            : 'd',
                          self.TASK_DFT3            : 'e',
                          self.TASK_DDMD_MD         : 'M',
                          self.TASK_DDMD_TRAIN      : 'T',
                          self.TASK_DDMD_SELECTION  : 'S',
                          self.TASK_DDMD_AGENT      : 'A',

        # bookkeeping
        # FIXME There are lots off un used item here
        self._iter           =  0
        self._iterDDMD1      =  0
        self._iterDDMD2      =  0
        self._threshold      =  1
        self._cores          = 16  # available cpu resources FIXME: maybe get from the user?
        self._gpus           =  4  # available gpu resources  ""
        self._avail_cores    = self._cores
        self._avail_gpus     = self._gpus
        self._cores_used     =  0
        self._gpus_used      =  0
        self._ddmd_tasks     =  0

        # FIXME Make sure everything is needed.
        self._lock   = mt.RLock()
        self._series = [1, 2]
        self._uids   = {s:list() for s in self._series}

        self._tasks  = {s: {ttype: dict() for ttype in self.TASK_TYPES}
                            for s in self._series}

        self._final_tasks = list()

        # silence RP reporter, use own
        os.environ['RADICAL_REPORT'] = 'false'
        self._rep = ru.Reporter('nwchem')
        self._rep.title('NWCHEM')

        # RP setup
        self._session = rp.Session()
        self._pmgr    = rp.PilotManager(session=self._session)
        self._tmgr    = rp.TaskManager(session=self._session)

        # Maybe get from user??
        pdesc = rp.PilotDescription({'resource': 'local.localhost',
                                     'runtime' : 30,
#                                     'runtime' : 4,
                                     'cores'   : self._cores})
#                                     'cores'   : 1})
        self._pilot = self._pmgr.submit_pilots(pdesc)

        self._tmgr.add_pilots(self._pilot)
        self._tmgr.register_callback(self._state_cb)

        # Parser
        self.set_argparse()
        self.get_json()

        #set aditional DDMD related setups:

        #FIXME: Makesure the names are not conflicting with others
        args = parse_args()
        cfg = ExperimentConfig.from_yaml(args.config)

        # Calculate total number of nodes required.
        # If gpus_per_node is 0, then we assume that the CPU is used for
        # simulation, in which case we request a node per simulation task.
        # Otherwise, we assume that each simulation task uses a single GPU.
        if cfg.gpus_per_node == 0:
            num_nodes = cfg.molecular_dynamics_stage.num_tasks
        else:
            num_nodes, extra_gpus = divmod(
                cfg.molecular_dynamics_stage.num_tasks, cfg.gpus_per_node
            )
            # If simulations don't pack evenly onto nodes, add an extra node
            num_nodes += int(extra_gpus > 0)

        num_nodes = max(1, num_nodes)

        #FIXME maybe we can use this but we need to be carefull here.
        ddmd_pilot_desc = rp.PilotDescription.({
            "resource": cfg.resource,
            "queue": cfg.queue,
            "access_schema": cfg.schema_,
            "walltime": cfg.walltime_min,
            "project": cfg.project,
            "cpus": cfg.cpus_per_node * cfg.hardware_threads_per_cpu * num_nodes,
            "gpus": cfg.gpus_per_node * num_nodes})

        api = DeepDriveMD_API(cfg.experiment_directory)

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # ---------FUNCINALITIES FROM DDME------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # this needs to converted to the RP task:
    #TODO Andre.
    def generate_task_description(cfg: BaseStageConfig) -> rp.TaskDescription:
        td = rp.TaskDescription()
        td.ranks          = cfg.cpu_reqs.cpu_processes
        td.cores_per_rank = cfg.cpu_reqs.cpu_threads
        td.gpus_per_rank  = cfg.gpu_reqs.gpu_processes
        td.pre_exec       = copy.deepcopy(cfg.pre_exec)
        td.executable     = copy.deepcopy(cfg.executable)
        td.arguments      = copy.deepcopy(cfg.arguments)
        return td


#we don't need this
    def _init_experiment_dir(self) -> None:
        # Make experiment directories
        self.cfg.experiment_directory.mkdir()
        self.api.molecular_dynamics_stage.runs_dir.mkdir()
        self.api.aggregation_stage.runs_dir.mkdir()
        self.api.machine_learning_stage.runs_dir.mkdir()
        self.api.model_selection_stage.runs_dir.mkdir()
        self.api.agent_stage.runs_dir.mkdir()

#FIXME Probably neeed to delete this one but I am not sure since it is checking max iteration
    def func_condition(self) -> None:
        if self.stage_idx < self.cfg.max_iteration:
            self.func_on_true()
        else:
            self.func_on_false()

#FIXME we definitly dont need following
#    def func_on_true(self) -> None:
#        print(f"Finishing stage {self.stage_idx} of {self.cfg.max_iteration}")
#        self._generate_pipeline_iteration()
#
#    def func_on_false(self) -> None:
#        print("Done")
#
#    def _generate_pipeline_iteration(self) -> None:
#
#        self.pipeline.add_stages(self.generate_molecular_dynamics_stage())
#
#        if not cfg.aggregation_stage.skip_aggregation:
#            self.pipeline.add_stages(self.generate_aggregating_stage())
#
#        if self.stage_idx % cfg.machine_learning_stage.retrain_freq == 0:
#            self.pipeline.add_stages(self.generate_machine_learning_stage())
#        self.pipeline.add_stages(self.generate_model_selection_stage())
#
#        agent_stage = self.generate_agent_stage()
#        agent_stage.post_exec = self.func_condition
#        self.pipeline.add_stages(agent_stage)
#
#        self.stage_idx += 1
#
#    def generate_pipelines(self) -> List[Pipeline]:
#        self._generate_pipeline_iteration()
#        return [self.pipeline]






    #TODO Andre.
#    def generate_molecular_dynamics_stage(self) -> Stage:
    def generate_molecular_dynamics_stage(self):
#        stage = Stage()
#        stage.name = self.MOLECULAR_DYNAMICS_STAGE_NAME
        # I created a List instead of the Stage
        tds =[]
        cfg = self.cfg.molecular_dynamics_stage
        stage_api = self.api.molecular_dynamics_stage

        if self.stage_idx == 0:
            initial_pdbs = self.api.get_initial_pdbs(cfg.task_config.initial_pdb_dir)
            filenames: Optional[itertools.cycle[Path]] = itertools.cycle(initial_pdbs)
        else:
            filenames = None

        for task_idx in range(cfg.num_tasks):

            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path
            if self.stage_idx == 0:
                assert filenames is not None
                cfg.task_config.pdb_file = next(filenames)
            else:
                cfg.task_config.pdb_file = None

            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            assert cfg_path is not None
            cfg.task_config.dump_yaml(cfg_path)
            td = generate_task_description(cfg)
            td.arguments += ["-c", cfg_path.as_posix()]
            #FIXME ANDRE can you check if this makes sense?
            #TODO ANDRE also do you think if there is a issue submitting tasks back to back here
            #           versus using n=cfg.num_tasks?
            #           I submit a single task here in a loop since
            #           it is setting taskid and probably output_path for each task
            td.uid = self.TASK_DDMD_MD
            self._submit_task(td, series = 1)
            tds.append(td)

#        return stage
        return tds

    #TODO Andre.
    #TODO HUUB:  DO we have aggregation  stage?
#    def generate_aggregating_stage(self) -> Stage:
    def generate_aggregating_stage(self):
#        stage = Stage()
#        stage.name = self.AGGREGATION_STAGE_NAME

        cfg = self.cfg.aggregation_stage
        stage_api = self.api.aggregation_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        assert cfg_path is not None
        cfg.task_config.dump_yaml(cfg_path)
        td = generate_task_description(cfg)
        td.arguments += ["-c", cfg_path.as_posix()]
        #FIXME ANDRE can you check if this makes sense?
        td.uid = self.TASK_DDMD_SELECTION
        self._submit_task(td, series = 1)

        return [td]

    #TODO Andre.
#    def generate_machine_learning_stage(self) -> Stage:
    def generate_machine_learning_stage(self):
#        stage = Stage()
#        stage.name = self.MACHINE_LEARNING_STAGE_NAME
        cfg = self.cfg.machine_learning_stage
        stage_api = self.api.machine_learning_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path
        cfg.task_config.model_tag = stage_api.unique_name(output_path)
        if self.stage_idx > 0:
            # Machine learning should use model selection API
            cfg.task_config.init_weights_path = None

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        assert cfg_path is not None
        cfg.task_config.dump_yaml(cfg_path)
        td = generate_task_description(cfg)
        td.arguments += ["-c", cfg_path.as_posiix()]
        #FIXME ANDRE can you check if this makes sense?
        td.uid = self.TASK_DDMD_TRAIN
        self._submit_task(td, series = 1)
        return [td]

    #TODO Andre.
#    def generate_model_selection_stage(self) -> Stage:
    def generate_model_selection_stage(self):
#        stage = Stage()
#        stage.name = self.MODEL_SELECTION_STAGE_NAME
        cfg = self.cfg.model_selection_stage
        stage_api = self.api.model_selection_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        assert cfg_path is not None
        cfg.task_config.dump_yaml(cfg_path)
        td = generate_task_description(cfg)
        td.arguments += ["-c", cfg_path.as_posix()]
        #FIXME ANDRE can you check if this makes sense?
        td.uid = self.TASK_DDMD_SELECTION
        self._submit_task(td, series = 1)

        return [td]

    #TODO Andre.
#    def generate_agent_stage(self) -> Stage:
    def generate_agent_stage(self):
#        stage = Stage()
#        stage.name = self.AGENT_STAGE_NAME
        cfg = self.cfg.agent_stage
        stage_api = self.api.agent_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        assert cfg_path is not None
        cfg.task_config.dump_yaml(cfg_path)
        td = generate_task_description(cfg)
        td.arguments += ["-c", cfg_path.as_posix()]
        #FIXME ANDRE can you check if this makes sense?
        td.uid = self.TASK_DDMD_AGENT
        self._submit_task(td, series = 1)
        return [td]



    # --------------------------------------------------------------------------
        def set_argparse(self):
        parser = argparse.ArgumentParser(description="NWChem - DeepDriveMD Synchronous")
        #FIXME Delete unneded ones and add the ones we need.
        parser.add_argument('--num_phases', type=int, default=3,
                        help='number of phases in the workflow')
        parser.add_argument('--mat_size', type=int, default=5000,
                        help='the matrix with have size of mat_size * mat_size')
        parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
        parser.add_argument('--num_step', type=int, default=1000,
                        help='number of step in MD simulation')
        parser.add_argument('--num_epochs_train', type=int, default=150,
                        help='number of epochs in training task')
        parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
        parser.add_argument('--conda_env', default=None,
                        help='the conda env where numpy/cupy installed, if not specified, no env will be loaded')
        parser.add_argument('--num_sample', type=int, default=500,
                        help='num of samples in matrix mult (training and agent)')
        parser.add_argument('--num_mult_train', type=int, default=4000,
                        help='number of matrix mult to perform in training task')
        parser.add_argument('--dense_dim_in', type=int, default=12544,
                        help='dim for most heavy dense layer, input')
        parser.add_argument('--dense_dim_out', type=int, default=128,
                        help='dim for most heavy dense layer, output')
        parser.add_argument('--preprocess_time_train', type=float, default=20.0,
                        help='time for doing preprocess in training')
        parser.add_argument('--preprocess_time_agent', type=float, default=10.0,
                        help='time for doing preprocess in agent')
        parser.add_argument('--num_epochs_agent', type=int, default=10,
                        help='number of epochs in agent task')
        parser.add_argument('--num_mult_agent', type=int, default=4000,
                        help='number of matrix mult to perform in agent task, inference')
        parser.add_argument('--num_mult_outlier', type=int, default=10,
                        help='number of matrix mult to perform in agent task, outlier')
        parser.add_argument('--enable_darshan', action='store_true',
                        help='enable darshan analyze')
        parser.add_argument('--project_id', required=True,
                        help='the project ID we used to launch the job')
        parser.add_argument('--queue', required=True,
                        help='the queue we used to submit the job')
        parser.add_argument('--work_dir', default=self.env_work_dir,
                        help='working dir, which is the dir of this repo')
        parser.add_argument('--num_sim', type=int, default=12,
                        help='number of tasks used for simulation')
        parser.add_argument('--num_nodes', type=int, default=3,
                        help='number of nodes used for simulation')
        parser.add_argument('--io_json_file', default="io_size.json",
                        help='the filename of json file for io size')

        args = parser.parse_args()
        self.args = args

    # FIXME: This is unused now but we may want to use  a json file in the future
    def get_json(self):
        json_file = "{}/launch-scripts/{}".format(self.args.work_dir, self.args.io_json_file)
        with open(json_file) as f:
            self.io_dict = json.load(f)

    #  FIXME do not use argument_val and get them from the user using arguments
    def get_arguments(self, ttype, argument_val=""):
        args = []

        if ttype == self.TASK_MD:
            args = ['{}/sim/lammps/main_ase_lammps.py'.format(self.args.work_dir),
                           '{}'.format(argument_val.split("|")[0]), # get pbd file path here #FIXME
                           '{}'.format(argument_val.split("|")[1])] # get test dir  path here #FIXME

        elif ttype == self.TASK_DFT1:
            args = ['{}/sim/nwchem/main1_nwchem.py'.format(self.args.work_dir)]
        elif ttype == self.TASK_DFT2:
            args = ['{}/sim/nwchem/main2_nwchem.py'.format(self.args.work_dir),
                           '{}'.format(argument_val))] # this will need to get the instance
        elif ttype == self.TASK_DFT3:
            args = ['{}/sim/nwchem/main3_nwchem.py'.format(self.args.work_dir)]

        elif ttype == self.TASK_TRAIN_FF:
            args = ['{}/model/deepm/main_deepmd.py'.format(self.args.work_dir),
                       '{}'.format(argument_val)] #training folder name
`

        elif ttype == self.TASK_DDMD: #TODO: ask to to HUUB
            args = ['{}/Executables/training.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs_train),
                       '--device=gpu',
                       '--phase={}'.format(phase_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--num_sample={}'.format(self.args.num_sample * (1 if phase_idx == 0 else 2)),
                       '--num_mult={}'.format(self.args.num_mult_train),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_train),
                       '--write_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["train"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(phase_idx)]["train"]["read"])]


        return args



    # --------------------------------------------------------------------------
    #
    def __del__(self):

        self.close()


    # --------------------------------------------------------------------------
    #
    def close(self):

        if self._session is not None:
            self._session.close()
            self._session = None


    # --------------------------------------------------------------------------
    #
    def dump(self, task=None, msg=''):
        '''
        dump a representation of current task set to stdout
        '''

        # this assumes one core per task

        self._rep.plain('<<|')

        idle = self._cores

        for ttype in self.TASK_TYPES:

            n = 0
            for series in self._series:
                n   += len(self._tasks[series][ttype])
                idle -= n

            self._rep.ok('%s' % self._glyphs[ttype] * n)

        self._rep.plain('%s' % '-' * idle +
                        '| %4d [%4d]' % (self._cores_used, self._cores))

        if task and msg:
            self._rep.plain(' %-15s: %s\n' % (task.uid, msg))
        else:
            if task:
                msg = task
            self._rep.plain(' %-15s: %s\n' % (' ', msg))


    # --------------------------------------------------------------------------
    #
    def start(self):
        '''
        submit initial set of Ab-initio MD similation tasks DFT
        '''

        self.dump('submit MD simulations')

        # start ab-initio loop
        self.stage = 1
        self._submit_task(self.TASK_DFT, ...)




    # --------------------------------------------------------------------------
    #
    def stop(self):

        os.kill(os.getpid(), signal.SIGKILL)
        os.kill(os.getpid(), signal.SIGTERM)


    # --------------------------------------------------------------------------
    #
    def _get_ttype(self, uid):
        '''
        get task type from task uid
        '''

        ttype = uid.split('.')[0]

        assert ttype in self.TASK_TYPES, 'unknown task type: %s' % uid
        return ttype


    # --------------------------------------------------------------------------
    #
    # FIXME: we need to consider this again  with new model.
    #TODO Andre.
    def _submit_task(self, ttype, args=None, n=1, cpu=1, gpu=0, series: int=1, argvals=''):
        '''
        submit 'n' new tasks of specified type

        NOTE: all tasks are uniform for now: they use a single core and sleep
              for a random number (0..3) of seconds.
        '''

        # NOTE: ttype can be a task description or a string.  In the first case,
        #       we submit `n` tasks with that description.  In the second case,
        #       we construct the task description from the remaining arguments
        #       and the ttype string

        if isinstance(ttype, rp.TaskDescription):
            tds = [ttype] * n
            for td in tds:
                td.uid = ru.generate_id(ttype)

        elif isinstance(ttype, str):

            cur_args = self.get_arguments(ttype, argument_val=argvals)
            tds = list()
            for _ in range(n):

                # FIXME: uuid=ttype won't work - the uid needs to be *unique*

                tds.append(rp.TaskDescription({
                           'pre_exec'   : ['. %s/bin/activate' % ve_path,
                                           'pip install pyyaml'], #FIXME: give correct environment name
                           'uid'            : ru.generate_id(ttype),
                           'ranks'          : 1
                           'cores_per_rank' : cpu,
                           'gpus_per_rank'  : gpu,
                           'executable'     : 'python',
                           'arguments'      : cur_args
                           }))

        with self._lock:

            tasks = self._tmgr.submit_tasks(tds)

            for task in tasks:
                self._register_task(task, series=series)


    # --------------------------------------------------------------------------
    #
    def _cancel_tasks(self, uids):
        '''
        cancel tasks with the given uids, and unregister them
        '''

        uids = ru.as_list(uids)

        # FIXME AM: does not work
        self._tmgr.cancel_tasks(uids)

        for uid in uids:

            series = self._get_series(uid=uid)
            ttype = self._get_ttype(uid)
            task  = self._tasks[series][ttype][uid]
            self.dump(task, 'cancel [%s]' % task.state)

            self._unregister_task(task)

        self.dump('cancelled')


    # --------------------------------------------------------------------------
    #
    def _register_task(self, task, series: int):
        '''
        add task to bookkeeping
        '''

        with self._lock:

            ttype = self._get_ttype(task.uid)

            self._uids[series].append(task.uid)

            self._tasks[series][ttype][task.uid] = task

            cores = task.description['cpu_processes'] \
                  * task.description['cpu_threads']
            self._cores_used += cores

            gpus = task.description['gpu_processes']
            self._gpus_used += gpus


    # --------------------------------------------------------------------------
    #
    def _unregister_task(self, task):
        '''
        remove completed task from bookkeeping
        '''

        with self._lock:

            series = self._get_series(task)
            ttype = self._get_ttype(task.uid)

            if task.uid not in self._tasks[series][ttype]:
                return

            # remove task from bookkeeping
            self._final_tasks.append(task.uid)
            del self._tasks[series][ttype][task.uid]
            self.dump(task, 'unregister %s' % task.uid)

            cores = task.description['cpu_processes'] \
                  * task.description['cpu_threads']
            self._cores_used -= cores

            gpus = task.description['gpu_processes']
            self._gpus_used -= gpus


    # --------------------------------------------------------------------------
    #
    def _state_cb(self, task, state):
        '''
        act on task state changes according to our protocol
        '''

        try:
            return self._checked_state_cb(task, state)

        except Exception as e:
            self._rep.exception('\n\n---------\nexception caught: %s\n\n' % repr(e))
            ru.print_exception_trace()
            self.stop()


    # --------------------------------------------------------------------------
    #
    def _checked_state_cb(self, task, state):

        # this cb will react on task state changes.  Specifically it will watch
        # out for task completion notification and react on them, depending on
        # the task type.

        if state in [rp.TMGR_SCHEDULING] + rp.FINAL:
            self.dump(task, ' -> %s' % task.state)

        # ignore all non-final state transitions
        if state not in rp.FINAL:
            return

        # ignore tasks which were already completed
        if task.uid in self._final_tasks:
            return

        # lock bookkeeping
        with self._lock:

            # raise alarm on failing tasks (but continue anyway)
            if state == rp.FAILED:
                self._rep.error('task %s failed: %s' % (task.uid, task.stderr))
                self.stop()

            # control flow depends on ttype
            ttype  = self._get_ttype(task.uid)
            action = self._protocol[ttype]
            if not action:
                self._rep.exit('no action found for task %s' % task.uid)
            action(task)

            # remove final task from bookkeeping
            self._unregister_task(task)


    # --------------------------------------------------------------------------
    #
    def _get_series(self, task=None, uid=None):

        if uid:
            # look up by uid
            for series in self._series:
                if uid in self._uids[series]:
                    return series

        else:
            # look up by task type
            for series in self._series:
                if task.uid in self._uids[series]:
                    return series

        raise ValueError('task does not belong to any serious')


    # --------------------------------------------------------------------------
    #
    def _control_md(self, task):
        '''
        react on completed ff training task
        '''

        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_MD]) > 1:
            return


        self.dump(task, 'completed ab-initio md ')

        #check if this satisfy:
        if False:
            #FIXME: Here we need to write resource allocation to the YAML file.
            # maybe for now we can skip this
            with open (self.args.yaml, 'a') as f:
                self.printYAML(cpus=cpus, gpus=gpus, sim=sim) #FIXME

            # FIXME: ttype is not defined here
            self._submit_task(self, ttype, args=None, n=1, cpu=1, gpu=0, series: int=1, argvals='') #FIXME
        else:
            self._submit_task(self, self.TASK_DFT1, args=None, n=1, cpu=1, gpu=0, series: int=1, argvals='')




    # --------------------------------------------------------------------------
    #
    def _control_train_ff(self, task):
        '''
        react on completed ff training task
        '''

        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_TRAIN_FF]) > 1:
            return

        self.dump(task, 'completed ff train')
        self._submit_task(self, self.TASK_MD, args=None, n=1, cpu=1, gpu=0, series: int=1, argvals='')

    # --------------------------------------------------------------------------
    #
    def _control_dft1(self, task):
        '''
        react on completed DFT task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DFT1]) > 1:
            return

        # TODO READ the inputs.txt
        # submit self.TASK_DFT2 for each line

        self.dump(task, 'completed dft1')
        self._submit_task(self, self.TASK_DFT2, args=None, n=1, cpu=1, gpu=0, series: int=1, argvals='')

    # --------------------------------------------------------------------------
    #
    def _control_dft2(self, task):
        '''
        react on completed DFT task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DFT2]) > 1:
            return

        self.dump(task, 'completed dft')
        self._submit_task(self, self.TASK_DFT3, args=None, n=1, cpu=1, gpu=0, series: int=1, argvals='')


    # --------------------------------------------------------------------------
    #
    def _control_dft3(self, task):
        '''
        react on completed DFT task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DFT3]) > 1:
            return

        self.dump(task, 'completed dft')
        self._submit_task(self, self.TASK_TRAIN_FF, args=None, n=1, cpu=1, gpu=0, series: int=1, argvals='')

    # --------------------------------------------------------------------------#
    #           CONTROLS FOR DDMD LOOP                                          #
    # --------------------------------------------------------------------------#
    def _control_ddmd_md(self, task):
        '''
        react on completed DDMD selection task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DDMD_MD]) > 1:
            return

        self.dump(task, 'completed DDMD MD')

        self.generate_machine_learning_stage()

    # --------------------------------------------------------------------------
    #
    def _control_ddmd_train(self, task):
        '''
        react on completed DDMD selection task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DDMD_TRAIN]) > 1:
            return

        self.dump(task, 'completed DDMD Training')

        self.generate_model_selection_stage()
    # --------------------------------------------------------------------------
    #
    def _control_ddmd_selection(self, task):
        '''
        react on completed DDMD selection task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DDMD_SELECTION]) > 1:
            return

        self.dump(task, 'completed DDMD Selection')

        self.generate_agent_stage()

    # --------------------------------------------------------------------------
    #
    def _control_ddmd_agent(self, task):
        '''
        react on completed DDMD selection task
        '''
        series = self._get_series(task)

        if len(self._tasks[series][self.TASK_DDMD_AGENT]) > 1:
            return

        self.dump(task, 'completed DDMD agent')

        #Check if  we are done with DDMD loop:
        if self.stage_idx < self.cfg.max_iteration:
            self.stage_idx += 1
            self.generate_molecular_dynamics_stage()
        else:
            self.dump("DONE!!!")
            ddmd.close() #TODO Check if this is needed!!!

# ------------------------------------------------------------------------------
#
if __name__ == '__main__':
    ddmd = DDMD()
    try:
        ddmd.start()
        while True:
          # ddmd.dump()
            time.sleep(1)

    finally:
        ddmd.close()


# ------------------------------------------------------------------------------
