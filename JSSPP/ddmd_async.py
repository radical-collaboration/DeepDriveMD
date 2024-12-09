import numpy as np
from typing import List
from radical.entk import Pipeline, Stage, Task

SUMMIT_CORES = 42
SUMMIT_GPU = 6
SF_SIM_TX = 1/40
SF_TX = 1/20
SF_TX_AGENT = 1/6
SF_NTASKS = 1/10
TX_SIM = 1360*SF_SIM_TX    # [s]
TX_PREPROC = 340*SF_TX    # [s]
TX_ML = 250*SF_TX     # [s]
TX_AGENT = 150*SF_TX_AGENT  # [s]

NUM_SIM_TASKS = 960*SF_NTASKS  # GPU tasks
NUM_PREPROC_TASKS = 420*SF_NTASKS  # CPU tasks
NUM_ML_TASKS = 1 # GPU task
NUM_AGENT_TASKS_CPU = 6720 * SF_NTASKS # CPU task
NUM_AGENT_TASKS_GPU = 960 * SF_NTASKS # GPU task

def generate_task(cfg, name, ttx) -> Task:
    task = Task()
    task.name = name
    task.executable = cfg.executable
    task.arguments = ['%s' % ttx]
    task.pre_exec = cfg.pre_exec.copy()
    task.cpu_reqs = cfg.cpu_reqs.dict().copy()
    task.gpu_reqs = cfg.gpu_reqs.dict().copy()
    return task


def generate_ttx(nsamples, mu=20, stddev=10):
    normal = np.random.normal(mu, stddev, nsamples)
    return normal

# We will want to have a distribution of run-times

class AsyncPipelineManager:

    def __init__(self, cfg):
        self.cfg = cfg


    def generate_sim_pipeline(self) -> List[Pipeline]:
        pipeline = Pipeline()
        pipeline.add_stages(self.generate_sim_stage())
        return pipeline


    def generate_sim_stage(self) -> List[Pipeline]:
        cfg = self.cfg.molecular_dynamics_stage
        stage = Stage()
        stage.name = "Simulation"

        # Generate normally-distributed pseudo-randoms for this
        # pipeline
        normal_rands = generate_ttx(cfg.num_tasks,
                                    TX_SIM, 0.25)

        # Number of simulation tasks per pipeline
        for t in range(0, cfg.num_tasks):
            tname = "Sim-" + t
            task = generate_task(cfg, tname, normal_rands[t])
            stage.add_tasks(task)

        return stage


    def generate_preproc_stage(self) -> Stage:
        cfg = self.cfg.machine_learning_stage
        stage = Stage()
        stage.name = "Preprocessing"

        # Generate normally-distributed pseudo-randoms
        normal_rands = generate_ttx(cfg.num_tasks, TX_PREPROC, 0.25)

        for t in range(0, cfg.num_tasks):
            tname = "Preproc-" + t
            stage.add_tasks(generate_task(cfg, tname, normal_rands[t]))

        return stage


    def generate_mlana_stage(self) -> Stage:
        cfg = self.cfg.machine_learning_stage
        stage = Stage()
        stage.name = "MachineLearning"

        # Generate normally-distributed pseudo-randoms
        normal_rands = generate_ttx(cfg.num_tasks, TX_ML, 0.25)

        for t in range(0, cfg.num_tasks):
            tname = "ML-" + t
            stage.add_tasks(generate_task(cfg, tname, normal_rands[t]))

        return stage


    def generate_agent_stage(self) -> Stage:
        cfg = self.cfg.agent_stage
        stage = Stage()
        stage.name = "AgentAna"

        normal_rands = generate_ttx(cfg.num_tasks, TX_AGENT, 0.05)

        for t in range(0, cfg.num_tasks):
            tname = "Agent-" + t
            task = generate_task(cfg, tname, normal_rands[t])
            stage.add_tasks(task)

        return stage


    def generate_async_stage(self) -> List[Pipeline]:
        """Generate a stage with the required number of each type of task.
        """
        s = Stage()
        s.name = "AsynchStage"
        
        # Simulation tasks
        cfg = self.cfg.molecular_dynamics_stage
        normal_rands = generate_ttx(cfg.num_tasks,
                                    TX_SIM, 1.0)

        # Number of simulation tasks per pipeline
        for t in range(0, cfg.num_tasks):
            task = generate_task(cfg, "Sim", normal_rands[t])
            s.add_tasks(task)

        # Preprocessing tasks
        cfg = self.cfg.machine_learning_stage
        normal_rands = generate_ttx(cfg.num_tasks, TX_PREPROC, 0.5)
        for t in range(0, cfg.num_tasks):
            s.add_tasks(generate_task(cfg, "Preproc", normal_rands[t]))

        # ML tasks
        cfg = self.cfg.machine_learning_stage
        normal_rands = generate_ttx(cfg.num_tasks, TX_ML, 0.5)
        for t in range(0, cfg.num_tasks):
            s.add_tasks(generate_task(cfg, "ML", normal_rands[t]))
        
        # Agent tasks
        cfg = self.cfg.agent_stage
        normal_rands = generate_ttx(cfg.num_tasks, TX_AGENT, 0.1)
        for t in range(0, cfg.num_tasks):
            task = generate_task(cfg, "Agent", normal_rands[t])
            s.add_tasks(task)

        return s


    def generate_async_pipeline(self) -> List[Pipeline]:
        pipeline = Pipeline()
        pipeline.add_stages(self.generate_async_stage())
        pipeline.add_stages(self.generate_async_stage())
        return pipeline


    def generate_final_pipeline(self) -> List[Pipeline]:
        pipeline = Pipeline()
        pipeline.add_stages(self.generate_preproc_stage())
        pipeline.add_stages(self.generate_mlana_stage())
        pipeline.add_stages(self.generate_agent_stage())
        return pipeline



    def mlana_pipeline(self) -> List[Pipeline]:
        pipelines = []
        num_models = 1

        for _ in range(num_models):
            self.pipeline = Pipeline()
            pre_stage = self.generate_preproc_stage()
            self.pipeline.add_stages(pre_stage)
            ml_stage = self.generate_mlana_stage()
            self.pipeline.add_stages(ml_stage)
            ana_stage = self.generate_agent_stage()
            self.pipeline.add_stages(ana_stage)
            pipelines.append(self.pipeline)
        return pipelines

