import numpy as np
from typing import List
from radical.entk import Pipeline, Stage, Task


def generate_task(cfg, name, ttx) -> Task:
    task = Task()
    # task.name = name
    task.executable = cfg.executable
    task.arguments = ['--cpu', '1', '--timeout', '%s' % ttx]
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


    def generate_mlana_stage(self) -> Stage:
        cfg = self.cfg.machine_learning_stage
        stage = Stage()
        stage.name = "MLAna"

        # Generate normally-distributed pseudo-randoms
        normal_rands = generate_ttx(cfg.num_tasks, 28.2, 0.5)

        for t in range(0, cfg.num_tasks):
            stage.add_tasks(generate_task(cfg, "MLAna", normal_rands[t]))

        return stage


    def generate_agent_stage(self) -> Stage:
        cfg = self.cfg.agent_stage
        stage = Stage()
        stage.name = "Agent"

        normal_rands = generate_ttx(cfg.num_tasks, 11.1, 0.25)

        for t in range(0, cfg.num_tasks):
            task = generate_task(cfg, "Agent", normal_rands[t])
            stage.add_tasks(task)

        return stage


    def sim_pipeline_full(self) -> List[Pipeline]:
        cfg = self.cfg.molecular_dynamics_stage
        pipelines = []

        for p in range(0, self.cfg.num_nodes):
            self.pipeline = Pipeline()
            stage = Stage()
            stage.name = "Simulation"

            # Generate normally-distributed pseudo-randoms
            normal_rands = generate_ttx(cfg.num_tasks, 59.1, 2.0)

            for t in range(0, cfg.num_tasks):
                task = generate_task(cfg,
                    "Sim", normal_rands[t])
                stage.add_tasks(task)

            self.pipeline.add_stages(stage)
            pipelines.append(self.pipeline)

        return pipelines

    def sim_pipeline_part(self) -> List[Pipeline]:
        cfg = self.cfg.molecular_dynamics_stage
        pipelines = []
        num_models = 1
        
        # Use all nodes except the number used by ML+ana
        for p in range(0, self.cfg.num_nodes - num_models):
            self.pipeline = Pipeline()
            stage = Stage()
            stage.name = "Simulation"

            # Generate normally-distributed pseudo-randoms
            normal_rands = generate_ttx(cfg.num_tasks, 59.1, 2.0)

            for t in range(0, cfg.num_tasks):
                task = generate_task(cfg,
                    "Sim", normal_rands[t])
                stage.add_tasks(task)

            self.pipeline.add_stages(stage)
            pipelines.append(self.pipeline)

        return pipelines

    def mlana_pipeline(self) -> List[Pipeline]:
        pipelines = []
        num_models = 1

        for _ in range(num_models):
            self.pipeline = Pipeline()
            ml_stage = self.generate_mlana_stage()
            self.pipeline.add_stages(ml_stage)
            ana_stage = self.generate_agent_stage()
            self.pipeline.add_stages(ana_stage)
            pipelines.append(self.pipeline)
        return pipelines

