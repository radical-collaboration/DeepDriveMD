import asyncio
import sys
import yaml
import os
import random
import shutil
from pathlib import Path
from ddmd.ddmd_manager import DDMD_manager

class NoopWorkflow(DDMD_manager):
    """Dummy workflow for managing DDMD simulations, training, and predictions."""

    def __init__(self, **kwargs):

        # Initialize parent class (sets up asyncflow, logger, queues, etc.)
        asyncflow = kwargs.get('asyncflow')
        super().__init__(asyncflow)
        
        # Paths for executables and model
        self.code_path = kwargs.get('code_path', f'{sys.executable} {os.getcwd()}')

        # Simulation/training config
        #################################
        # Max number of simulation to run at once
        self.max_sim_batch            = kwargs.get('max_sim_batch', 4)
        # Number of cores reserved for training
        self.training_cores           = kwargs.get('training_cores', 1)
        # Initial size of simulation batch before training starts
        self.sim_batch_size           = self.max_sim_batch + self.training_cores
        
        # Stop pipeline after all simulation are done
        self.total_num_sim = kwargs.get('total_num_sim', 25)
        #Training iteration
        self.iteration = 0
        self.total_num_iteration = 3
        self.sleep_time = 1
        self.retrain_model = True     # Stop training model if training accuracy has been achieved 

        # Tuning parameters (should be configurable)
        self.time_between_predictions = 2.0     # Delay between prediction checks

        # Register learner tasks
        self._register_learner_tasks()


    # --------------------------------------------------------------------------
    def _even_or_odd(self, x):     # 1 = even, 0 = odd
        return 0 if x % 20 == 0 else 1

    # --------------------------------------------------------------------------
    async def collect_predictions(self):
        predictions = {key: 0.5  for key in self.registered_sims.keys()}
        for sim_tag in self.registered_sims.keys():
            predictions[sim_tag] = self._even_or_odd(sim_tag)

        return predictions
    
    # --------------------------------------------------------------------------
    def stop_simulation(self, sim_tag, *args, **kwargs):
        """If True then simulation will be canceled"""
        if sim_tag % 20 == 0:
            return True
        else:
            return False

    # --------------------------------------------------------------------------
    async def init_sim_queue(self):
        """Collect all simulation inputs into task queue."""
        for s in range(self.total_num_sim):
            await self.sim_task_queue.put({'sim_tag': s})

    # --------------------------------------------------------------------------
    async def skip_training(self):
        '''Sleep instaed of training so prediction is not called too often'''
        await asyncio.sleep(self.time_between_predictions)

    # --------------------------------------------------------------------------
    def _register_learner_tasks(self):
        """Register learner tasks: simulation, training, active learning, prediction."""

        sleep_time=self.sleep_time

        @self.learner.simulation_task(as_executable=True)
        async def noop_simulation(*arg, **kwargs):
            sim_tag = kwargs["sim_tag"]
            #sleep_time += (sim_tag % 10) + 100
            #self.logger.info(f"Sim spleeps for {sleep_time}")
            return f'{self.code_path}/noop_simulation.py --sleep_time {sleep_time}'
            
        self.simulation = noop_simulation
      
        @self.learner.training_task(as_executable=True)
        async def noop_training(*arg, **kwargs):
            return f'{self.code_path}/noop_training.py --sleep_time {sleep_time}'
        
        self.training = noop_training

        @self.learner.prediction_task(as_executable=True)
        async def noop_prediction(*arg, **kwargs):
            return f'{self.code_path}/noop_prediction.py --sleep_time {sleep_time}'
            
        self.prediction = noop_prediction

        @self.learner.utility_task(as_executable=True)
        async def noop_selection(*arg, **kwargs):
            return f'{self.code_path}/noop_selection.py --sleep_time {sleep_time}'
        
        self.selection = noop_selection
      
    # --------------------------------------------------------------------------
    async def train_model(self):
        """Train until accuracy threshold is met or epochs are exhausted."""
        self.iteration += 1
        self.logger.info(f'\nTraining Iteration {self.iteration}')
        #self.logger.info(f'{len(self.registered_sims)} simulation(s) running....')

        if self.retrain_model:
            train_task = await self.training()
            selected_model = await self.selection()
        else:
            await self.skip_training()

        pred = await self.prediction()
        predictions = await self.collect_predictions()
        self.logger.task_completed('Model Prediction', component="prediction")

        if self.iteration == self.total_num_iteration:
            self.retrain_model = False
            self.sim_batch_size += self.training_cores
            self.logger.info(f'\nTraining & Simulation STOPPED at iteration {self.iteration}')
            self.sim_task_queue = asyncio.Queue()
        else:
            self.logger.task_completed('Training Completed')
        

        return predictions