#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Async-friendly DDMD manager for orchestrating simulations
# ------------------------------------------------------------------------------

import asyncio
import yaml
from collections import OrderedDict
from rose import Learner
from ddmd.logger import Logger
import subprocess

def gpu_available():
    try:
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

class DDMD_manager:
    """
    Orchestrates the scheduling, monitoring, and cancellation of simulations
    in an AI-steered ensemble simulation workflow.
    """

    def __init__(self, asyncflow):
        self.learner = Learner(asyncflow)
        self.logger = Logger(use_colors=True)
        self.registered_sims = OrderedDict()   # Active simulations: {tag: asyncio.Task}
        self.sim_task_queue = asyncio.Queue()  # Queue of pending simulation inputs
        self.completed_sims = set()      # To Store completed simulations

        # Tuning parameters (should be configurable)
        self.time_between_predictions = 20     # Delay between prediction checks
        self.time_before_shutdown    = 5       # Grace period before shutdown
        self.debug = False

        if gpu_available():
            self.device = 'gpu'
        else:
            self.device = 'cpu'
        self.logger.info(f"DDMD Manager initialized on device: {self.device}")

        """These attributes should be defined in pipeline subclass:
        Make sure they are set correctly!
            - self.sim_batch_size
            - self.max_sim_batch
            - self.retrain_model
            - self.sim_batch_size
            - self.clean_unregistered_sims
            - self.selection: should be set to None if training task returns pretrained model
        _register_learner_tasks must be defined in subclass to register all learner tasks: simulation, training, active learning, prediction.
        """

 
    # --------------------------------------------------------------------------
    def check_prediction(self, pred):
        """
        Decide whether a simulation should be canceled based on prediction.
        Override this with actual logic in pipeline subclass.
        """
        raise NotImplementedError("check_prediction must be implemented")
    # --------------------------------------------------------------------------
    async def collect_sim_inputs(self):
        """
        Collect all simulation input files into task queue (sim_task_queue).
        Override this with actual logic in pipeline subclass.
        """
        raise NotImplementedError("collect_sim_inputs must be implemented")
    # --------------------------------------------------------------------------
    async def check_training_data(self):
        """
        Check if enough training data is available to start training.
        Override this with actual logic in pipeline subclass.
        """
        raise NotImplementedError("check_training_data must be implemented")
    # --------------------------------------------------------------------------
    async def del_files(self, sim_ind):
        """
        Delete all files associated with a simulation index (sim_ind) 
        if clean_unregistered_sims is set to True in pipeline subclass.
        Override this with actual logic if needed.
        """
        raise NotImplementedError("del_files must be implemented")
    # --------------------------------------------------------------------------
    async def train_model(self):
        """
        Define all step required for model training.
        Override this with actual logic in pipeline subclass.
        """
        raise NotImplementedError("train_model must be implemented")
    
    # --------------------------------------------------------------------------
    async def close(self):
        """Gracefully shutdown learner."""
        try:
            await self.learner.shutdown()
        except Exception:
            pass

    async def stop(self):
        """Alias for close(), can be used for external termination."""
        try:
            await self.learner.shutdown()
        except Exception:
            pass

    # --------------------------------------------------------------------------
    async def _unregister_sims(self, unregistered_sims, clean_unregistered_sims):
        """
            Remove completed or canceled simulations from the registry, 
            and optionally clean up files from canceled simulations 
            to exclude them from the training batch.
        """
        for tag in unregistered_sims:
            self.registered_sims.pop(tag, None)
            self.completed_sims.add(tag)
            if clean_unregistered_sims:
                #await asyncio.to_thread(self.del_files, tag)
                await self.del_files(tag)

        if unregistered_sims and not self.sim_task_queue.empty():
            # Adjust next batch size (ensure it does not exceed max_sim_batch)
            num_to_submit = min(self.sim_batch_size, self.sim_task_queue.qsize())
            if num_to_submit > 0:
                self.logger.info(
                f"{num_to_submit} simulations will start at next iteration"
            )

    # --------------------------------------------------------------------------
    async def submit_sims(self):
        """Submit simulations from the queue and register them."""
        while True:
            await self.monitor_sims()  # Clean up completed/failed tasks

            if self.sim_task_queue.empty():
                self.logger.info("No more simulation inputs in queue.")
                break

            if self.sim_batch_size <= 0:
                await asyncio.sleep(1)
                continue

            # Don't submit more than sim_batch_size simulation 
            # to have enough resources for training task
            num_to_submit = min(self.sim_batch_size, self.sim_task_queue.qsize())
            for _ in range(num_to_submit):
                try:
                    sim_inputs = self.sim_task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    self.logger.info("No more simulation inputs in queue.")
                    break

                simul = self.simulation(sim_inputs=sim_inputs)
                sim_tag = sim_inputs["sim_tag"]
                self.logger.task_started(f"Simulation {sim_tag}")
                self.registered_sims[sim_tag] = simul

            if num_to_submit > 0:
                self.logger.info(
                    f"[DDMD] Submitted {num_to_submit} new simulation(s)"
                )
            # Update sim_batch_size (subtract submitted items)
            self.sim_batch_size -= num_to_submit
            await asyncio.sleep(0.1)

    # --------------------------------------------------------------------------
    async def monitor_sims(self):
        """Unregister completed/failed simulations and prepare next batch."""
        unregistered_sims = []

        for sim_tag, task in self.registered_sims.items():   
            if task.done():
                unregistered_sims.append(sim_tag)
                self.logger.task_completed(f"Simulation {sim_tag}")
                self.sim_batch_size += 1

                if task.exception():
                    self.logger.error(
                        f"Simulation {sim_tag} failed: {task.exception()}"
                    )                   
        await self._unregister_sims(unregistered_sims, False)

    # --------------------------------------------------------------------------
    async def monitor_training_data(self):
        """Cancel sims when enough training data is available and free resources for model training."""
        while True:
            try:
                start_training = await self.check_training_data()
            except Exception as e:
                self.logger.error(f"Error while checking training data: {e}")
                await asyncio.sleep(self.time_between_predictions)
                continue

            if start_training and self.registered_sims:
                unregistered_sims = []
                resubmitted_sims = []
                count = 0
                total_sims = len(self.registered_sims)

                self.logger.info(
                    f"Preparing to cancel {self.training_cores} simulations "
                    f"out of {total_sims} to free resources for training"
                )

                # Suspend simulations to free up resources for training
                for sim_tag, task in list(self.registered_sims.items()):
                    if task.done():
                        unregistered_sims.append(sim_tag)
                    else:
                        try:
                            task.cancel()
                            unregistered_sims.append(sim_tag)
                            self.logger.task_killed(
                                f"Cancelling {sim_tag} to start training "
                                f"(ROSE task ID {getattr(task, 'id', 'N/A')})"
                            )
                            resubmitted_sims.append(sim_tag)
                        except Exception as e:
                            self.logger.error(f"Error cancelling {sim_tag}: {e}")
                            continue

                    count += 1
                    self.sim_batch_size += 1
                    if count >= self.training_cores:
                        break

                self.logger.info(f"Cancelled {count} simulations; Training will now start.")

                # Remove canceled sims from registry
                await self._unregister_sims(unregistered_sims, False)

                # Re-add canceled sims back to task queue for later rescheduling
                for sim_tag in resubmitted_sims:
                    await self.sim_task_queue.put({'sim_tag': sim_tag})
                    self.logger.info(f"Re-added {sim_tag} back to sim_task_queue")

                break  # Exit loop after canceling
            else:
                await asyncio.sleep(self.time_between_predictions)

    # --------------------------------------------------------------------------
    async def cancel_sims(self):
        """Cancel sims based on prediction score."""
        unregister_sims = []

        for sim_tag, pred in self.sim_predictions.items():

            if sim_tag not in self.registered_sims.keys():
                continue

            if self.debug:
                self.logger.info(f"{sim_tag} prediction: {pred}")

            if self.check_prediction(prediction=pred):
                task = self.registered_sims[sim_tag]
                task.cancel()
                unregister_sims.append(sim_tag)
                self.logger.task_killed(
                    f"Simulation {sim_tag} canceled due to prediction score {pred} "
                    f"(task ID {getattr(task, 'id', 'N/A')})"
                )
                self.sim_batch_size += 1

        await self._unregister_sims(unregister_sims, self.clean_unregistered_sims)

    # --------------------------------------------------------------------------
    async def teach(self):
        """
        Main event loop:
        - Collects input simulations
        - Submits and monitors tasks
        - Cancels based on predictions
        - Waits until all sims finish or queue empties
        """
        
        self.logger.separator("DDMD MANAGER STARTING")
        await self.collect_sim_inputs()
        submit_task = asyncio.create_task(self.submit_sims())

        # Skip waiting for training data if it is available at start
        if not self.force_start_training:
           await self.monitor_training_data()  # blocks until training starts

        # Release training resources and ensure the DDMD 
        # manager does not allocate them for simulations
        self.sim_batch_size -= self.training_cores  
        
        while True:
            self.logger.info(f"{len(self.registered_sims)} simulation(s) running...")
            if self.debug:
                self.logger.info(f"{list(self.registered_sims.keys())}")

            # Train model if flag is set
            if self.retrain_model:
                train = await self.train_model()
            else:
                await asyncio.sleep(self.time_between_predictions) 

            # Perform model selection if selection function is provided
            if self.selection:
                selected_model = await self.selection(train)
                # Perform prediction
                self.logger.task_started("Prediction")
            else:
                selected_model = train

            # Collect prediction scores for all simulations
            if self.run_prediction_as_exe:
                predict = await self.exe_prediction(selected_model)
                with open(self.prediction_file, 'r') as f:
                    predictions = yaml.safe_load(f)
            else:
                predictions = await self.prediction(selected_model)

            self.sim_predictions = predictions
            self.logger.info(f"[Task-Prediction] completed with {len(predictions)} results.")

            if predictions:
                await self.cancel_sims()

            if self.sim_task_queue.empty():
                await self.monitor_sims()

            # Exit if no sims left
            if self.sim_task_queue.empty() and len(self.registered_sims) == 0:
                break

            await asyncio.sleep(1) 

        await submit_task
        self.logger.manager_exiting()
        self.logger.separator("DDMD MANAGER FINISHED")
