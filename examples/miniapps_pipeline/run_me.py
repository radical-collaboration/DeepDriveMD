#!/usr/bin/env python3
import asyncio
from radical.asyncflow import WorkflowEngine, DragonExecutionBackendV3
from radical.asyncflow import ConcurrentExecutionBackend
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from radical.asyncflow import RadicalExecutionBackend
from ddmd import MiniAppsWorkflow
# from radical.asyncflow import DaskExecutionBackend


SIM_CORES = 63
TRAIN_CORE = 1
TOTAL_CORES= SIM_CORES + TRAIN_CORE

RESOURCES = {
            'runtime': 300, 
           # 'resource': 'local.localhost', 
            'resource': 'purdue.anvil',
            'cores': TOTAL_CORES
        }

raptor_config = {
    "masters": [{
        "ranks": 1,
        "workers": [{
            "ranks": TRAIN_CORE
        }]
    }]
}

async def run_ddmd():

    #engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    #engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    engine = await RadicalExecutionBackend(RESOURCES)   #, raptor_config)
    # Create Dragon backend - it does not work with current version of miniApps - mpi4py is not compatible with dragonhpc.
    #engine = await DragonExecutionBackendV3(num_workers=15, disable_background_batching=False)

    # Create the async workflow engine
    asyncflow = await WorkflowEngine.create(engine)
    
    # Initialize the workflow
    workflow = MiniAppsWorkflow(asyncflow=asyncflow, training_cores=TRAIN_CORE, max_sim_batch=SIM_CORES)
    
    #try:
        # Run the workflow
    await workflow.teach()
    #except Exception as e:
    #    print(f"An error occurred during teaching: {e}")
    #finally:
        # Ensure cleanup regardless of errors
    await workflow.close()

if __name__ == '__main__':
    asyncio.run(run_ddmd())