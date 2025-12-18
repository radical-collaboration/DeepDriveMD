#!/usr/bin/env python3
import asyncio
import os
from radical.asyncflow import WorkflowEngine
from ddmd import NoopWorkflow
import argparse



async def run_ddmd(backend='rp', simulation_cores=16, train_cores=1):

    total_cores = simulation_cores + train_cores
    
    if backend == 'rp':
        #from radical.asyncflow import ConcurrentExecutionBackend
        #from concurrent.futures import ProcessPoolExecutor
        from radical.asyncflow import RadicalExecutionBackend

        RESOURCES = {
            'runtime': 30, 
            #'resource': 'local.localhost', 
            'resource': 'purdue.anvil',
            'cores': total_cores
        }

        raptor_config = {
            "masters": [{
                "ranks": 1,
                "workers": [{
                    "ranks": 1
                }]
            }]
        }

        #engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
        engine = await RadicalExecutionBackend(RESOURCES, raptor_config)

    elif backend == 'dragon':
        from radical.asyncflow import DragonExecutionBackendV3
        from radical.asyncflow import DragonTelemetryCollector
        import multiprocessing as mp

        collector_dir = 'telemetry_results'

        mp.set_start_method("dragon")
        engine = await DragonExecutionBackendV3(num_workers=2*total_cores)   #, disable_background_batching=False)

        collector = DragonTelemetryCollector(
            collection_rate=1.0,              # Collect every second
            checkpoint_interval=30.0,         # Checkpoint every 30 seconds
            checkpoint_dir=os.path.join(os.getcwd(), collector_dir),  # Save checkpoints here
            checkpoint_count=10,              # Keep last 10 checkpoints
            enable_cpu=True,
            enable_gpu=True,
            enable_memory=True,
            metric_prefix="infer-asyncflow"   # Prefix all metrics
        )

        # Start collection (spawns processes on all nodes)
        collector.start()
        
    else:
        print('ERROR: Please rerun with "--backend dragon" or  "--backend rp"...')
        return

    # Create the async workflow engine
    asyncflow = await WorkflowEngine.create(engine)
    # Initialize the workflow
    workflow = NoopWorkflow(asyncflow=asyncflow, training_cores=train_cores, max_sim_batch=simulation_cores)

   # try:
        # Run the workflow
    await workflow.teach()
    #except Exception as e:
    #    print(f"An error occurred during teaching: {e}")
    #finally:
        # Ensure cleanup regardless of errors
    await workflow.close()
    if backend == 'dragon':
        collector.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot CPU/GPU telemetry from JSON files.")
    parser.add_argument("--backend", default="rp", type=str, help="Asyncflow Backend to use (rp or dragon) ")
    parser.add_argument("--simulation_cores", default=16, type = int, help="Number of cores reserved for simulation ")
    parser.add_argument("--train_cores", default=1, type = int, help="Number of cores reserved for ML training ")

    args = parser.parse_args()
    asyncio.run(run_ddmd(args.backend, args.simulation_cores, args.train_cores))

# python run_me.py --backend rp --simulation_cores 4 --train_cores 1

# dragon run_me.py --backend dragon --simulation_cores 4 --train_cores 1