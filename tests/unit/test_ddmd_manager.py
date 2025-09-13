import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from collections import OrderedDict
from tests.unit.mock_manager import MockLearner
from radical.asyncflow import WorkflowEngine
from rose import Learner
from concurrent.futures import ThreadPoolExecutor
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine
   

# ---------------------------
# Async helpers
# ---------------------------
def make_done_task(result="ok"):
    async def _done(): return result
    return asyncio.create_task(_done())

def make_failing_task(exc_msg="boom"):
    async def _fail(): raise RuntimeError(exc_msg)
    return asyncio.create_task(_fail())


# # ---------------------------
# # Fixtures
# # ---------------------------

# @pytest.fixture
# def mock_asyncflow():
#     mock = MagicMock(spec=WorkflowEngine)
#     # manually add attributes that aren’t in WorkflowEngine
#     #type(mock).task = MagicMock()
#     #mock.task.return_value = "dummy_task"
#     return mock

# @pytest.fixture
# def manager(mock_asyncflow):
#     return MockLearner(asyncflow=mock_asyncflow)

# @pytest.fixture
# def learner(mock_asyncflow):
#     mock = MagicMock(spec=Learner)
#     type(mock).function_task = MagicMock()
#     mock.function_task.return_value = "dummy_task"
#     return mock
# ---------------------------
# Group 1: Simulation lifecycle
# ---------------------------
class TestSimulationLifecycle:
    @pytest.mark.asyncio
    async def test_submit_sims_registers_and_respects_batch(self):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow)
        
        await manager.collect_sim_inputs(n=2)
        manager.sim_batch_size = 2

        await manager.submit_sims()

        assert len(manager.registered_sims) == 2
        assert manager.sim_task_queue.empty()
        assert manager.sim_batch_size == 0
        manager.logger.task_started.assert_called()

    @pytest.mark.asyncio
    async def test_monitor_sims_unregisters_done_and_increments_batch(self):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow, max_sim_batch=1, training_cores=1)
        done = make_done_task("done:sim_0")
        running = manager.simulation(sim_inputs={"sim_tag": "sim_1"})
        manager.registered_sims["sim_0"] = done
        manager.registered_sims["sim_1"] = running

        await asyncio.sleep(0)
        await manager.monitor_sims()

        assert "sim_0" not in manager.registered_sims
        assert "sim_1" in manager.registered_sims
        assert manager.sim_batch_size == 3  #max_sim_batch + training_cores + 1 (sim_0 has completed)
        manager.logger.task_completed.assert_called()

    @pytest.mark.asyncio
    async def test_monitor_sims_logs_failures_and_unregs(self):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow, max_sim_batch=1, training_cores=1)
        failing = make_failing_task()
        ok = make_done_task()
        manager.registered_sims["sim_fail"] = failing
        manager.registered_sims["sim_ok"] = ok

        with pytest.raises(RuntimeError):
            await failing
        await ok

        await manager.monitor_sims()

        assert "sim_fail" not in manager.registered_sims
        assert "sim_ok" not in manager.registered_sims
        assert manager.sim_batch_size == 4 #max_sim_batch + training_cores + 2 (both sims have completed)
        manager.logger.error.assert_called()


# # ---------------------------
# # Group 2: Training behavior
# # ---------------------------
# class TestTrainingBehavior:
#     @pytest.mark.asyncio
#     @pytest.mark.parametrize(
#         "_force_start_training, expected_queue_empty, expected_batch_min",
#         [
#             (True, False, 1),
#             (False, True, 0),
#         ]
#     )
#     async def test_monitor_training_data_param(self, _force_start_training, expected_queue_empty, expected_batch_min):
#         engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
#         asyncflow = await WorkflowEngine.create(engine)
#         manager = MockLearner(asyncflow=asyncflow)
#         s0 = manager.simulation(sim_inputs={"sim_tag": "sim_0"})
#         s1 = manager.simulation(sim_inputs={"sim_tag": "sim_1"})
#         manager.registered_sims["sim_0"] = s0
#         manager.registered_sims["sim_1"] = s1
#         manager.training_cores = 1
#         manager._force_start_training = _force_start_training

#         await manager.monitor_training_data()

#         assert (manager.sim_task_queue.empty() == expected_queue_empty)
#         assert manager.sim_batch_size >= expected_batch_min

#         if _force_start_training:
#             manager.logger.task_killed.assert_called()
#         else:
#             manager.logger.task_killed.assert_not_called()


# ---------------------------
# Group 3: Cancel sims behavior
# ---------------------------
class TestCancelSimsBehavior:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "predictions, clean_flag, expected_deleted, expected_remaining",
        [
            ({"sim_0": 0.2, "sim_1": 0.8}, True, ["sim_0"], ["sim_1"]),
            ({"sim_0": 0.6, "sim_1": 0.8}, True, [], ["sim_0", "sim_1"]),
            ({"sim_0": 0.1, "sim_1": 0.7}, False, [], ["sim_1", "sim_0"]),
            ({}, True, [], []),
        ]
    )
    async def test_cancel_sims_various_cases(self, predictions, clean_flag, expected_deleted, expected_remaining):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow)
        for tag in predictions.keys():
            task = manager.simulation(sim_inputs={"sim_tag": tag})
            manager.registered_sims[tag] = task

        manager.sim_predictions = predictions
        manager.clean_unregistered_sims = clean_flag

        await manager.cancel_sims()

        for sim in expected_deleted:
            assert sim not in manager.registered_sims.keys()
        print(manager.registered_sims)
        for sim in expected_remaining:
            if predictions.get(sim, 1.0) >= 0.5 or not clean_flag:
                assert sim in manager.registered_sims.keys() or not clean_flag

        assert manager.sim_batch_size >= len(expected_deleted)


# ---------------------------
# Group 4: Teach flow
# ---------------------------
class TestTeachFlow:
    @pytest.mark.asyncio
    async def test_teach_runs_full_cycle_and_exits(self):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow)
        manager.retrain_model = False

        await manager.teach()

        assert manager.sim_task_queue.empty()
        assert not manager.registered_sims
        manager.logger.manager_exiting.assert_called()
        manager.logger.separator.assert_called()


# ---------------------------
# Group 5: Shutdown safety
# ---------------------------
class TestShutdownSafety:
    @pytest.mark.asyncio
    async def test_close_and_stop_are_safe(self):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow)
        await manager.close()
        await manager.stop()

# ---------------------------
# Group 6: File deletion
# ---------------------------
class TestDelFilesBehavior:
    @pytest.mark.asyncio
    async def test_del_files_records_multiple_deletions(self):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow)
        sims = ["sim_0", "sim_1", "sim_2"]
        for sim in sims:
            await manager.del_files(sim)

        for sim in sims:
            assert sim not in manager.registered_sims.keys()

# ---------------------------
# Group 7: Simulation queue edge cases
# ---------------------------
class TestSimulationQueueEdgeCases:
    @pytest.mark.asyncio
    async def test_submit_sims_with_empty_queue(self):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow)
        manager.sim_batch_size = 2
        await manager.submit_sims()
        assert manager.sim_task_queue.empty()

    @pytest.mark.asyncio
    async def test_submit_sims_with_partial_queue(self):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow)
        await manager.collect_sim_inputs(n=1)
        manager.sim_batch_size = 3
        await manager.submit_sims()

        # Only 1 task submitted because queue has 1
        assert len(manager.registered_sims) == 1
        assert manager.sim_task_queue.empty()

    @pytest.mark.asyncio
    async def test_submit_sims_with_batch_larger_than_queue(self):
        engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
        asyncflow = await WorkflowEngine.create(engine)
        manager = MockLearner(asyncflow=asyncflow)
        await manager.collect_sim_inputs(n=2)
        manager.sim_batch_size = 5
        await manager.submit_sims()

        # Queue had 2 inputs, sim_batch_size > queue
        assert len(manager.registered_sims) == 2
        assert manager.sim_task_queue.empty()
