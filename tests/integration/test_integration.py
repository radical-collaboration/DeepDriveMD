
import asyncio
import pytest
from tests.unit.mock_manager import MockLearner
from concurrent.futures import ThreadPoolExecutor
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine

@pytest.mark.asyncio
async def test_integration():
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    manager = MockLearner(asyncflow=asyncflow)

    await manager.teach()
    assert manager.registered_sims == {}
    assert manager.sim_task_queue.empty()
    await manager.close()