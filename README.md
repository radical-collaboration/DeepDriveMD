# DeepDriveMD

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Deep learning-driven Adaptive Molecular Simulations for Protein Folding**

DeepDriveMD is a toolkit developed by Brookhaven National Laboratory (BNL) / RADICAL Laboratory at Rutgers University, in collaboration with Argonne National Laboratory. It implements an AI-steered ensemble simulation workflow that uses deep learning models to guide and optimize protein folding simulations in real-time.

## Features

- **Adaptive Simulation Management**: Dynamically manages molecular simulations based on ML predictions
- **Active Learning Loop**: Implements simulation → training → prediction → cancellation → re-submission cycle
- **Multiple Execution Backends**: Supports local execution, RADICAL-Pilot (HPC), and Dragon distributed computing
- **Resource-Aware Scheduling**: Automatically balances resources between simulations and training
- **GPU Support**: Automatic GPU detection and utilization
- **Extensible Architecture**: Easy to customize for different simulation types and ML models

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DDMD Manager                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Simulation  │  │  Training   │  │     Prediction      │  │
│  │   Queue     │──│   Module    │──│      Module         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                │                    │             │
│         ▼                ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              ROSE / RADICAL-AsyncFlow               │    │
│  │           (Execution Backend Abstraction)           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### From Source

```bash
git clone https://github.com/radical-collaboration/DeepDriveMD.git
cd DeepDriveMD
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With Documentation Dependencies

```bash
pip install -e ".[doc]"
```

## Quick Start

### 1. Basic Usage with DummyWorkflow (for testing)

```python
import asyncio
from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine
from concurrent.futures import ThreadPoolExecutor
from ddmd import DummyWorkflow

async def main():
    # Create execution backend
    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)

    # Initialize workflow
    workflow = DummyWorkflow(
        asyncflow=asyncflow,
        max_sim_batch=4,
        training_cores=1,
        num_files=10
    )

    # Run the adaptive learning loop
    await workflow.teach()
    await workflow.close()

asyncio.run(main())
```

### 2. Creating a Custom Workflow

Extend `DDMD_manager` to create your own workflow:

```python
from ddmd import DDMD_manager

class MyWorkflow(DDMD_manager):
    def __init__(self, asyncflow, **kwargs):
        super().__init__(asyncflow)
        # Your initialization code
        self._register_learner_tasks()

    def _register_learner_tasks(self):
        @self.learner.simulation_task(as_executable=False)
        async def simulation(*args, **kwargs):
            # Your simulation logic
            pass
        self.simulation = simulation

    def stop_simulation(self, prediction):
        # Return True to cancel simulation based on prediction
        return prediction < 0.5

    async def init_sim_queue(self):
        # Populate self.sim_task_queue with simulation inputs
        pass

    async def check_train_data(self):
        # Return True when ready to start training
        return True

    async def train_model(self):
        # Your training logic
        pass

    async def clean_sim_data(self, sim_ind):
        # Cleanup files for canceled simulations
        pass
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_sim_batch` | Maximum concurrent simulations | 4 |
| `training_cores` | CPU cores reserved for training | 1 |
| `training_threshold` | Accuracy threshold for training | 0.5 |
| `prediction_threshold` | Score threshold for cancellation | 0.5 |
| `force_start_training` | Skip waiting for data threshold | False |
| `clean_unregistered_sims` | Delete files from canceled sims | True |

## Examples

See the [examples/](examples/) directory for complete working examples:

- **[dummy_pipeline/](examples/dummy_pipeline/)**: Standalone demo with synthetic data
- **[miniapps_pipeline/](examples/miniapps_pipeline/)**: Production HPC pipeline

## Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run with coverage
pytest --cov=ddmd --cov-report=html
```

## Development

### Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check code style
ruff check ddmd tests

# Format code
ruff format ddmd tests
```

### Using tox

```bash
# Run all tests across Python versions
tox

# Run linting
tox -e lint

# Run formatting
tox -e format
```

## Dependencies

- **[RADICAL-AsyncFlow](https://github.com/radical-cybertools/radical-asyncflow)**: Async workflow orchestration
- **[ROSE](https://radical-cybertools.github.io/ROSE/)**: Machine learning integration for HPC
- **[PyYAML](https://pyyaml.org/)**: Configuration file parsing

## Citation

If you use DeepDriveMD in your research, please cite:

```bibtex
@inproceedings{lee2019deepdrivemd,
  author={Lee, Hyungro and Turilli, Matteo and Jha, Shantenu and Bhowmik, Debsindhu and Ma, Heng and Ramanathan, Arvind},
  booktitle={2019 IEEE/ACM Third Workshop on Deep Learning on Supercomputers (DLS)},
  title={DeepDriveMD: Deep-Learning Driven Adaptive Molecular Simulations for Protein Folding},
  year={2019},
  pages={12-19},
  doi={10.1109/DLS49591.2019.00007}
}
```

**Paper**: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8945122)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Brookhaven National Laboratory (BNL)
- RADICAL Laboratory at Rutgers University
- Argonne National Laboratory
- This work was supported by the DOE Office of Science
