# Contributing to DeepDriveMD

Thank you for your interest in contributing to DeepDriveMD! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior vs actual behavior
4. Your environment (Python version, OS, etc.)
5. Any relevant log output or error messages

### Suggesting Features

Feature requests are welcome! Please open an issue with:

1. A clear description of the feature
2. The use case or problem it solves
3. Any implementation ideas you have

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Set up your development environment**:
   ```bash
   pip install -e ".[dev,lint]"
   ```

3. **Make your changes** following our coding standards (see below)

4. **Add tests** for any new functionality

5. **Run the test suite** to ensure nothing is broken:
   ```bash
   pytest tests/unit
   ```

6. **Run linting** to check code style:
   ```bash
   ruff check ddmd tests
   ruff format --check ddmd tests
   ```

7. **Commit your changes** with a clear commit message:
   ```bash
   git commit -m "Add feature: description of the feature"
   ```

8. **Push to your fork** and submit a pull request

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/DeepDriveMD.git
cd DeepDriveMD

# Install in development mode with all dependencies
pip install -e ".[dev,lint,doc]"
```

### Running Tests

```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run with coverage
pytest --cov=ddmd --cov-report=html

# Run specific test file
pytest tests/unit/test_logger.py

# Run specific test
pytest tests/unit/test_logger.py::TestLoggerInit::test_default_initialization
```

### Using tox

```bash
# Run tests across all Python versions
tox

# Run only linting
tox -e lint

# Run only formatting check
tox -e format

# Run coverage report
tox -e coverage
```

## Coding Standards

### Style Guide

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting. The configuration is in `pyproject.toml`.

Key style points:
- Line length: 88 characters
- Use double quotes for strings
- Use spaces for indentation (4 spaces)
- Follow PEP 8 guidelines

### Code Formatting

Before committing, format your code:

```bash
ruff format ddmd tests
ruff check --fix ddmd tests
```

### Type Hints

We encourage the use of type hints for function signatures:

```python
async def check_train_data(self) -> bool:
    """Check if enough training data is available."""
    ...
```

### Docstrings

Use descriptive docstrings for classes and public methods:

```python
class MyWorkflow(DDMD_manager):
    """
    Custom workflow for specific simulation type.

    This workflow handles X, Y, and Z by doing A, B, and C.
    """

    def stop_simulation(self, prediction: float) -> bool:
        """
        Decide whether to stop a simulation based on prediction score.

        Args:
            prediction: The ML model's prediction score (0.0 to 1.0)

        Returns:
            True if the simulation should be stopped, False otherwise
        """
        ...
```

### Async Code

- Use `async`/`await` consistently
- Prefer `asyncio.gather()` for parallel operations
- Use `asyncio.to_thread()` for blocking I/O operations

## Project Structure

```
DeepDriveMD/
├── ddmd/                    # Main package
│   ├── __init__.py
│   ├── ddmd_manager.py      # Base manager class
│   ├── logger.py            # Logging utilities
│   └── pipelines/           # Workflow implementations
│       ├── dummy_learner.py
│       └── miniapps_pipeline.py
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Example implementations
├── doc/                    # Documentation
├── pyproject.toml          # Project configuration
└── tox.ini                 # Test automation
```

## Creating a New Workflow

To create a custom workflow, extend `DDMD_manager`:

```python
from ddmd import DDMD_manager

class MyWorkflow(DDMD_manager):
    def __init__(self, asyncflow, **kwargs):
        super().__init__(asyncflow)
        self._register_learner_tasks()

    def _register_learner_tasks(self):
        # Register your simulation, training, and prediction tasks
        pass

    # Implement required abstract methods
    def stop_simulation(self, prediction):
        ...

    async def init_sim_queue(self):
        ...

    async def check_train_data(self):
        ...

    async def train_model(self):
        ...

    async def clean_sim_data(self, sim_ind):
        ...
```

## Questions?

If you have questions about contributing, feel free to:

1. Open an issue on GitHub
2. Contact the maintainers at mariya.goliyad@rutgers.edu

## License

By contributing to DeepDriveMD, you agree that your contributions will be licensed under the MIT License.
