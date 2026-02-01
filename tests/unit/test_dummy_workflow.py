import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

from ddmd.pipelines.dummy_learner import DummyWorkflow


class TestDummyWorkflowInit:
    """Test DummyWorkflow initialization."""

    @pytest.fixture
    def temp_home(self):
        """Create a temporary home directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_asyncflow(self):
        """Create a mock asyncflow engine."""
        mock = MagicMock()
        return mock

    def test_default_config_values(self, temp_home, mock_asyncflow):
        """Test that default configuration values are set correctly."""
        workflow = DummyWorkflow(
            asyncflow=mock_asyncflow,
            home_dir=temp_home
        )
        assert workflow.max_sim_batch == 4
        assert workflow.training_cores == 1
        assert workflow.training_threshold == 0.5
        assert workflow.prediction_threshold == 0.5
        assert workflow.force_start_training is False

    def test_custom_config_values(self, temp_home, mock_asyncflow):
        """Test that custom configuration values are respected."""
        workflow = DummyWorkflow(
            asyncflow=mock_asyncflow,
            home_dir=temp_home,
            max_sim_batch=8,
            training_cores=2,
            training_threshold=0.7,
            prediction_threshold=0.6,
            force_start_training=True
        )
        assert workflow.max_sim_batch == 8
        assert workflow.training_cores == 2
        assert workflow.training_threshold == 0.7
        assert workflow.prediction_threshold == 0.6
        assert workflow.force_start_training is True

    def test_directories_created(self, temp_home, mock_asyncflow):
        """Test that workflow directories are created."""
        workflow = DummyWorkflow(
            asyncflow=mock_asyncflow,
            home_dir=temp_home
        )
        assert workflow.sim_output_dir.exists()
        assert workflow.sim_inputs_dir.exists()
        assert workflow.train_dir.exists()
        assert workflow.train_al_dir.exists()
        assert workflow.val_dir.exists()

    def test_sim_inputs_generated(self, temp_home, mock_asyncflow):
        """Test that simulation input files are generated."""
        num_files = 5
        workflow = DummyWorkflow(
            asyncflow=mock_asyncflow,
            home_dir=temp_home,
            num_files=num_files
        )
        input_files = list(workflow.sim_inputs_dir.glob("*.npz"))
        assert len(input_files) == num_files


class TestDummyWorkflowStaticMethods:
    """Test static helper methods."""

    def test_ensure_dir_creates_new_directory(self):
        """Test _ensure_dir creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_subdir"
            result = DummyWorkflow._ensure_dir(new_dir)
            assert result.exists()
            assert result.is_dir()

    def test_ensure_dir_existing_directory(self):
        """Test _ensure_dir handles existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = DummyWorkflow._ensure_dir(temp_dir)
            assert result.exists()

    def test_clean_dir_removes_directory(self):
        """Test _clean_dir removes existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "to_delete"
            test_dir.mkdir()
            (test_dir / "file.txt").write_text("test")

            DummyWorkflow._clean_dir(test_dir)
            assert not test_dir.exists()

    def test_clean_dir_nonexistent_directory(self):
        """Test _clean_dir handles nonexistent directory gracefully."""
        # Should not raise an exception
        DummyWorkflow._clean_dir("/nonexistent/path/12345")

    def test_generate_sim_inputs_creates_npz_files(self):
        """Test _generate_sim_inputs creates correct number of .npz files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            num_files = 3
            DummyWorkflow._generate_sim_inputs(temp_dir, num_files=num_files)

            files = list(Path(temp_dir).glob("*.npz"))
            assert len(files) == num_files

            # Verify file contents
            for f in files:
                data = np.load(f)
                assert 'X' in data
                assert data['X'].shape == (100, 1)


class TestDummyWorkflowStopSimulation:
    """Test stop_simulation logic."""

    @pytest.fixture
    def workflow(self):
        """Create a workflow with mocked asyncflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_asyncflow = MagicMock()
            workflow = DummyWorkflow(
                asyncflow=mock_asyncflow,
                home_dir=temp_dir,
                prediction_threshold=0.5
            )
            yield workflow

    def test_stop_simulation_below_threshold(self, workflow):
        """Test that simulations below threshold are stopped."""
        assert workflow.stop_simulation(prediction=0.3) is True
        assert workflow.stop_simulation(prediction=0.1) is True
        assert workflow.stop_simulation(prediction=0.49) is True

    def test_stop_simulation_above_threshold(self, workflow):
        """Test that simulations above threshold are not stopped."""
        assert workflow.stop_simulation(prediction=0.6) is False
        assert workflow.stop_simulation(prediction=0.9) is False
        assert workflow.stop_simulation(prediction=1.0) is False

    def test_stop_simulation_at_threshold(self, workflow):
        """Test behavior at exactly threshold."""
        # At threshold (0.5), prediction is not < threshold, so should not stop
        assert workflow.stop_simulation(prediction=0.5) is False


class TestDummyWorkflowAsync:
    """Test async methods."""

    @pytest.fixture
    def workflow(self):
        """Create a workflow with mocked asyncflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_asyncflow = MagicMock()
            workflow = DummyWorkflow(
                asyncflow=mock_asyncflow,
                home_dir=temp_dir,
                num_files=3,
                start_training_threshold=2
            )
            yield workflow

    @pytest.mark.asyncio
    async def test_init_sim_queue_populates_queue(self, workflow):
        """Test that init_sim_queue populates the simulation queue."""
        await workflow.init_sim_queue()

        # Should have 3 items (num_files=3)
        assert workflow.sim_task_queue.qsize() == 3

    @pytest.mark.asyncio
    async def test_check_train_data_insufficient(self, workflow):
        """Test check_train_data returns False when insufficient data."""
        # No output files yet
        result = await workflow.check_train_data()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_train_data_sufficient(self, workflow):
        """Test check_train_data returns True when sufficient data."""
        # Create enough output files
        sim_dir = workflow.sim_output_dir / "sim_0"
        sim_dir.mkdir(parents=True)
        for i in range(3):
            (sim_dir / f"output_{i}.txt").write_text("data")

        result = await workflow.check_train_data()
        assert result is True

    @pytest.mark.asyncio
    async def test_clean_sim_data_removes_files(self, workflow):
        """Test clean_sim_data removes simulation files."""
        sim_ind = "test_sim"

        # Create test files
        sim_dir = workflow.sim_output_dir / sim_ind
        sim_dir.mkdir(parents=True)
        (sim_dir / "output.txt").write_text("data")

        train_file = workflow.train_dir / f"{sim_ind}_train.txt"
        train_file.write_text("train data")

        # Clean the simulation data
        await workflow.clean_sim_data(sim_ind)

        # Verify cleanup
        assert not sim_dir.exists()
        assert not train_file.exists()


class TestDummyWorkflowCollectPredictions:
    """Test prediction collection."""

    @pytest.fixture
    def workflow(self):
        """Create a workflow with mocked asyncflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_asyncflow = MagicMock()
            workflow = DummyWorkflow(
                asyncflow=mock_asyncflow,
                home_dir=temp_dir
            )
            yield workflow

    @pytest.mark.asyncio
    async def test_collect_predictions_reads_yaml(self, workflow):
        """Test that collect_predictions reads YAML file correctly."""
        # Create prediction file
        predictions = {"sim_0": 0.8, "sim_1": 0.3}
        import yaml
        with open(workflow.prediction_file, 'w') as f:
            yaml.dump(predictions, f)

        result = await workflow.collect_predictions()
        assert result == predictions


class TestDummyWorkflowFinalize:
    """Test finalize_pipeline method."""

    @pytest.fixture
    def workflow(self):
        """Create a workflow with mocked asyncflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_asyncflow = MagicMock()
            workflow = DummyWorkflow(
                asyncflow=mock_asyncflow,
                home_dir=temp_dir
            )
            yield workflow

    @pytest.mark.asyncio
    async def test_finalize_pipeline_exports_stats(self, workflow):
        """Test finalize_pipeline exports cancel_stats to JSON."""
        import json

        workflow.cancel_stats = {
            "total": 2,
            "timestamps": [
                {"simulation": "sim_0", "score": 0.3, "timestamp": "10:30"},
                {"simulation": "sim_1", "score": 0.2, "timestamp": "10:35"}
            ]
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        result = await workflow.finalize_pipeline(path=output_path)

        # Verify file was created and contains correct data
        with open(output_path) as f:
            saved_data = json.load(f)

        assert saved_data["total"] == 2
        assert len(saved_data["timestamps"]) == 2

        # Cleanup
        Path(output_path).unlink()

    @pytest.mark.asyncio
    async def test_finalize_pipeline_raises_without_stats(self, workflow):
        """Test finalize_pipeline raises error if cancel_stats doesn't exist."""
        # Remove cancel_stats if it exists
        if hasattr(workflow, 'cancel_stats'):
            delattr(workflow, 'cancel_stats')

        with pytest.raises(AttributeError):
            await workflow.finalize_pipeline()
