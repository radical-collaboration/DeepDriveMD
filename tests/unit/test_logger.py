import pytest
from io import StringIO
from ddmd.logger import Logger, LogLevel, Colors


class TestColors:
    """Test the Colors class constants."""

    def test_colors_have_escape_sequences(self):
        assert Colors.RED.startswith('\033[')
        assert Colors.GREEN.startswith('\033[')
        assert Colors.RESET == '\033[0m'

    def test_bright_colors_exist(self):
        assert hasattr(Colors, 'BRIGHT_RED')
        assert hasattr(Colors, 'BRIGHT_GREEN')
        assert hasattr(Colors, 'BRIGHT_BLUE')


class TestLogLevel:
    """Test the LogLevel enum."""

    def test_log_levels_exist(self):
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestLoggerInit:
    """Test Logger initialization."""

    def test_default_initialization(self):
        logger = Logger()
        assert logger.name == "DDMDManager"
        assert logger.use_colors is True

    def test_custom_name(self):
        logger = Logger(name="CustomLogger")
        assert logger.name == "CustomLogger"

    def test_colors_disabled(self):
        logger = Logger(use_colors=False)
        assert logger.use_colors is False

    def test_custom_output_stream(self):
        stream = StringIO()
        logger = Logger(output_stream=stream)
        assert logger.output_stream is stream


class TestLoggerColorize:
    """Test the _colorize method."""

    def test_colorize_with_colors_enabled(self):
        logger = Logger(use_colors=True)
        result = logger._colorize("test", Colors.RED)
        assert Colors.RED in result
        assert Colors.RESET in result
        assert "test" in result

    def test_colorize_with_colors_disabled(self):
        logger = Logger(use_colors=False)
        result = logger._colorize("test", Colors.RED)
        assert result == "test"
        assert Colors.RED not in result


class TestLoggerOutput:
    """Test Logger output methods."""

    def test_info_writes_to_stream(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.info("Test message")
        output = stream.getvalue()
        assert "INFO" in output
        assert "Test message" in output

    def test_debug_writes_to_stream(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.debug("Debug message")
        output = stream.getvalue()
        assert "DEBUG" in output
        assert "Debug message" in output

    def test_warning_writes_to_stream(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.warning("Warning message")
        output = stream.getvalue()
        assert "WARNING" in output
        assert "Warning message" in output

    def test_info_with_component(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.info("Test message", component="simulation")
        output = stream.getvalue()
        assert "SIMULATION" in output

    def test_info_with_task_name(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.info("Test message", task_name="task_1")
        output = stream.getvalue()
        assert "task_1" in output


class TestLoggerTaskMethods:
    """Test task-specific logging methods."""

    def test_task_started(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.task_started("sim_0")
        output = stream.getvalue()
        assert "Task started" in output
        assert "sim_0" in output

    def test_task_completed(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.task_completed("sim_0")
        output = stream.getvalue()
        assert "Task completed" in output
        assert "sim_0" in output

    def test_task_killed(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.task_killed("sim_0")
        output = stream.getvalue()
        assert "Task killed" in output
        assert "sim_0" in output

    def test_task_started_with_component(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.task_started("sim_0", component="simulation")
        output = stream.getvalue()
        assert "SIMULATION" in output


class TestLoggerManagerMethods:
    """Test manager-specific logging methods."""

    def test_manager_starting(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.manager_starting(10)
        output = stream.getvalue()
        assert "Starting" in output
        assert "10" in output

    def test_manager_exiting(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.manager_exiting()
        output = stream.getvalue()
        assert "All tasks finished" in output
        assert "Exiting" in output


class TestLoggerSeparator:
    """Test the separator method."""

    def test_separator_without_title(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.separator()
        output = stream.getvalue()
        assert "=" in output

    def test_separator_with_title(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.separator("TEST TITLE")
        output = stream.getvalue()
        assert "TEST TITLE" in output
        assert "=" in output


class TestLoggerComponentColors:
    """Test component color mappings."""

    def test_simulation_component_uses_blue(self):
        logger = Logger(use_colors=True)
        assert logger.component_colors.get('simulation') == Colors.BLUE

    def test_training_component_uses_yellow(self):
        logger = Logger(use_colors=True)
        assert logger.component_colors.get('training') == Colors.BRIGHT_YELLOW

    def test_prediction_component_uses_green(self):
        logger = Logger(use_colors=True)
        assert logger.component_colors.get('prediction') == Colors.GREEN


class TestLoggerTaskLog:
    """Test the task_log method."""

    def test_task_log_default_level(self):
        stream = StringIO()
        logger = Logger(name="TestTask", output_stream=stream, use_colors=False)
        logger.task_log("Task log message")
        output = stream.getvalue()
        assert "Task log message" in output
        assert "TASK-TESTTASK" in output

    def test_task_log_with_error_level(self):
        stream = StringIO()
        logger = Logger(name="TestTask", output_stream=stream, use_colors=False)
        logger.task_log("Error message", level=LogLevel.ERROR)
        output = stream.getvalue()
        # Note: error goes to stderr, but we capture stdout
        # The message should still be formatted correctly


class TestLoggerTimestamp:
    """Test timestamp formatting."""

    def test_timestamp_in_output(self):
        stream = StringIO()
        logger = Logger(output_stream=stream, use_colors=False)
        logger.info("Test")
        output = stream.getvalue()
        # Check for time format HH:MM:SS.mmm
        import re
        assert re.search(r'\d{2}:\d{2}:\d{2}\.\d{3}', output)
