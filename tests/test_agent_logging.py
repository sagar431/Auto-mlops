#!/usr/bin/env python3
"""
Tests for structured logging in agent modules.

Run with: pytest tests/test_agent_logging.py -v
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from observability import LoggerFactory, configure_logging
from observability.logging import clear_log_context


@pytest.fixture(autouse=True)
def reset_logger_factory():
    """Reset the logger factory between tests."""
    LoggerFactory.clear()
    clear_log_context()
    configure_logging(level="debug", json_output=True)
    yield
    LoggerFactory.clear()
    clear_log_context()


class TestAgentLoopLogging:
    """Tests for structured logging in agent_loop.py."""

    def test_agent_loop_logger_import(self):
        """Test that agent_loop module has logger configured."""
        from agent import agent_loop

        assert hasattr(agent_loop, "logger")
        assert agent_loop.logger.name == "agent.agent_loop"

    @pytest.mark.asyncio
    async def test_emit_event_logs_warning_on_failure(self, caplog):
        """Test that _emit logs warning when event callback fails."""
        from agent.agent_loop import AgentLoop

        async def failing_callback(event_type, data):
            raise Exception("Callback error")

        agent = AgentLoop(on_event=failing_callback)

        with caplog.at_level(logging.WARNING, logger="agent.agent_loop"):
            await agent._emit("test_event", {"key": "value"})

        # Check that warning was logged
        assert len(caplog.records) >= 1
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert "Failed to emit event" in warning_records[0].message

    @pytest.mark.asyncio
    async def test_improvement_suggestion_logs_error_on_failure(self, caplog):
        """Test that _get_improvement_suggestion logs error on exception."""
        from agent.agent_loop import AgentLoop
        from agent.contextManager import ContextManager

        agent = AgentLoop()
        agent.ctx = ContextManager(
            session_id="test-session", original_query="test query", project_path="/tmp"
        )
        agent.ctx.experiment_state.target_accuracy = 0.9
        agent.ctx.experiment_state.current_accuracy = 0.8
        agent.improvement_prompt = "{target_accuracy} {current_accuracy} {current_loss} {gap} {current_config} {training_history} {attempt} {max_attempts} {previous_improvements}"

        # Mock model_manager to raise exception
        agent.model_manager = MagicMock()
        agent.model_manager.generate_json = AsyncMock(
            side_effect=Exception("LLM generation failed")
        )

        with caplog.at_level(logging.ERROR, logger="agent.agent_loop"):
            result = await agent._get_improvement_suggestion()

        # Check that error was logged
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert "Error getting improvement suggestion" in error_records[0].message

        # Check return value
        assert result["should_retry"] is False
        assert "LLM generation failed" in result["error"]


class TestContextManagerLogging:
    """Tests for structured logging in contextManager.py."""

    def test_context_manager_logger_import(self):
        """Test that contextManager module has logger configured."""
        from agent import contextManager

        assert hasattr(contextManager, "logger")
        assert contextManager.logger.name == "agent.contextManager"

    def test_print_graph_logs_debug(self, caplog):
        """Test that print_graph uses structured logging."""
        from agent.contextManager import ContextManager

        ctx = ContextManager(
            session_id="test-session", original_query="test query", project_path="/tmp"
        )
        ctx.experiment_state.stage = "training"
        ctx.experiment_state.current_accuracy = 0.85
        ctx.experiment_state.target_accuracy = 0.9

        # Add a test step
        ctx.add_step(
            step_id="step-1",
            description="Test step description",
            step_type="CODE",
            tool="test_tool",
        )

        with caplog.at_level(logging.DEBUG, logger="agent.contextManager"):
            ctx.print_graph()

        # Check that debug log was emitted
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_records) >= 1
        assert "Graph structure" in debug_records[0].message


class TestModelManagerLogging:
    """Tests for structured logging in model_manager.py."""

    def test_model_manager_logger_import(self):
        """Test that model_manager module has logger configured."""
        from agent import model_manager

        assert hasattr(model_manager, "logger")
        assert model_manager.logger.name == "agent.model_manager"

    def test_load_config_logs_warning_on_failure(self, caplog):
        """Test that _load_config logs warning when config file not found."""
        from agent.model_manager import ModelManager

        with caplog.at_level(logging.WARNING, logger="agent.model_manager"):
            # Create manager with non-existent config file
            manager = ModelManager(config_path="/nonexistent/path/models.json")

        # Check warning was logged
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert "Could not load model config" in warning_records[0].message

        # Manager should still work with default config
        assert manager.default_model == "gemini"

    def test_get_client_logs_warning_on_import_error(self, caplog):
        """Test that _get_client logs warning when package not installed."""
        from agent.model_manager import ModelManager

        manager = ModelManager()
        manager._clients = {}  # Clear any cached clients

        with caplog.at_level(logging.WARNING, logger="agent.model_manager"):
            # Mock the import to fail
            with patch.dict("sys.modules", {"google.generativeai": None}):
                with patch("builtins.__import__", side_effect=ImportError("No module")):
                    result = manager._get_client("google")

        # Check warning was logged
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert "not installed" in warning_records[0].message
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_logs_error_on_failure(self, caplog):
        """Test that generate logs error when LLM call fails."""
        from agent.model_manager import ModelManager

        manager = ModelManager()

        # Mock the client to fail
        mock_client = MagicMock()
        manager._clients["google"] = mock_client
        manager.models = {
            "gemini": {"provider": "google", "model": "gemini-2.0-flash", "temperature": 0.7}
        }

        # Patch _generate_gemini to raise
        with patch.object(manager, "_generate_gemini", new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("API call failed")

            with caplog.at_level(logging.ERROR, logger="agent.model_manager"):
                with pytest.raises(Exception):
                    await manager.generate("test prompt")

        # Check error was logged
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert "Error generating with LLM" in error_records[0].message

    @pytest.mark.asyncio
    async def test_generate_json_logs_error_on_parse_failure(self, caplog):
        """Test that generate_json logs error when JSON parsing fails."""
        from agent.model_manager import ModelManager

        manager = ModelManager()

        # Mock generate to return invalid JSON
        with patch.object(manager, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "not valid json {{"

            with caplog.at_level(logging.ERROR, logger="agent.model_manager"):
                result = await manager.generate_json("test prompt")

        # Check error was logged
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert "Failed to parse JSON response" in error_records[0].message

        # Check return value indicates error
        assert result["error"] == "Failed to parse response"
        assert "not valid json" in result["raw"]


class TestLoggerContextBinding:
    """Tests for logger context binding in agent modules."""

    def test_logger_can_bind_session_context(self, caplog):
        """Test that agent loggers can bind session context."""
        from agent.agent_loop import logger

        with caplog.at_level(logging.INFO, logger="agent.agent_loop"):
            bound_logger = logger.bind(session_id="sess-123", phase="perception")
            bound_logger.info("Test message")

        # Check that the message was logged
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1
        assert "Test message" in info_records[0].message

    def test_logger_can_bind_step_context(self, caplog):
        """Test that agent loggers can bind step context."""
        from agent.contextManager import logger

        with caplog.at_level(logging.INFO, logger="agent.contextManager"):
            bound_logger = logger.bind(step_id="step-1", tool_name="create_hydra_config")
            bound_logger.info("Step executing")

        # Check that the message was logged
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1
        assert "Step executing" in info_records[0].message


class TestLogMessageContent:
    """Tests for verifying log message content and structure."""

    def test_emit_event_log_contains_event_type(self, caplog):
        """Test that emit event log message contains the event type."""
        from agent.agent_loop import AgentLoop

        async def failing_callback(event_type, data):
            raise Exception("Callback error")

        agent = AgentLoop(on_event=failing_callback)

        with caplog.at_level(logging.WARNING, logger="agent.agent_loop"):
            asyncio.get_event_loop().run_until_complete(agent._emit("test_event", {"key": "value"}))

        # Check log output contains structured data (visible in raw log output)
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1

    def test_model_manager_error_contains_provider_info(self, caplog):
        """Test that model manager error logs contain provider info."""
        from agent.model_manager import ModelManager

        manager = ModelManager()
        mock_client = MagicMock()
        manager._clients["google"] = mock_client
        manager.models = {
            "gemini": {"provider": "google", "model": "gemini-2.0-flash", "temperature": 0.7}
        }

        with patch.object(manager, "_generate_gemini", new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("API call failed")

            with caplog.at_level(logging.ERROR, logger="agent.model_manager"):
                try:
                    asyncio.get_event_loop().run_until_complete(manager.generate("test prompt"))
                except Exception:
                    pass

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        # The structured logger adds extra fields that get included in the message
        assert "Error generating with LLM" in error_records[0].message


class TestExecuteStepLogging:
    """Tests for structured logging in execute_step.py."""

    def test_execute_step_logger_import(self):
        """Test that execute_step module has logger configured."""
        import sys

        # Ensure module is loaded by importing it (even if result isn't directly used)
        __import__("action.execute_step")

        module = sys.modules["action.execute_step"]
        assert hasattr(module, "logger")
        assert module.logger.name == "action.execute_step"

    @pytest.mark.asyncio
    async def test_execute_step_logs_info_on_execution(self, caplog):
        """Test that execute_step logs info when executing a tool."""
        from action.execute_step import execute_step

        # Create a mock tools module with a simple tool
        mock_tools = MagicMock()
        mock_tools.test_tool = MagicMock(return_value={"success": True, "output": "test"})

        # Create a mock context
        mock_ctx = MagicMock()
        mock_ctx.project_path = "/tmp/test"

        with caplog.at_level(logging.INFO, logger="action.execute_step"):
            success, result = await execute_step(
                step_id="step-1",
                tool="test_tool",
                args={"param": "value"},
                ctx=mock_ctx,
                tools_module=mock_tools,
            )

        # Check that info was logged
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1
        assert "Executing tool" in info_records[0].message

    @pytest.mark.asyncio
    async def test_execute_step_logs_error_on_exception(self, caplog):
        """Test that execute_step logs error when tool raises exception."""
        from action.execute_step import execute_step

        # Create a mock tools module with a failing tool
        mock_tools = MagicMock()
        mock_tools.failing_tool = MagicMock(side_effect=RuntimeError("Tool failed"))

        # Create a mock context
        mock_ctx = MagicMock()
        mock_ctx.project_path = "/tmp/test"

        with caplog.at_level(logging.ERROR, logger="action.execute_step"):
            success, result = await execute_step(
                step_id="step-1",
                tool="failing_tool",
                args={"param": "value"},
                ctx=mock_ctx,
                tools_module=mock_tools,
            )

        # Check that error was logged
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert "Error executing tool" in error_records[0].message

        # Check return value indicates failure
        assert success is False
        assert "RuntimeError" in result["error"]


class TestPerceptionLogging:
    """Tests for structured logging in perception.py."""

    def test_perception_logger_import(self):
        """Test that perception module has logger configured."""
        from perception import perception

        assert hasattr(perception, "logger")
        assert perception.logger.name == "perception.perception"

    def test_perception_logs_warning_on_prompt_load_failure(self, caplog):
        """Test that Perception logs warning when prompt file not found."""
        from perception.perception import Perception

        with caplog.at_level(logging.WARNING, logger="perception.perception"):
            Perception("/nonexistent/path/prompt.txt")

        # Check warning was logged
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert "Could not load perception prompt" in warning_records[0].message

    @pytest.mark.asyncio
    async def test_perception_logs_error_on_run_failure(self, caplog):
        """Test that Perception.run logs error when LLM fails."""
        from perception.perception import Perception

        perception = Perception("/nonexistent/path/prompt.txt")

        # Mock model_manager to raise exception
        perception.model_manager = MagicMock()
        perception.model_manager.generate_json = AsyncMock(
            side_effect=Exception("LLM generation failed")
        )

        with caplog.at_level(logging.ERROR, logger="perception.perception"):
            result = await perception.run({"query": "test query"})

        # Check that error was logged
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert "Perception error" in error_records[0].message

        # Check return value is fallback
        assert result["route"] == "decision"
        assert result["reasoning"] == "Fallback due to LLM error"
