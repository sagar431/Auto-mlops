#!/usr/bin/env python3
"""
Unit tests for action/execute_step.py - MCP tool execution module.

Run with: pytest tests/unit/test_execute_step.py -v
"""

import asyncio
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from action.execute_step import (
    AVAILABLE_TOOLS,
    _format_args,
    _get_tool_function,
    _get_tool_params,
    execute_step,
    get_available_tools,
)

# ============================================================================
# execute_step Tests
# ============================================================================


class TestExecuteStep:
    """Tests for execute_step async function."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock context manager."""
        ctx = MagicMock()
        ctx.project_path = "/test/project"
        return ctx

    @pytest.fixture
    def mock_tools_module(self):
        """Create a mock tools module with sample tools."""
        module = ModuleType("mock_tools")

        # Add a sync tool
        def sync_tool(project_path: str, config_name: str = "default") -> dict:
            return {"success": True, "message": f"Created config at {project_path}"}

        # Add an async tool
        async def async_tool(project_path: str) -> dict:
            return {"success": True, "analyzed": True}

        # Add a tool that returns failure
        def failing_tool(project_path: str) -> dict:
            return {"success": False, "error": "Tool failed intentionally"}

        # Add a tool that raises exception
        def exception_tool(project_path: str) -> dict:
            raise ValueError("Something went wrong")

        # Add a tool that returns non-dict
        def non_dict_tool(project_path: str) -> str:
            return "Plain string result"

        module.sync_tool = sync_tool
        module.async_tool = async_tool
        module.failing_tool = failing_tool
        module.exception_tool = exception_tool
        module.non_dict_tool = non_dict_tool

        return module

    @pytest.mark.asyncio
    async def test_execute_sync_tool_success(self, mock_context, mock_tools_module):
        """Test executing a synchronous tool successfully."""
        success, result = await execute_step(
            step_id="step-1",
            tool="sync_tool",
            args={"config_name": "test_config"},
            ctx=mock_context,
            tools_module=mock_tools_module,
        )

        assert success is True
        assert result["success"] is True
        assert result["step_id"] == "step-1"
        assert result["tool"] == "sync_tool"
        assert "timestamp" in result
        assert "duration" in result
        assert result["result"]["message"] == "Created config at /test/project"

    @pytest.mark.asyncio
    async def test_execute_async_tool_success(self, mock_context, mock_tools_module):
        """Test executing an asynchronous tool successfully."""
        success, result = await execute_step(
            step_id="step-2",
            tool="async_tool",
            args={},
            ctx=mock_context,
            tools_module=mock_tools_module,
        )

        assert success is True
        assert result["success"] is True
        assert result["step_id"] == "step-2"
        assert result["tool"] == "async_tool"
        assert result["result"]["analyzed"] is True

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mock_context, mock_tools_module):
        """Test executing a non-existent tool returns error."""
        success, result = await execute_step(
            step_id="step-3",
            tool="nonexistent_tool",
            args={},
            ctx=mock_context,
            tools_module=mock_tools_module,
        )

        assert success is False
        assert "error" in result
        assert "not found" in result["error"]
        assert result["step_id"] == "step-3"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_execute_tool_returns_failure(self, mock_context, mock_tools_module):
        """Test handling tool that returns success=False."""
        success, result = await execute_step(
            step_id="step-4",
            tool="failing_tool",
            args={},
            ctx=mock_context,
            tools_module=mock_tools_module,
        )

        assert success is False
        assert "error" in result
        assert result["step_id"] == "step-4"
        assert result["tool"] == "failing_tool"
        assert "Tool failed intentionally" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_raises_exception(self, mock_context, mock_tools_module):
        """Test handling tool that raises an exception."""
        success, result = await execute_step(
            step_id="step-5",
            tool="exception_tool",
            args={},
            ctx=mock_context,
            tools_module=mock_tools_module,
        )

        assert success is False
        assert "error" in result
        assert "ValueError" in result["error"]
        assert "Something went wrong" in result["error"]
        assert result["step_id"] == "step-5"
        assert "traceback" in result

    @pytest.mark.asyncio
    async def test_execute_tool_returns_non_dict(self, mock_context, mock_tools_module):
        """Test handling tool that returns non-dict result."""
        success, result = await execute_step(
            step_id="step-6",
            tool="non_dict_tool",
            args={},
            ctx=mock_context,
            tools_module=mock_tools_module,
        )

        assert success is True
        assert result["success"] is True
        assert result["step_id"] == "step-6"
        assert result["result"]["output"] == "Plain string result"

    @pytest.mark.asyncio
    async def test_execute_tool_injects_project_path(self, mock_context, mock_tools_module):
        """Test that project_path is injected from context when not provided."""
        success, result = await execute_step(
            step_id="step-7",
            tool="sync_tool",
            args={"config_name": "injected_test"},
            ctx=mock_context,
            tools_module=mock_tools_module,
        )

        assert success is True
        # project_path should be from context
        assert result["result"]["message"] == "Created config at /test/project"

    @pytest.mark.asyncio
    async def test_execute_tool_does_not_override_provided_project_path(
        self, mock_context, mock_tools_module
    ):
        """Test that provided project_path is not overridden."""
        success, result = await execute_step(
            step_id="step-8",
            tool="sync_tool",
            args={"project_path": "/custom/path", "config_name": "custom"},
            ctx=mock_context,
            tools_module=mock_tools_module,
        )

        assert success is True
        # Should use provided path, not context path
        assert result["result"]["message"] == "Created config at /custom/path"

    @pytest.mark.asyncio
    async def test_execute_tool_handles_type_error(self, mock_context):
        """Test handling TypeError from wrong arguments."""
        module = ModuleType("mock_tools")

        def strict_tool(required_arg: str) -> dict:
            return {"success": True}

        module.strict_tool = strict_tool

        # Don't provide required_arg
        success, result = await execute_step(
            step_id="step-9",
            tool="strict_tool",
            args={},
            ctx=mock_context,
            tools_module=module,
        )

        assert success is False
        assert "error" in result
        assert "Invalid arguments" in result["error"]
        assert result["step_id"] == "step-9"

    @pytest.mark.asyncio
    async def test_execute_tool_with_no_project_path_in_context(self, mock_tools_module):
        """Test executing when context has no project_path."""
        ctx = MagicMock()
        ctx.project_path = None

        success, result = await execute_step(
            step_id="step-10",
            tool="sync_tool",
            args={"project_path": "/explicit/path", "config_name": "test"},
            ctx=ctx,
            tools_module=mock_tools_module,
        )

        assert success is True
        assert result["result"]["message"] == "Created config at /explicit/path"

    @pytest.mark.asyncio
    async def test_execute_tool_result_has_duration(self, mock_context, mock_tools_module):
        """Test that successful execution includes duration."""
        success, result = await execute_step(
            step_id="step-11",
            tool="sync_tool",
            args={"config_name": "test"},
            ctx=mock_context,
            tools_module=mock_tools_module,
        )

        assert success is True
        assert "duration" in result
        assert isinstance(result["duration"], float)
        assert result["duration"] >= 0


# ============================================================================
# _get_tool_function Tests
# ============================================================================


class TestGetToolFunction:
    """Tests for _get_tool_function helper."""

    @pytest.fixture
    def mock_tools_module(self):
        """Create a mock tools module."""
        module = ModuleType("mock_tools")

        def sample_tool():
            pass

        module.sample_tool = sample_tool
        return module

    def test_get_tool_from_provided_module(self, mock_tools_module):
        """Test getting tool from provided module."""
        func = _get_tool_function("sample_tool", mock_tools_module)
        assert func is not None
        assert func.__name__ == "sample_tool"

    def test_get_tool_not_in_module(self, mock_tools_module):
        """Test getting non-existent tool returns None."""
        func = _get_tool_function("nonexistent", mock_tools_module)
        assert func is None

    def test_get_tool_with_none_module(self):
        """Test getting tool with None module tries mcp_mlops_tools."""
        # This will try to import mcp_mlops_tools, which may or may not exist
        # In tests, we just verify it doesn't crash
        func = _get_tool_function("nonexistent_tool_12345", None)
        assert func is None

    def test_get_tool_from_mcp_mlops_tools(self):
        """Test getting tool from mcp_mlops_tools module."""
        # Patch the import to test the fallback path
        mock_mcp_module = ModuleType("mcp_mlops_tools")

        def create_hydra_config():
            pass

        mock_mcp_module.create_hydra_config = create_hydra_config

        with patch.dict("sys.modules", {"mcp_mlops_tools": mock_mcp_module}):
            func = _get_tool_function("create_hydra_config", None)
            assert func is not None


# ============================================================================
# _get_tool_params Tests
# ============================================================================


class TestGetToolParams:
    """Tests for _get_tool_params helper."""

    def test_get_params_from_function(self):
        """Test getting parameter names from function."""

        def sample_func(arg1: str, arg2: int, arg3: float = 1.0):
            pass

        params = _get_tool_params(sample_func)
        assert params == {"arg1", "arg2", "arg3"}

    def test_get_params_from_function_no_args(self):
        """Test getting params from function with no arguments."""

        def no_args_func():
            pass

        params = _get_tool_params(no_args_func)
        assert params == set()

    def test_get_params_from_lambda(self):
        """Test getting params from lambda function."""
        func = lambda x, y: x + y  # noqa: E731
        params = _get_tool_params(func)
        assert params == {"x", "y"}

    def test_get_params_handles_invalid_callable(self):
        """Test getting params from non-callable returns empty set."""

        class NoSignature:
            """Object that will raise error on signature inspection."""

            pass

        obj = NoSignature()
        params = _get_tool_params(obj)
        assert params == set()

    def test_get_params_with_kwargs(self):
        """Test getting params from function with **kwargs."""

        def func_with_kwargs(arg1: str, **kwargs):
            pass

        params = _get_tool_params(func_with_kwargs)
        assert "arg1" in params
        assert "kwargs" in params


# ============================================================================
# _format_args Tests
# ============================================================================


class TestFormatArgs:
    """Tests for _format_args helper."""

    def test_format_simple_args(self):
        """Test formatting simple arguments."""
        args = {"name": "test", "value": 123}
        result = _format_args(args)
        assert "name=test" in result
        assert "value=123" in result

    def test_format_truncates_long_values(self):
        """Test that long values are truncated."""
        args = {"long_value": "x" * 100}
        result = _format_args(args)
        assert "..." in result
        assert len(result) < 100

    def test_format_empty_args(self):
        """Test formatting empty arguments."""
        args = {}
        result = _format_args(args)
        assert result == ""

    def test_format_args_with_special_characters(self):
        """Test formatting args with special characters."""
        args = {"path": "/home/user/project", "config": "model.yaml"}
        result = _format_args(args)
        assert "path=/home/user/project" in result
        assert "config=model.yaml" in result

    def test_format_args_with_none_value(self):
        """Test formatting args with None value."""
        args = {"value": None}
        result = _format_args(args)
        assert "value=None" in result

    def test_format_args_with_dict_value(self):
        """Test formatting args with dict value."""
        args = {"config": {"key": "value"}}
        result = _format_args(args)
        assert "config=" in result

    def test_format_args_truncation_at_50_chars(self):
        """Test truncation happens at exactly 50 characters."""
        args = {"value": "a" * 60}
        result = _format_args(args)
        # Should have 50 chars plus "..."
        assert result == "value=" + "a" * 50 + "..."


# ============================================================================
# get_available_tools Tests
# ============================================================================


class TestGetAvailableTools:
    """Tests for get_available_tools function."""

    def test_returns_list(self):
        """Test that get_available_tools returns a list."""
        tools = get_available_tools()
        assert isinstance(tools, list)

    def test_returns_copy(self):
        """Test that get_available_tools returns a copy, not the original."""
        tools1 = get_available_tools()
        tools2 = get_available_tools()
        assert tools1 is not tools2
        assert tools1 == tools2

    def test_modifying_returned_list_does_not_affect_original(self):
        """Test that modifying returned list doesn't affect AVAILABLE_TOOLS."""
        tools = get_available_tools()
        original_len = len(tools)
        tools.append("new_tool")
        assert len(get_available_tools()) == original_len


# ============================================================================
# AVAILABLE_TOOLS Tests
# ============================================================================


class TestAvailableTools:
    """Tests for AVAILABLE_TOOLS constant."""

    def test_contains_hydra_tools(self):
        """Test AVAILABLE_TOOLS contains Hydra tools."""
        assert "analyze_project_config" in AVAILABLE_TOOLS
        assert "create_hydra_config" in AVAILABLE_TOOLS
        assert "update_hydra_config" in AVAILABLE_TOOLS
        assert "validate_hydra_config" in AVAILABLE_TOOLS

    def test_contains_mlflow_tools(self):
        """Test AVAILABLE_TOOLS contains MLflow tools."""
        assert "init_mlflow_experiment" in AVAILABLE_TOOLS
        assert "start_mlflow_run" in AVAILABLE_TOOLS
        assert "log_mlflow_params" in AVAILABLE_TOOLS
        assert "log_mlflow_metrics" in AVAILABLE_TOOLS
        assert "log_mlflow_artifact" in AVAILABLE_TOOLS
        assert "register_mlflow_model" in AVAILABLE_TOOLS
        assert "get_best_mlflow_run" in AVAILABLE_TOOLS
        assert "end_mlflow_run" in AVAILABLE_TOOLS

    def test_contains_dvc_tools(self):
        """Test AVAILABLE_TOOLS contains DVC tools."""
        assert "init_dvc_repo" in AVAILABLE_TOOLS
        assert "configure_dvc_remote" in AVAILABLE_TOOLS
        assert "add_data_to_dvc" in AVAILABLE_TOOLS
        assert "create_dvc_pipeline" in AVAILABLE_TOOLS
        assert "dvc_push" in AVAILABLE_TOOLS
        assert "dvc_pull" in AVAILABLE_TOOLS
        assert "dvc_reproduce" in AVAILABLE_TOOLS
        assert "configure_validate_capstone_dvc_remote" in AVAILABLE_TOOLS
        assert "push_capstone_data" in AVAILABLE_TOOLS
        assert "pull_capstone_data" in AVAILABLE_TOOLS

    def test_contains_docker_tools(self):
        """Test AVAILABLE_TOOLS contains Docker tools."""
        assert "create_ml_dockerfile" in AVAILABLE_TOOLS
        assert "build_ml_docker_image" in AVAILABLE_TOOLS
        assert "run_training_container" in AVAILABLE_TOOLS
        assert "push_docker_image" in AVAILABLE_TOOLS

    def test_contains_github_tools(self):
        """Test AVAILABLE_TOOLS contains GitHub Actions tools."""
        assert "create_github_workflow" in AVAILABLE_TOOLS
        assert "add_workflow_step" in AVAILABLE_TOOLS

    def test_contains_training_tools(self):
        """Test AVAILABLE_TOOLS contains training control tools."""
        assert "detect_training_project" in AVAILABLE_TOOLS
        assert "detect_capstone_data_layouts" in AVAILABLE_TOOLS
        assert "run_bounded_training" in AVAILABLE_TOOLS
        assert "track_training_in_mlflow" in AVAILABLE_TOOLS
        assert "select_best_model_artifact" in AVAILABLE_TOOLS
        assert "record_capstone_data_stage_evidence" in AVAILABLE_TOOLS
        assert "record_capstone_orchestrator_skeleton" in AVAILABLE_TOOLS
        assert "resolve_capstone_container_upstream_evidence" in AVAILABLE_TOOLS
        assert "generate_validate_capstone_runtime_image_spec" in AVAILABLE_TOOLS
        assert "analyze_training_results" in AVAILABLE_TOOLS
        assert "suggest_improvements" in AVAILABLE_TOOLS
        assert "check_accuracy_threshold" in AVAILABLE_TOOLS

    def test_contains_deployment_tools(self):
        """Test AVAILABLE_TOOLS contains deployment and rollback tools."""
        assert "create_gradio_interface" in AVAILABLE_TOOLS
        assert "create_inference_service_yaml" in AVAILABLE_TOOLS
        assert "rollback_deployment" in AVAILABLE_TOOLS
        assert "create_helm_chart" in AVAILABLE_TOOLS

    def test_tool_count(self):
        """Test AVAILABLE_TOOLS has expected number of tools."""
        # Core tools plus deployment/K8s/AWS helpers should be present
        assert len(AVAILABLE_TOOLS) >= 28


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestExecuteStepIntegration:
    """Integration-style tests for execute_step."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        ctx = MagicMock()
        ctx.project_path = "/test/project"
        return ctx

    @pytest.mark.asyncio
    async def test_execute_step_with_complex_args(self, mock_context):
        """Test executing step with complex nested arguments."""
        module = ModuleType("mock_tools")

        def complex_tool(project_path: str, config: dict, options: list) -> dict:
            return {
                "success": True,
                "project_path": project_path,
                "config_keys": list(config.keys()),
                "option_count": len(options),
            }

        module.complex_tool = complex_tool

        success, result = await execute_step(
            step_id="complex-1",
            tool="complex_tool",
            args={
                "config": {"model": {"type": "mlp", "layers": [64, 32]}},
                "options": ["opt1", "opt2", "opt3"],
            },
            ctx=mock_context,
            tools_module=module,
        )

        assert success is True
        assert result["result"]["config_keys"] == ["model"]
        assert result["result"]["option_count"] == 3

    @pytest.mark.asyncio
    async def test_execute_step_async_with_delay(self, mock_context):
        """Test executing async tool with simulated delay."""
        module = ModuleType("mock_tools")

        async def slow_tool(project_path: str) -> dict:
            await asyncio.sleep(0.01)  # Small delay
            return {"success": True, "delayed": True}

        module.slow_tool = slow_tool

        success, result = await execute_step(
            step_id="slow-1",
            tool="slow_tool",
            args={},
            ctx=mock_context,
            tools_module=module,
        )

        assert success is True
        assert result["result"]["delayed"] is True
        assert result["duration"] >= 0.01

    @pytest.mark.asyncio
    async def test_execute_step_preserves_result_metadata(self, mock_context):
        """Test that tool result metadata is preserved."""
        module = ModuleType("mock_tools")

        def metadata_tool(project_path: str) -> dict:
            return {
                "success": True,
                "experiment_id": "exp-123",
                "run_id": "run-456",
                "metrics": {"accuracy": 0.95, "loss": 0.05},
            }

        module.metadata_tool = metadata_tool

        success, result = await execute_step(
            step_id="meta-1",
            tool="metadata_tool",
            args={},
            ctx=mock_context,
            tools_module=module,
        )

        assert success is True
        assert result["result"]["experiment_id"] == "exp-123"
        assert result["result"]["run_id"] == "run-456"
        assert result["result"]["metrics"]["accuracy"] == 0.95
