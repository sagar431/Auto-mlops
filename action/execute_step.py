"""
Step Execution Module for MLOps Agent.
Executes MCP tool calls for ML pipeline operations.
"""

import asyncio
import traceback
from collections.abc import Callable
from datetime import datetime
from typing import Any

from observability import get_logger

logger = get_logger("action.execute_step")


async def execute_step(
    step_id: str, tool: str, args: dict[str, Any], ctx: Any, tools_module: Any = None
) -> tuple[bool, dict[str, Any]]:
    """
    Execute a single step by calling the specified MCP tool.

    Args:
        step_id: Unique identifier for this step
        tool: Name of the MCP tool to call
        args: Arguments to pass to the tool
        ctx: Context manager for state tracking
        tools_module: Module containing tool functions (mcp_mlops_tools)

    Returns:
        Tuple of (success: bool, result: Dict)
    """
    start_time = datetime.utcnow()

    try:
        # Get the tool function
        tool_func = _get_tool_function(tool, tools_module)

        if tool_func is None:
            return False, {
                "error": f"Tool '{tool}' not found",
                "step_id": step_id,
                "timestamp": start_time.isoformat(),
            }

        # Inject project_path from context if not provided
        if ctx.project_path and "project_path" in _get_tool_params(tool_func):
            args.setdefault("project_path", ctx.project_path)

        # Execute the tool
        logger.info("Executing tool", tool=tool, args=_format_args(args), step_id=step_id)

        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**args)
        else:
            result = tool_func(**args)

        # Handle result
        if isinstance(result, dict):
            success = result.get("success", True)
            if not success:
                error_msg = result.get("error", "Tool returned success=False")
                return False, {
                    "error": error_msg,
                    "result": result,
                    "step_id": step_id,
                    "tool": tool,
                    "timestamp": start_time.isoformat(),
                }

            return True, {
                "success": True,
                "result": result,
                "step_id": step_id,
                "tool": tool,
                "timestamp": start_time.isoformat(),
                "duration": (datetime.utcnow() - start_time).total_seconds(),
            }
        else:
            # Non-dict result (shouldn't happen with our tools)
            return True, {
                "success": True,
                "result": {"output": str(result)},
                "step_id": step_id,
                "tool": tool,
                "timestamp": start_time.isoformat(),
            }

    except TypeError as e:
        # Common error: wrong arguments
        return False, {
            "error": f"Invalid arguments for tool '{tool}': {str(e)}",
            "step_id": step_id,
            "tool": tool,
            "args": args,
            "timestamp": start_time.isoformat(),
        }

    except Exception as e:
        logger.error("Error executing tool", tool=tool, error=str(e), step_id=step_id)
        return False, {
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
            "step_id": step_id,
            "tool": tool,
            "timestamp": start_time.isoformat(),
        }


def _get_tool_function(tool_name: str, tools_module: Any) -> Callable | None:
    """Get tool function by name from the tools module."""
    # Try to get from provided module
    if tools_module is not None:
        if hasattr(tools_module, tool_name):
            return getattr(tools_module, tool_name)

    # Try to import from mcp_mlops_tools
    try:
        import mcp_mlops_tools

        if hasattr(mcp_mlops_tools, tool_name):
            return getattr(mcp_mlops_tools, tool_name)
    except ImportError:
        pass

    return None


def _get_tool_params(func: callable) -> set:
    """Get parameter names for a function."""
    import inspect

    try:
        sig = inspect.signature(func)
        return set(sig.parameters.keys())
    except (ValueError, TypeError):
        return set()


def _format_args(args: dict) -> str:
    """Format args for logging (truncate long values)."""
    formatted = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 50:
            v_str = v_str[:50] + "..."
        formatted.append(f"{k}={v_str}")
    return ", ".join(formatted)


# Tool registry for available MLOps tools
AVAILABLE_TOOLS = [
    # Hydra Configuration
    "analyze_project_config",
    "create_hydra_config",
    "update_hydra_config",
    "validate_hydra_config",
    # MLflow Tracking
    "init_mlflow_experiment",
    "start_mlflow_run",
    "log_mlflow_params",
    "log_mlflow_metrics",
    "log_mlflow_artifact",
    "register_mlflow_model",
    "get_best_mlflow_run",
    "end_mlflow_run",
    # DVC Data Versioning
    "init_dvc_repo",
    "configure_dvc_remote",
    "add_data_to_dvc",
    "create_dvc_pipeline",
    "dvc_push",
    "dvc_pull",
    "dvc_reproduce",
    # Docker
    "create_ml_dockerfile",
    "build_ml_docker_image",
    "run_training_container",
    "push_docker_image",
    # GitHub Actions
    "create_github_workflow",
    "add_workflow_step",
    # Training Control
    "detect_training_project",
    "run_bounded_training",
    "analyze_training_results",
    "suggest_improvements",
    "check_accuracy_threshold",
    # Deployment (Phase 4)
    "detect_runtime_environment",
    "detect_gpu_cuda",
    "select_best_model_artifact",
    "create_litserve_api",
    "record_litserve_image_build_skipped",
    "start_litserve_server",
    "test_litserve_health_endpoint",
    "test_litserve_prediction_endpoint",
    "capture_litserve_logs_and_endpoint",
    "record_litserve_gpu_rollback_readiness",
    "configure_litserver",
    "create_gradio_interface",
    "deploy_to_huggingface",
    "create_fastapi_app",
    "create_lambda_dockerfile",
    "generate_cdk_stack",
    "create_torchserve_handler",
    "create_mar_archive",
    "generate_torchserve_config",
    "create_inference_service_yaml",
    "generate_kserve_config",
    # Kubernetes manifests
    "create_k8s_deployment_yaml",
    "create_k8s_service_yaml",
    "create_k8s_ingress_yaml",
    "create_k8s_hpa_yaml",
    "create_k8s_configmap_yaml",
    "create_k8s_secret_yaml",
    "generate_rollback_plan",
    "create_helm_chart",
    "rollback_k8s_deployment",
    "rollback_lambda_stack",
    "rollback_deployment",
    # AWS automation
    "list_eks_clusters",
    "update_kubeconfig",
    "create_ecr_repo",
    "get_ecr_login",
    "generate_iam_policy",
    "estimate_deployment_cost",
]


def get_available_tools() -> list:
    """Get list of available MLOps tools."""
    return AVAILABLE_TOOLS.copy()
