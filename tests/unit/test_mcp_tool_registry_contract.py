"""Contract tests for the modular MCP registry and root compatibility facade."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

import mcp_mlops_tools
from action.execute_step import execute_step

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "mcp_tool_contract.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text())


def _tool_payload(tool) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_model": tool.inputSchema["title"],
        "input_schema": tool.inputSchema,
        "handler": tool.name,
    }


def _text_result(contents) -> dict[str, object]:
    assert len(contents) == 1
    assert contents[0].type == "text"
    return json.loads(contents[0].text)


@pytest.mark.asyncio
async def test_complete_catalog_matches_canonical_pre_refactor_contract():
    observed = [_tool_payload(tool) for tool in await mcp_mlops_tools.list_tools()]
    assert observed == _contract()["tools"]


def test_registry_has_unique_complete_reachable_specs():
    from mcp_servers.mlops.registry import ToolRegistry

    registry = mcp_mlops_tools.TOOL_REGISTRY
    contract = _contract()
    assert isinstance(registry, ToolRegistry)
    names = [spec.name for spec in registry.specs]
    assert len(names) == len(set(names)) == contract["registered_tool_count"] == 98
    assert contract["duplicate_tool_names"] == []
    assert contract["registration_entries_without_callable_handlers"] == []
    assert contract["implemented_tool_handlers_not_registered"] == []
    for spec, expected in zip(registry.specs, contract["tools"], strict=True):
        assert spec.description.strip()
        assert issubclass(spec.input_model, BaseModel)
        assert callable(spec.handler)
        assert spec.handler.handler_name == expected["handler"]


def test_tool_spec_rejects_duplicate_names_and_non_callable_handlers():
    from mcp_servers.mlops.registry import ToolRegistry, ToolSpec
    from mcp_servers.mlops.schemas.hydra import AnalyzeProjectConfigInput

    spec = ToolSpec(
        name="sample",
        description="Sample tool",
        input_model=AnalyzeProjectConfigInput,
        handler=lambda validated: validated.model_dump(),
    )
    registry = ToolRegistry([spec])
    with pytest.raises(ValueError, match="Duplicate MCP tool name"):
        registry.register(spec)
    with pytest.raises(TypeError, match="handler must be callable"):
        ToolSpec(
            name="invalid",
            description="Invalid tool",
            input_model=AnalyzeProjectConfigInput,
            handler=None,
        )


@pytest.mark.asyncio
async def test_unknown_and_invalid_input_response_contracts_are_unchanged():
    unknown = _text_result(await mcp_mlops_tools.call_tool("not-a-tool", {}))
    assert unknown == {"error": "Unknown tool: not-a-tool"}

    try:
        mcp_mlops_tools.AnalyzeProjectConfigInput(**{})
    except ValidationError as exc:
        expected_error = str(exc)
    else:  # pragma: no cover - the schema contract requires validation failure
        raise AssertionError("missing project_path unexpectedly validated")
    invalid = _text_result(await mcp_mlops_tools.call_tool("analyze_project_config", {}))
    assert invalid == {"error": expected_error, "tool": "analyze_project_config"}


@pytest.mark.asyncio
async def test_representative_mcp_call_matches_direct_root_call(tmp_path):
    (tmp_path / "requirements.txt").write_text("hydra-core\ntorch\n")
    (tmp_path / "train.py").write_text("print('train')\n")
    direct = mcp_mlops_tools.analyze_project_config(str(tmp_path))
    through_mcp = _text_result(
        await mcp_mlops_tools.call_tool(
            "analyze_project_config", {"project_path": str(tmp_path)}
        )
    )
    assert through_mcp == direct


@pytest.mark.asyncio
async def test_root_handler_monkeypatch_remains_effective(monkeypatch):
    expected = {"success": True, "source": "root-monkeypatch"}
    monkeypatch.setattr(
        mcp_mlops_tools,
        "analyze_project_config",
        lambda project_path: {**expected, "project_path": project_path},
    )
    result = _text_result(
        await mcp_mlops_tools.call_tool(
            "analyze_project_config", {"project_path": "/patched"}
        )
    )
    assert result == {**expected, "project_path": "/patched"}


@pytest.mark.asyncio
async def test_extracted_hydra_preserves_root_filesystem_patch_seam(tmp_path, monkeypatch):
    observed: list[Path] = []
    original = mcp_mlops_tools.ensure_directory

    def tracking_ensure_directory(path):
        observed.append(Path(path))
        return original(path)

    monkeypatch.setattr(mcp_mlops_tools, "ensure_directory", tracking_ensure_directory)
    result = _text_result(
        await mcp_mlops_tools.call_tool(
            "create_hydra_config",
            {
                "project_path": str(tmp_path),
                "ml_model_config": {"name": "contract-model"},
            },
        )
    )

    assert result["success"] is True
    assert result["artifact_manifest"]["entries"][0]["path"] == "configs/config.yaml"
    assert (tmp_path / "configs" / "model" / "default.yaml").read_text() == (
        "name: contract-model\n"
    )
    assert observed == [
        tmp_path / "configs",
        tmp_path / "configs" / "model",
        tmp_path / "configs" / "training",
        tmp_path / "configs" / "data",
    ]


def test_root_compatibility_imports_and_extracted_hydra_identity():
    from mcp_mlops_tools import (
        AnalyzeProjectConfigInput,
        create_hydra_config,
        detect_data_drift,
        setup_alerting,
    )
    from mcp_servers.mlops.domains.hydra import create_hydra_config as extracted_create
    from mcp_servers.mlops.schemas.hydra import AnalyzeProjectConfigInput as extracted_model

    assert create_hydra_config is extracted_create
    assert AnalyzeProjectConfigInput is extracted_model
    assert callable(detect_data_drift)
    assert callable(setup_alerting)


def test_inventoried_root_imports_and_patch_seams_remain_available():
    contract = _contract()
    for name in contract["public_root_imports_elsewhere"]:
        assert hasattr(mcp_mlops_tools, name), name

    for dotted_name in contract["root_module_test_patch_seams"]:
        value = mcp_mlops_tools
        for part in dotted_name.split("."):
            value = getattr(value, part)
        assert value is not None


@pytest.mark.asyncio
async def test_execute_step_direct_python_invocation_remains_compatible(tmp_path):
    context = SimpleNamespace(project_path=str(tmp_path))
    success, result = await execute_step(
        "analyze",
        "analyze_project_config",
        {},
        context,
        tools_module=mcp_mlops_tools,
    )
    assert success is True
    assert result["result"]["success"] is True
    assert result["tool"] == "analyze_project_config"


def test_mcp_server_import_and_bounded_stdio_startup():
    process = subprocess.Popen(
        [sys.executable, "mcp_mlops_tools.py"],
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.5)
        assert process.poll() is None
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    stderr = process.stderr.read() if process.stderr else ""
    assert "Traceback" not in stderr


def test_async_catalog_is_deterministic_across_calls():
    async def names() -> list[str]:
        return [tool.name for tool in await mcp_mlops_tools.list_tools()]

    assert asyncio.run(names()) == asyncio.run(names())
