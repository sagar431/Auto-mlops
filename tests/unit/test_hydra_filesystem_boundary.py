"""Tests for Hydra's explicit, scoped filesystem dependency boundary."""

from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import yaml

import mcp_mlops_tools
from mcp_servers.mlops.domains.hydra import HydraDependencies, use_dependencies
from mcp_servers.mlops.server import build_tool_registry


class RecordingHydraFilesystem:
    """Small in-memory filesystem that records every boundary operation."""

    def __init__(
        self,
        *,
        directories: set[str] | None = None,
        files: dict[str, str] | None = None,
    ) -> None:
        self.directories = {str(Path(path)) for path in directories or set()}
        self.files = {str(Path(path)): content for path, content in (files or {}).items()}
        self.operations: list[tuple[Any, ...]] = []

    def exists(self, path: str | Path) -> bool:
        normalized = str(Path(path))
        self.operations.append(("exists", normalized))
        return normalized in self.directories or normalized in self.files

    def ensure_directory(self, path: str | Path) -> Path:
        normalized = str(Path(path))
        self.operations.append(("ensure_directory", normalized))
        self.directories.add(normalized)
        return Path(normalized)

    def glob(self, path: str | Path, pattern: str) -> list[Path]:
        base = Path(path)
        self.operations.append(("glob", str(base), pattern))
        matches = []
        for candidate_text in self.files:
            candidate = Path(candidate_text)
            try:
                relative = candidate.relative_to(base)
            except ValueError:
                continue
            if pattern.startswith("**/"):
                matched = relative.match(pattern.removeprefix("**/"))
            else:
                matched = relative.parent == Path(".") and relative.match(pattern)
            if matched:
                matches.append(candidate)
        return sorted(matches)

    def read_text(self, path: str | Path) -> str:
        normalized = str(Path(path))
        self.operations.append(("read_text", normalized))
        return self.files[normalized]

    def read_yaml(self, path: str | Path) -> Any:
        normalized = str(Path(path))
        self.operations.append(("read_yaml", normalized))
        return yaml.safe_load(self.files[normalized])

    def write_yaml(
        self, path: str | Path, value: Any, *, sort_keys: bool = True
    ) -> None:
        normalized = str(Path(path))
        self.operations.append(("write_yaml", normalized, sort_keys))
        self.files[normalized] = yaml.dump(
            value, default_flow_style=False, sort_keys=sort_keys
        )

    def relative_to_project(self, project_path: str, artifact_path: str | Path) -> str:
        self.operations.append(
            ("relative_to_project", project_path, str(Path(artifact_path)))
        )
        path = Path(artifact_path)
        try:
            return str(path.relative_to(Path(project_path)))
        except ValueError:
            return str(path)


def _dependencies(filesystem) -> HydraDependencies:
    return HydraDependencies(filesystem=filesystem)


def _text_result(contents) -> dict[str, Any]:
    assert len(contents) == 1
    return json.loads(contents[0].text)


def test_all_hydra_handlers_use_recording_filesystem_boundary():
    filesystem = RecordingHydraFilesystem(
        directories={"/project"},
        files={
            "/project/requirements.txt": "hydra-core\ntorch\n",
            "/project/train.py": "print('train')\n",
            "/project/existing.yaml": "value: true\n",
        },
    )

    with use_dependencies(_dependencies(filesystem)):
        analysis = mcp_mlops_tools.analyze_project_config("/project")
        created = mcp_mlops_tools.create_hydra_config("/project")
        updated = mcp_mlops_tools.update_hydra_config(
            "/project", updates={"seed": 7}
        )
        validated = mcp_mlops_tools.validate_hydra_config("/project")

    assert analysis["success"] is True
    assert analysis["framework"]["hydra"] is True
    assert created["success"] is True
    assert updated["success"] is True
    assert updated["updated_config"]["seed"] == 7
    assert validated["success"] is True
    assert ("glob", "/project", "*.py") in filesystem.operations
    assert ("read_text", "/project/requirements.txt") in filesystem.operations
    assert ("read_yaml", "/project/configs/config.yaml") in filesystem.operations
    assert ("write_yaml", "/project/configs/config.yaml", False) in filesystem.operations


def test_create_records_exact_directories_writes_and_established_yaml():
    filesystem = RecordingHydraFilesystem(directories={"/project"})

    with use_dependencies(_dependencies(filesystem)):
        result = mcp_mlops_tools.create_hydra_config(
            "/project", model_config={"name": "contract-model"}
        )

    assert result["artifact_manifest"]["entries"][0]["path"] == "configs/config.yaml"
    assert [operation for operation in filesystem.operations if operation[0] == "ensure_directory"] == [
        ("ensure_directory", "/project/configs"),
        ("ensure_directory", "/project/configs/model"),
        ("ensure_directory", "/project/configs/training"),
        ("ensure_directory", "/project/configs/data"),
    ]
    assert [operation for operation in filesystem.operations if operation[0] == "write_yaml"] == [
        ("write_yaml", "/project/configs/config.yaml", False),
        ("write_yaml", "/project/configs/model/default.yaml", True),
        ("write_yaml", "/project/configs/training/default.yaml", True),
        ("write_yaml", "/project/configs/data/default.yaml", True),
    ]
    assert filesystem.files["/project/configs/config.yaml"] == (
        "defaults:\n"
        "- model: default\n"
        "- training: default\n"
        "- data: default\n"
        "- _self_\n"
        "experiment_name: ${model.name}_${training.optimizer}_lr${training.learning_rate}\n"
        "seed: 42\n"
        "device: cuda\n"
        "output_dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}\n"
        "mlflow:\n"
        "  tracking_uri: mlruns\n"
        "  experiment_name: ${experiment_name}\n"
    )
    assert filesystem.files["/project/configs/model/default.yaml"] == (
        "name: contract-model\n"
    )
    assert filesystem.files["/project/configs/training/default.yaml"] == (
        "batch_size: 32\n"
        "early_stopping:\n"
        "  min_delta: 0.001\n"
        "  patience: 5\n"
        "epochs: 10\n"
        "learning_rate: 0.001\n"
        "optimizer: adam\n"
        "scheduler: cosine\n"
    )
    assert filesystem.files["/project/configs/data/default.yaml"] == (
        "augmentation: true\n"
        "num_workers: 4\n"
        "test_path: data/test\n"
        "train_path: data/train\n"
        "val_path: data/val\n"
    )


@pytest.mark.asyncio
async def test_failing_filesystem_preserves_mcp_error_contract():
    class FailingFilesystem(RecordingHydraFilesystem):
        def write_yaml(
            self, path: str | Path, value: Any, *, sort_keys: bool = True
        ) -> None:
            raise OSError("recording filesystem is read-only")

    filesystem = FailingFilesystem(directories={"/project"})
    with use_dependencies(_dependencies(filesystem)):
        result = _text_result(
            await mcp_mlops_tools.call_tool(
                "create_hydra_config", {"project_path": "/project"}
            )
        )

    assert result == {
        "error": "recording filesystem is read-only",
        "tool": "create_hydra_config",
    }


def test_dependency_override_is_scoped_and_does_not_leak(tmp_path):
    first = RecordingHydraFilesystem(directories={"/project"})
    second = RecordingHydraFilesystem(directories={"/project"})

    with use_dependencies(_dependencies(first)):
        mcp_mlops_tools.create_hydra_config("/project")
        with use_dependencies(_dependencies(second)):
            mcp_mlops_tools.create_hydra_config("/project")
        mcp_mlops_tools.validate_hydra_config("/project")

    assert ("read_yaml", "/project/configs/config.yaml") in first.operations
    assert ("read_yaml", "/project/configs/config.yaml") not in second.operations
    result = mcp_mlops_tools.create_hydra_config(str(tmp_path))
    assert result["success"] is True
    assert (tmp_path / "configs" / "config.yaml").exists()


def test_root_filesystem_patch_seam_is_visible_in_worker_threads(tmp_path, monkeypatch):
    observed: list[Path] = []
    original = mcp_mlops_tools.ensure_directory

    def tracking_ensure_directory(path):
        observed.append(Path(path))
        return original(path)

    monkeypatch.setattr(mcp_mlops_tools, "ensure_directory", tracking_ensure_directory)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            mcp_mlops_tools.create_hydra_config, str(tmp_path)
        ).result()

    assert result["success"] is True
    assert observed == [
        tmp_path / "configs",
        tmp_path / "configs" / "model",
        tmp_path / "configs" / "training",
        tmp_path / "configs" / "data",
    ]


@pytest.mark.asyncio
async def test_all_direct_and_mcp_calls_are_equivalent_with_injected_filesystem():
    direct_analysis_filesystem = RecordingHydraFilesystem(
        directories={"/project"},
        files={"/project/requirements.txt": "hydra-core\n"},
    )
    mcp_analysis_filesystem = RecordingHydraFilesystem(
        directories={"/project"},
        files={"/project/requirements.txt": "hydra-core\n"},
    )
    with use_dependencies(_dependencies(direct_analysis_filesystem)):
        direct_analysis = mcp_mlops_tools.analyze_project_config("/project")
    with use_dependencies(_dependencies(mcp_analysis_filesystem)):
        mcp_analysis = _text_result(
            await mcp_mlops_tools.call_tool(
                "analyze_project_config", {"project_path": "/project"}
            )
        )
    assert mcp_analysis == direct_analysis

    direct_create_filesystem = RecordingHydraFilesystem(directories={"/project"})
    mcp_create_filesystem = RecordingHydraFilesystem(directories={"/project"})
    with use_dependencies(_dependencies(direct_create_filesystem)):
        direct_create = mcp_mlops_tools.create_hydra_config(
            "/project", model_config={"name": "equivalent"}
        )
    with use_dependencies(_dependencies(mcp_create_filesystem)):
        mcp_create = _text_result(
            await mcp_mlops_tools.call_tool(
                "create_hydra_config",
                {
                    "project_path": "/project",
                    "ml_model_config": {"name": "equivalent"},
                },
            )
        )
    assert mcp_create == direct_create
    assert mcp_create_filesystem.files == direct_create_filesystem.files

    initial_config = "defaults: []\nseed: 42\n"
    direct_update_filesystem = RecordingHydraFilesystem(
        directories={"/project", "/project/configs"},
        files={"/project/configs/config.yaml": initial_config},
    )
    mcp_update_filesystem = RecordingHydraFilesystem(
        directories={"/project", "/project/configs"},
        files={"/project/configs/config.yaml": initial_config},
    )
    with use_dependencies(_dependencies(direct_update_filesystem)):
        direct_update = mcp_mlops_tools.update_hydra_config(
            "/project", updates={"seed": 7}
        )
    with use_dependencies(_dependencies(mcp_update_filesystem)):
        mcp_update = _text_result(
            await mcp_mlops_tools.call_tool(
                "update_hydra_config",
                {"project_path": "/project", "updates": {"seed": 7}},
            )
        )
    assert mcp_update == direct_update
    assert mcp_update_filesystem.files == direct_update_filesystem.files

    direct_validate_filesystem = RecordingHydraFilesystem(
        directories={"/project", "/project/configs"},
        files={"/project/configs/config.yaml": initial_config},
    )
    mcp_validate_filesystem = RecordingHydraFilesystem(
        directories={"/project", "/project/configs"},
        files={"/project/configs/config.yaml": initial_config},
    )
    with use_dependencies(_dependencies(direct_validate_filesystem)):
        direct_validate = mcp_mlops_tools.validate_hydra_config("/project")
    with use_dependencies(_dependencies(mcp_validate_filesystem)):
        mcp_validate = _text_result(
            await mcp_mlops_tools.call_tool(
                "validate_hydra_config", {"project_path": "/project"}
            )
        )

    assert mcp_validate == direct_validate


def test_missing_files_and_invalid_yaml_keep_established_results():
    filesystem = RecordingHydraFilesystem(
        directories={"/project", "/project/configs"},
        files={"/project/configs/broken.yaml": "defaults: [unterminated\n"},
    )

    with use_dependencies(_dependencies(filesystem)):
        missing_project = mcp_mlops_tools.analyze_project_config("/missing")
        missing_create = mcp_mlops_tools.create_hydra_config("/missing")
        missing_update = mcp_mlops_tools.update_hydra_config("/project")
        missing_config = mcp_mlops_tools.validate_hydra_config("/project")
        invalid = mcp_mlops_tools.validate_hydra_config(
            "/project", "configs/broken.yaml"
        )
        invalid_update = mcp_mlops_tools.update_hydra_config(
            "/project", "configs/broken.yaml", {"seed": 7}
        )

    assert missing_project == {
        "success": False,
        "error": "Project path /missing does not exist",
    }
    assert missing_create == missing_project
    assert missing_update == {
        "success": False,
        "error": "Config file /project/configs/config.yaml does not exist",
    }
    assert missing_config == {
        "success": False,
        "error": "Config file /project/configs/config.yaml does not exist",
    }
    assert invalid["success"] is False
    assert invalid["error"].startswith("Invalid YAML: ")
    assert invalid_update["success"] is False
    assert invalid_update["error"] == invalid["error"].removeprefix("Invalid YAML: ")


def test_registry_construction_performs_no_filesystem_operations():
    filesystem = RecordingHydraFilesystem(directories={"/project"})

    with use_dependencies(_dependencies(filesystem)):
        registry = build_tool_registry(mcp_mlops_tools)

    assert len(registry.specs) == 98
    assert filesystem.operations == []


def test_historical_public_signatures_are_unchanged():
    assert str(inspect.signature(mcp_mlops_tools.analyze_project_config)) == (
        "(project_path: 'str') -> 'dict[str, Any]'"
    )
    assert str(inspect.signature(mcp_mlops_tools.create_hydra_config)) == (
        "(project_path: 'str', config_name: 'str' = 'config', "
        "model_config: 'dict[str, Any] | None' = None, "
        "training_config: 'dict[str, Any] | None' = None, "
        "data_config: 'dict[str, Any] | None' = None) -> 'dict[str, Any]'"
    )
    assert str(inspect.signature(mcp_mlops_tools.update_hydra_config)) == (
        "(project_path: 'str', config_path: 'str' = 'configs/config.yaml', "
        "updates: 'dict[str, Any]' = None) -> 'dict[str, Any]'"
    )
    assert str(inspect.signature(mcp_mlops_tools.validate_hydra_config)) == (
        "(project_path: 'str', config_path: 'str' = 'configs/config.yaml') "
        "-> 'dict[str, Any]'"
    )
