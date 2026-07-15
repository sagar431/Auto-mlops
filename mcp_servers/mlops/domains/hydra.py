"""Hydra configuration tool implementations and declarative registrations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from ..compatibility import RootModuleHandler
from ..registry import ToolSpec
from ..schemas.hydra import (
    AnalyzeProjectConfigInput,
    CreateHydraConfigInput,
    UpdateHydraConfigInput,
    ValidateHydraConfigInput,
)
from .hydra_filesystem import HydraFilesystem, LocalHydraFilesystem


@dataclass(frozen=True)
class HydraDependencies:
    """Immutable dependencies used by the extracted Hydra implementations."""

    filesystem: HydraFilesystem = field(default_factory=LocalHydraFilesystem)


_configured_dependencies = HydraDependencies()
_dependency_override: ContextVar[HydraDependencies | None] = ContextVar(
    "hydra_dependency_override", default=None
)


def configure_dependencies(dependencies: HydraDependencies) -> None:
    """Configure the immutable dependency baseline used by the root facade."""
    global _configured_dependencies
    _configured_dependencies = dependencies


@contextmanager
def use_dependencies(dependencies: HydraDependencies) -> Iterator[None]:
    """Temporarily inject Hydra dependencies and restore prior state afterward."""
    token = _dependency_override.set(dependencies)
    try:
        yield
    finally:
        _dependency_override.reset(token)


def _current_filesystem() -> HydraFilesystem:
    dependencies = _dependency_override.get() or _configured_dependencies
    return dependencies.filesystem


def _analyze_project_config(
    project_path: str, filesystem: HydraFilesystem
) -> dict[str, Any]:
    path = Path(project_path)
    if not filesystem.exists(path):
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    analysis = {
        "has_hydra": filesystem.exists(path / "configs"),
        "has_config_yaml": filesystem.exists(path / "configs" / "config.yaml"),
        "has_requirements": filesystem.exists(path / "requirements.txt"),
        "has_train_script": filesystem.exists(path / "train.py"),
        "has_model_dir": filesystem.exists(path / "model")
        or filesystem.exists(path / "models"),
        "python_files": [file.name for file in filesystem.glob(path, "*.py")],
        "config_files": [file.name for file in filesystem.glob(path, "**/*.yaml")]
        + [file.name for file in filesystem.glob(path, "**/*.yml")],
    }
    requirements_path = path / "requirements.txt"
    if filesystem.exists(requirements_path):
        content = filesystem.read_text(requirements_path).lower()
        analysis["framework"] = {
            "pytorch": "torch" in content or "pytorch" in content,
            "tensorflow": "tensorflow" in content,
            "sklearn": "scikit-learn" in content or "sklearn" in content,
            "hydra": "hydra-core" in content,
            "mlflow": "mlflow" in content,
            "dvc": "dvc" in content,
        }
    analysis["success"] = True
    analysis["recommendations"] = []
    if not analysis["has_hydra"]:
        analysis["recommendations"].append(
            "Create configs/ directory for Hydra configuration"
        )
    if not analysis["has_requirements"]:
        analysis["recommendations"].append("Add requirements.txt for dependencies")
    if not analysis["has_train_script"]:
        analysis["recommendations"].append("Create train.py as main entry point")
    return analysis


def analyze_project_config(project_path: str) -> dict[str, Any]:
    """Analyze project structure for configuration needs."""
    return _analyze_project_config(project_path, _current_filesystem())


def _create_hydra_config(
    project_path: str,
    config_name: str,
    model_config: dict[str, Any] | None,
    training_config: dict[str, Any] | None,
    data_config: dict[str, Any] | None,
    filesystem: HydraFilesystem,
) -> dict[str, Any]:
    path = Path(project_path)
    if not filesystem.exists(path):
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    default_model = model_config or {
        "name": "resnet18",
        "pretrained": True,
        "num_classes": 2,
        "dropout": 0.5,
    }
    default_training = training_config or {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "scheduler": "cosine",
        "early_stopping": {"patience": 5, "min_delta": 0.001},
    }
    default_data = data_config or {
        "train_path": "data/train",
        "val_path": "data/val",
        "test_path": "data/test",
        "num_workers": 4,
        "augmentation": True,
    }
    configs_dir = filesystem.ensure_directory(path / "configs")
    filesystem.ensure_directory(configs_dir / "model")
    filesystem.ensure_directory(configs_dir / "training")
    filesystem.ensure_directory(configs_dir / "data")
    created_files = []
    main_config = {
        "defaults": [{"model": "default"}, {"training": "default"}, {"data": "default"}, "_self_"],
        "experiment_name": "${model.name}_${training.optimizer}_lr${training.learning_rate}",
        "seed": 42,
        "device": "cuda",
        "output_dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        "mlflow": {"tracking_uri": "mlruns", "experiment_name": "${experiment_name}"},
    }
    config_path = configs_dir / f"{config_name}.yaml"
    filesystem.write_yaml(config_path, main_config, sort_keys=False)
    created_files.append(str(config_path))
    model_path = configs_dir / "model" / "default.yaml"
    filesystem.write_yaml(model_path, default_model)
    created_files.append(str(model_path))
    training_path = configs_dir / "training" / "default.yaml"
    filesystem.write_yaml(training_path, default_training)
    created_files.append(str(training_path))
    data_path = configs_dir / "data" / "default.yaml"
    filesystem.write_yaml(data_path, default_data)
    created_files.append(str(data_path))
    return {
        "success": True,
        "created_files": created_files,
        "config_dir": str(configs_dir),
        "verification_results": [
            {
                "check_name": "hydra_config_validates",
                "evidence_type": "declared",
                "source_step": "create_or_validate_hydra_config",
                "passed": True,
                "evidence": (
                    "Hydra config generated at "
                    f"{filesystem.relative_to_project(project_path, config_path)}."
                ),
            }
        ],
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "configuration",
                    "producing_step": "create_or_validate_hydra_config",
                    "state": "generated",
                    "path": filesystem.relative_to_project(project_path, config_path),
                }
            ]
        },
        "message": f"Hydra configuration created at {configs_dir}",
    }


def create_hydra_config(
    project_path: str,
    config_name: str = "config",
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    data_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create Hydra configuration structure."""
    return _create_hydra_config(
        project_path,
        config_name,
        model_config,
        training_config,
        data_config,
        _current_filesystem(),
    )


def _deep_update(destination: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _deep_update(destination[key], value)
        else:
            destination[key] = value


def _update_hydra_config(
    project_path: str,
    config_path: str,
    updates: dict[str, Any] | None,
    filesystem: HydraFilesystem,
) -> dict[str, Any]:
    full_path = Path(project_path) / config_path
    if not filesystem.exists(full_path):
        return {"success": False, "error": f"Config file {full_path} does not exist"}
    try:
        config = filesystem.read_yaml(full_path)
        _deep_update(config, updates or {})
        filesystem.write_yaml(full_path, config, sort_keys=False)
        return {
            "success": True,
            "config_path": str(full_path),
            "updated_config": config,
            "message": f"Configuration updated at {full_path}",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def update_hydra_config(
    project_path: str,
    config_path: str = "configs/config.yaml",
    updates: dict[str, Any] = None,
) -> dict[str, Any]:
    """Update existing Hydra configuration."""
    return _update_hydra_config(
        project_path,
        config_path,
        updates,
        _current_filesystem(),
    )


def _validate_hydra_config(
    project_path: str, config_path: str, filesystem: HydraFilesystem
) -> dict[str, Any]:
    full_path = Path(project_path) / config_path
    if not filesystem.exists(full_path):
        return {"success": False, "error": f"Config file {full_path} does not exist"}
    issues = []
    warnings = []
    try:
        config = filesystem.read_yaml(full_path)
        if "defaults" not in config:
            warnings.append("No 'defaults' section found - Hydra composition may not work")
        if isinstance(config.get("defaults"), list):
            for default in config["defaults"]:
                if isinstance(default, dict):
                    for key, value in default.items():
                        if key not in ["_self_"]:
                            sub_config_path = (
                                Path(project_path) / "configs" / key / f"{value}.yaml"
                            )
                            if not filesystem.exists(sub_config_path):
                                issues.append(f"Missing config file: {sub_config_path}")
        return {
            "success": len(issues) == 0,
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "config": config,
        }
    except yaml.YAMLError as exc:
        return {"success": False, "error": f"Invalid YAML: {str(exc)}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def validate_hydra_config(
    project_path: str, config_path: str = "configs/config.yaml"
) -> dict[str, Any]:
    """Validate Hydra configuration."""
    return _validate_hydra_config(
        project_path,
        config_path,
        _current_filesystem(),
    )


def tool_specs(root_module: ModuleType) -> tuple[ToolSpec, ...]:
    """Return Hydra ToolSpecs with dynamic root-facade handler resolution."""
    definitions = (
        (
            "analyze_project_config",
            "Analyze ML project structure for configuration needs (Hydra, requirements, scripts)",
            AnalyzeProjectConfigInput,
            {},
        ),
        (
            "create_hydra_config",
            "Create Hydra configuration structure with model, training, and data configs",
            CreateHydraConfigInput,
            {"ml_model_config": "model_config"},
        ),
        (
            "update_hydra_config",
            "Update existing Hydra configuration with new values",
            UpdateHydraConfigInput,
            {},
        ),
        (
            "validate_hydra_config",
            "Validate Hydra configuration for errors and missing files",
            ValidateHydraConfigInput,
            {},
        ),
    )
    return tuple(
        ToolSpec(
            name=name,
            description=description,
            input_model=input_model,
            handler=RootModuleHandler(root_module, name, aliases),
        )
        for name, description, input_model, aliases in definitions
    )
