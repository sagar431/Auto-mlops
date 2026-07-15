"""Hydra configuration tool implementations and declarative registrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from ..common.paths import ensure_directory, relative_to_project
from ..compatibility import RootModuleHandler
from ..registry import ToolSpec
from ..schemas.hydra import (
    AnalyzeProjectConfigInput,
    CreateHydraConfigInput,
    UpdateHydraConfigInput,
    ValidateHydraConfigInput,
)


@dataclass(frozen=True)
class HydraDependencies:
    """Injectable filesystem helpers used by extracted Hydra implementations."""

    ensure_directory: Callable[[str | Path], Path] = ensure_directory
    relative_to_project: Callable[[str, str | Path], str] = relative_to_project


_dependencies = HydraDependencies()


def configure_dependencies(dependencies: HydraDependencies) -> None:
    """Configure compatibility-aware dependencies during root-facade startup."""
    global _dependencies
    _dependencies = dependencies


def analyze_project_config(project_path: str) -> dict[str, Any]:
    """Analyze project structure for configuration needs."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    analysis = {
        "has_hydra": (path / "configs").exists(),
        "has_config_yaml": (path / "configs" / "config.yaml").exists(),
        "has_requirements": (path / "requirements.txt").exists(),
        "has_train_script": (path / "train.py").exists(),
        "has_model_dir": (path / "model").exists() or (path / "models").exists(),
        "python_files": [file.name for file in path.glob("*.py")],
        "config_files": [file.name for file in path.glob("**/*.yaml")]
        + [file.name for file in path.glob("**/*.yml")],
    }
    requirements_path = path / "requirements.txt"
    if requirements_path.exists():
        content = requirements_path.read_text().lower()
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


def create_hydra_config(
    project_path: str,
    config_name: str = "config",
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    data_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create Hydra configuration structure."""
    path = Path(project_path)
    if not path.exists():
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
    configs_dir = _dependencies.ensure_directory(path / "configs")
    _dependencies.ensure_directory(configs_dir / "model")
    _dependencies.ensure_directory(configs_dir / "training")
    _dependencies.ensure_directory(configs_dir / "data")
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
    with config_path.open("w") as stream:
        yaml.dump(main_config, stream, default_flow_style=False, sort_keys=False)
    created_files.append(str(config_path))
    model_path = configs_dir / "model" / "default.yaml"
    with model_path.open("w") as stream:
        yaml.dump(default_model, stream, default_flow_style=False)
    created_files.append(str(model_path))
    training_path = configs_dir / "training" / "default.yaml"
    with training_path.open("w") as stream:
        yaml.dump(default_training, stream, default_flow_style=False)
    created_files.append(str(training_path))
    data_path = configs_dir / "data" / "default.yaml"
    with data_path.open("w") as stream:
        yaml.dump(default_data, stream, default_flow_style=False)
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
                    f"{_dependencies.relative_to_project(project_path, config_path)}."
                ),
            }
        ],
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "configuration",
                    "producing_step": "create_or_validate_hydra_config",
                    "state": "generated",
                    "path": _dependencies.relative_to_project(project_path, config_path),
                }
            ]
        },
        "message": f"Hydra configuration created at {configs_dir}",
    }


def update_hydra_config(
    project_path: str,
    config_path: str = "configs/config.yaml",
    updates: dict[str, Any] = None,
) -> dict[str, Any]:
    """Update existing Hydra configuration."""
    full_path = Path(project_path) / config_path
    if not full_path.exists():
        return {"success": False, "error": f"Config file {full_path} does not exist"}
    try:
        with full_path.open() as stream:
            config = yaml.safe_load(stream)

        def deep_update(destination, incoming):
            for key, value in incoming.items():
                if isinstance(value, dict) and isinstance(destination.get(key), dict):
                    deep_update(destination[key], value)
                else:
                    destination[key] = value

        deep_update(config, updates or {})
        with full_path.open("w") as stream:
            yaml.dump(config, stream, default_flow_style=False, sort_keys=False)
        return {
            "success": True,
            "config_path": str(full_path),
            "updated_config": config,
            "message": f"Configuration updated at {full_path}",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def validate_hydra_config(
    project_path: str, config_path: str = "configs/config.yaml"
) -> dict[str, Any]:
    """Validate Hydra configuration."""
    full_path = Path(project_path) / config_path
    if not full_path.exists():
        return {"success": False, "error": f"Config file {full_path} does not exist"}
    issues = []
    warnings = []
    try:
        with full_path.open() as stream:
            config = yaml.safe_load(stream)
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
                            if not sub_config_path.exists():
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
