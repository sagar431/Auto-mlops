"""Input schemas for Hydra configuration MCP tools."""

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeProjectConfigInput(BaseModel):
    """Analyze project for configuration needs."""

    project_path: str = Field(..., description="Path to the ML project")


class CreateHydraConfigInput(BaseModel):
    """Create Hydra configuration structure."""

    project_path: str = Field(..., description="Path to the ML project")
    config_name: str = Field(default="config", description="Name of main config file")
    ml_model_config: dict[str, Any] | None = Field(default=None, description="Model configuration")
    training_config: dict[str, Any] | None = Field(
        default=None, description="Training configuration"
    )
    data_config: dict[str, Any] | None = Field(default=None, description="Data configuration")


class UpdateHydraConfigInput(BaseModel):
    """Update existing Hydra configuration."""

    project_path: str = Field(..., description="Path to the ML project")
    config_path: str = Field(
        default="configs/config.yaml", description="Relative path to config file"
    )
    updates: dict[str, Any] = Field(..., description="Dictionary of updates to apply")


class ValidateHydraConfigInput(BaseModel):
    """Validate Hydra configuration."""

    project_path: str = Field(..., description="Path to the ML project")
    config_path: str = Field(
        default="configs/config.yaml", description="Relative path to config file"
    )
