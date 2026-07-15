"""Filesystem boundary used by the extracted Hydra MCP domain."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from ..common.paths import ensure_directory, relative_to_project


class HydraFilesystem(Protocol):
    """The complete set of filesystem operations used by Hydra handlers."""

    def exists(self, path: str | Path) -> bool:
        """Return whether a path exists."""

    def ensure_directory(self, path: str | Path) -> Path:
        """Create a directory and return its normalized path."""

    def glob(self, path: str | Path, pattern: str) -> list[Path]:
        """Discover files below a path using a glob pattern."""

    def read_text(self, path: str | Path) -> str:
        """Read a text file."""

    def read_yaml(self, path: str | Path) -> Any:
        """Safely parse a YAML file."""

    def write_yaml(
        self, path: str | Path, value: Any, *, sort_keys: bool = True
    ) -> None:
        """Write YAML using the established Hydra serialization options."""

    def relative_to_project(self, project_path: str, artifact_path: str | Path) -> str:
        """Return a project-relative artifact path when possible."""


@dataclass(frozen=True)
class LocalHydraFilesystem:
    """Production Hydra filesystem backed by the real local filesystem."""

    directory_creator: Callable[[str | Path], Path] = ensure_directory
    project_relativizer: Callable[[str, str | Path], str] = relative_to_project

    def exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    def ensure_directory(self, path: str | Path) -> Path:
        return self.directory_creator(path)

    def glob(self, path: str | Path, pattern: str) -> list[Path]:
        return list(Path(path).glob(pattern))

    def read_text(self, path: str | Path) -> str:
        return Path(path).read_text()

    def read_yaml(self, path: str | Path) -> Any:
        with Path(path).open() as stream:
            return yaml.safe_load(stream)

    def write_yaml(
        self, path: str | Path, value: Any, *, sort_keys: bool = True
    ) -> None:
        with Path(path).open("w") as stream:
            yaml.dump(
                value,
                stream,
                default_flow_style=False,
                sort_keys=sort_keys,
            )

    def relative_to_project(self, project_path: str, artifact_path: str | Path) -> str:
        return self.project_relativizer(project_path, artifact_path)
