"""Filesystem path helpers shared by MCP domain modules."""

from pathlib import Path


def ensure_directory(path: str) -> Path:
    """Ensure directory exists and return a Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def relative_to_project(project_path: str, artifact_path: str | Path) -> str:
    """Return a project-relative artifact path when possible."""
    path = Path(artifact_path)
    try:
        return str(path.relative_to(Path(project_path)))
    except ValueError:
        return str(path)
