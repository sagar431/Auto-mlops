"""Pytest configuration for text classification tests."""

import sys
from pathlib import Path

import pytest

# Add project to path for imports
project_path = Path(__file__).parent.parent / "project"
sys.path.insert(0, str(project_path))


@pytest.fixture(scope="session")
def project_dir():
    """Return the project directory path."""
    return project_path
