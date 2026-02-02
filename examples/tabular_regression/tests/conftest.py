"""Pytest fixtures for tabular regression tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project directory to path
project_dir = Path(__file__).parent.parent / "project"
sys.path.insert(0, str(project_dir))


@pytest.fixture
def sample_features():
    """Sample feature data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 8).astype(np.float32)


@pytest.fixture
def sample_targets():
    """Sample target data for testing."""
    np.random.seed(42)
    return np.random.randn(100).astype(np.float32)


@pytest.fixture
def feature_names():
    """Feature names for testing."""
    return [f"feature_{i}" for i in range(8)]


@pytest.fixture
def tmp_data_dir(tmp_path, sample_features, sample_targets, feature_names):
    """Create temporary data directory with train/test files."""
    import pandas as pd

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Split data
    train_features = sample_features[:80]
    train_targets = sample_targets[:80]
    test_features = sample_features[80:]
    test_targets = sample_targets[80:]

    # Create train DataFrame
    train_df = pd.DataFrame(train_features, columns=feature_names)
    train_df["target"] = train_targets
    train_df.to_csv(data_dir / "train.csv", index=False)

    # Create test DataFrame
    test_df = pd.DataFrame(test_features, columns=feature_names)
    test_df["target"] = test_targets
    test_df.to_csv(data_dir / "test.csv", index=False)

    return data_dir


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
