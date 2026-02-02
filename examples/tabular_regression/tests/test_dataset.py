"""Tests for dataset utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project directory to path
project_dir = Path(__file__).parent.parent / "project"
sys.path.insert(0, str(project_dir))

from dataset import (  # noqa: E402
    TabularDataset,
    create_dataloaders,
    create_synthetic_data,
    load_data,
    save_synthetic_data,
)


class TestTabularDataset:
    """Tests for TabularDataset class."""

    def test_init_with_targets(self, sample_features, sample_targets, feature_names):
        """Test dataset initialization with targets."""
        dataset = TabularDataset(sample_features, sample_targets, feature_names)

        assert len(dataset) == 100
        assert dataset.feature_names == feature_names

    def test_init_without_targets(self, sample_features, feature_names):
        """Test dataset initialization without targets."""
        dataset = TabularDataset(sample_features, feature_names=feature_names)

        assert len(dataset) == 100
        assert dataset.targets is None

    def test_getitem_with_targets(self, sample_features, sample_targets, feature_names):
        """Test __getitem__ with targets."""
        dataset = TabularDataset(sample_features, sample_targets, feature_names)
        features, target = dataset[0]

        assert features.shape == (8,)
        assert target.shape == ()

    def test_getitem_without_targets(self, sample_features, feature_names):
        """Test __getitem__ without targets."""
        dataset = TabularDataset(sample_features, feature_names=feature_names)
        result = dataset[0]

        assert len(result) == 1
        assert result[0].shape == (8,)


class TestLoadData:
    """Tests for load_data function."""

    def test_load_train_data(self, tmp_data_dir, feature_names):
        """Test loading training data."""
        features, targets, names = load_data(str(tmp_data_dir), split="train")

        assert features.shape == (80, 8)
        assert targets.shape == (80,)
        assert names == feature_names

    def test_load_test_data(self, tmp_data_dir, feature_names):
        """Test loading test data."""
        features, targets, names = load_data(str(tmp_data_dir), split="test")

        assert features.shape == (20, 8)
        assert targets.shape == (20,)

    def test_file_not_found(self, tmp_path):
        """Test error when file not found."""
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path), split="train")


class TestCreateDataloaders:
    """Tests for create_dataloaders function."""

    def test_create_dataloaders(self, tmp_data_dir):
        """Test creating dataloaders."""
        train_loader, test_loader, info = create_dataloaders(str(tmp_data_dir), batch_size=16)

        assert info["input_dim"] == 8
        assert info["train_samples"] == 80
        assert info["test_samples"] == 20

        # Test iteration
        batch = next(iter(train_loader))
        assert len(batch) == 2
        assert batch[0].shape[1] == 8


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_create_synthetic_data(self):
        """Test creating synthetic data."""
        features, targets, names = create_synthetic_data(n_samples=100, n_features=5, seed=42)

        assert features.shape == (100, 5)
        assert targets.shape == (100,)
        assert len(names) == 5

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        f1, t1, _ = create_synthetic_data(n_samples=50, seed=42)
        f2, t2, _ = create_synthetic_data(n_samples=50, seed=42)

        assert np.allclose(f1, f2)
        assert np.allclose(t1, t2)

    def test_save_synthetic_data(self, tmp_path):
        """Test saving synthetic data."""
        info = save_synthetic_data(str(tmp_path), n_train=100, n_test=50, n_features=6, seed=42)

        assert (tmp_path / "train.csv").exists()
        assert (tmp_path / "test.csv").exists()
        assert (tmp_path / "data_info.json").exists()
        assert info["n_train"] == 100
        assert info["n_test"] == 50
        assert info["n_features"] == 6
