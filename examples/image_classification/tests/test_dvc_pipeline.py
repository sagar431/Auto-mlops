"""Tests for the DVC pipeline scripts and configuration."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from evaluate import evaluate_model, get_test_transforms, load_model
from prepare_data import prepare_cifar10
from train import CIFAR10CNN


class TestPrepareData:
    """Tests for prepare_data.py script."""

    @pytest.fixture(autouse=True)
    def _offline_cifar10(self, monkeypatch):
        """Preserve CIFAR metadata checks without network access."""

        class OfflineCIFAR10:
            def __init__(self, root, train, download):
                self.root = root
                self.train = train
                self.download = download

            def __len__(self):
                return 50_000 if self.train else 10_000

        monkeypatch.setattr("prepare_data.datasets.CIFAR10", OfflineCIFAR10)

    def test_prepare_cifar10_downloads_data(self, tmp_path):
        """Test that prepare_cifar10 downloads and prepares data correctly."""
        data_dir = tmp_path / "data"
        output_file = tmp_path / "data_info.json"

        result = prepare_cifar10(str(data_dir), str(output_file))

        assert result["dataset"] == "cifar10"
        assert result["train_samples"] == 50000
        assert result["test_samples"] == 10000
        assert result["num_classes"] == 10
        assert len(result["class_names"]) == 10
        assert result["image_size"] == [32, 32]
        assert result["channels"] == 3

    def test_prepare_cifar10_creates_output_file(self, tmp_path):
        """Test that prepare_cifar10 creates the data info JSON file."""
        data_dir = tmp_path / "data"
        output_file = tmp_path / "output" / "data_info.json"

        prepare_cifar10(str(data_dir), str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            data_info = json.load(f)
        assert "train_samples" in data_info
        assert "test_samples" in data_info
        assert "class_names" in data_info

    def test_prepare_cifar10_class_names(self, tmp_path):
        """Test that class names are correct."""
        data_dir = tmp_path / "data"
        output_file = tmp_path / "data_info.json"

        result = prepare_cifar10(str(data_dir), str(output_file))

        expected_classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        assert result["class_names"] == expected_classes


class TestEvaluate:
    """Tests for evaluate.py script."""

    def test_get_test_transforms(self):
        """Test that test transforms are created correctly."""
        transform = get_test_transforms(image_size=32)
        assert transform is not None

    def test_get_test_transforms_different_size(self):
        """Test transforms with different image size."""
        transform = get_test_transforms(image_size=64)
        assert transform is not None

    def test_load_model(self, tmp_path):
        """Test model loading from checkpoint."""
        # Create and save a model
        model = CIFAR10CNN(num_classes=10, dropout=0.5)
        model_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), model_path)

        # Load the model
        loaded_model = load_model(str(model_path), num_classes=10, dropout=0.5)
        assert isinstance(loaded_model, CIFAR10CNN)
        assert loaded_model.num_classes == 10

    def test_load_model_custom_params(self, tmp_path):
        """Test model loading with custom parameters."""
        model = CIFAR10CNN(num_classes=5, dropout=0.3)
        model_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), model_path)

        loaded_model = load_model(str(model_path), num_classes=5, dropout=0.3)
        assert loaded_model.num_classes == 5

    @pytest.fixture
    def mock_test_loader(self):
        """Create a mock test data loader."""
        batch_size = 8
        num_batches = 4
        data = [
            (torch.randn(batch_size, 3, 32, 32), torch.randint(0, 10, (batch_size,)))
            for _ in range(num_batches)
        ]
        loader = MagicMock()
        loader.__iter__ = MagicMock(side_effect=lambda: iter(data))
        loader.__len__ = MagicMock(return_value=num_batches)
        loader.dataset = MagicMock()
        loader.dataset.__len__ = MagicMock(return_value=batch_size * num_batches)
        return loader

    def test_evaluate_model_returns_metrics(self, mock_test_loader, tmp_path):
        """Test that evaluate_model returns expected metrics."""
        model = CIFAR10CNN(num_classes=10)

        # Mock the datasets.CIFAR10 to avoid downloading
        with patch("evaluate.datasets.CIFAR10") as mock_cifar:
            mock_dataset = MagicMock()
            mock_cifar.return_value = mock_dataset

            # Mock DataLoader
            with patch("evaluate.torch.utils.data.DataLoader") as mock_dataloader:
                mock_dataloader.return_value = mock_test_loader

                metrics = evaluate_model(model, str(tmp_path), batch_size=8)

        assert "test_loss" in metrics
        assert "test_accuracy" in metrics
        assert "total_samples" in metrics
        assert "correct_predictions" in metrics
        assert "per_class_accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert "class_names" in metrics

    def test_evaluate_model_accuracy_range(self, mock_test_loader, tmp_path):
        """Test that accuracy is between 0 and 1."""
        model = CIFAR10CNN(num_classes=10)

        with patch("evaluate.datasets.CIFAR10") as mock_cifar:
            mock_dataset = MagicMock()
            mock_cifar.return_value = mock_dataset

            with patch("evaluate.torch.utils.data.DataLoader") as mock_dataloader:
                mock_dataloader.return_value = mock_test_loader

                metrics = evaluate_model(model, str(tmp_path), batch_size=8)

        assert 0.0 <= metrics["test_accuracy"] <= 1.0


class TestDVCPipelineConfig:
    """Tests for the DVC pipeline configuration file."""

    @pytest.fixture
    def dvc_yaml_path(self):
        """Return path to dvc.yaml file."""
        return Path(__file__).parent.parent / "project" / "dvc.yaml"

    def test_dvc_yaml_exists(self, dvc_yaml_path):
        """Test that dvc.yaml exists."""
        assert dvc_yaml_path.exists(), f"dvc.yaml not found at {dvc_yaml_path}"

    def test_dvc_yaml_valid_yaml(self, dvc_yaml_path):
        """Test that dvc.yaml is valid YAML."""
        with open(dvc_yaml_path) as f:
            config = yaml.safe_load(f)
        assert config is not None

    def test_dvc_yaml_has_stages(self, dvc_yaml_path):
        """Test that dvc.yaml has stages defined."""
        with open(dvc_yaml_path) as f:
            config = yaml.safe_load(f)
        assert "stages" in config

    def test_dvc_yaml_prepare_data_stage(self, dvc_yaml_path):
        """Test prepare_data stage configuration."""
        with open(dvc_yaml_path) as f:
            config = yaml.safe_load(f)

        assert "prepare_data" in config["stages"]
        stage = config["stages"]["prepare_data"]
        assert "cmd" in stage
        assert "deps" in stage
        assert "outs" in stage
        assert "prepare_data.py" in stage["deps"]

    def test_dvc_yaml_train_stage(self, dvc_yaml_path):
        """Test train stage configuration."""
        with open(dvc_yaml_path) as f:
            config = yaml.safe_load(f)

        assert "train" in config["stages"]
        stage = config["stages"]["train"]
        assert "cmd" in stage
        assert "deps" in stage
        assert "outs" in stage
        assert "params" in stage
        assert "train.py" in stage["deps"]

    def test_dvc_yaml_evaluate_stage(self, dvc_yaml_path):
        """Test evaluate stage configuration."""
        with open(dvc_yaml_path) as f:
            config = yaml.safe_load(f)

        assert "evaluate" in config["stages"]
        stage = config["stages"]["evaluate"]
        assert "cmd" in stage
        assert "deps" in stage
        assert "metrics" in stage
        assert "evaluate.py" in stage["deps"]
        assert "models/best_model.pt" in stage["deps"]

    def test_dvc_yaml_stage_dependencies(self, dvc_yaml_path):
        """Test that stage dependencies are properly defined."""
        with open(dvc_yaml_path) as f:
            config = yaml.safe_load(f)

        # train should depend on data from prepare_data
        train_deps = config["stages"]["train"]["deps"]
        assert "data/cifar-10-batches-py" in train_deps

        # evaluate should depend on model from train
        evaluate_deps = config["stages"]["evaluate"]["deps"]
        assert "models/best_model.pt" in evaluate_deps

    def test_dvc_yaml_train_params(self, dvc_yaml_path):
        """Test that train stage has proper parameter tracking."""
        with open(dvc_yaml_path) as f:
            config = yaml.safe_load(f)

        params = config["stages"]["train"]["params"]
        # Should track Hydra config parameters
        assert len(params) > 0

    def test_dvc_yaml_metrics_config(self, dvc_yaml_path):
        """Test that metrics are configured correctly."""
        with open(dvc_yaml_path) as f:
            config = yaml.safe_load(f)

        evaluate_metrics = config["stages"]["evaluate"]["metrics"]
        assert any("metrics.json" in str(m) for m in evaluate_metrics)


class TestDVCIgnore:
    """Tests for .dvcignore file."""

    @pytest.fixture
    def dvcignore_path(self):
        """Return path to .dvcignore file."""
        return Path(__file__).parent.parent / "project" / ".dvcignore"

    def test_dvcignore_exists(self, dvcignore_path):
        """Test that .dvcignore exists."""
        assert dvcignore_path.exists(), f".dvcignore not found at {dvcignore_path}"

    def test_dvcignore_excludes_pycache(self, dvcignore_path):
        """Test that .dvcignore excludes Python cache."""
        with open(dvcignore_path) as f:
            content = f.read()
        assert "__pycache__" in content

    def test_dvcignore_excludes_venv(self, dvcignore_path):
        """Test that .dvcignore excludes virtual environments."""
        with open(dvcignore_path) as f:
            content = f.read()
        assert "venv" in content or ".venv" in content

    def test_dvcignore_excludes_hydra_outputs(self, dvcignore_path):
        """Test that .dvcignore excludes Hydra outputs."""
        with open(dvcignore_path) as f:
            content = f.read()
        assert "outputs/" in content or ".hydra/" in content
