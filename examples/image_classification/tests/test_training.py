"""Tests for the training module with CIFAR-10 and Hydra."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from train import (
    CIFAR10CNN,
    create_cifar10_loaders,
    get_cifar10_transforms,
    train,
    train_epoch,
    validate,
)


class TestCIFAR10CNN:
    """Tests for the CIFAR10CNN model."""

    def test_model_creation(self):
        """Test model can be created with default parameters."""
        model = CIFAR10CNN()
        assert model.num_classes == 10

    def test_model_with_custom_classes(self):
        """Test model with custom number of classes."""
        model = CIFAR10CNN(num_classes=5, dropout=0.3)
        assert model.num_classes == 5

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = CIFAR10CNN(num_classes=10)
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        output = model(x)
        assert output.shape == (batch_size, 10)

    def test_forward_pass_different_classes(self):
        """Test forward pass with different number of classes."""
        model = CIFAR10CNN(num_classes=5)
        batch_size = 8
        x = torch.randn(batch_size, 3, 32, 32)
        output = model(x)
        assert output.shape == (batch_size, 5)


class TestTransforms:
    """Tests for CIFAR-10 transforms."""

    def test_get_transforms(self):
        """Test transform creation."""
        train_transform, val_transform = get_cifar10_transforms(image_size=32)
        assert train_transform is not None
        assert val_transform is not None

    def test_transforms_different_size(self):
        """Test transforms with different image size."""
        train_transform, val_transform = get_cifar10_transforms(image_size=64)
        assert train_transform is not None
        assert val_transform is not None


class TestTrainingFunctions:
    """Tests for training helper functions."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock data loader for testing."""
        # Create synthetic data
        batch_size = 8
        num_batches = 4
        data = [
            (torch.randn(batch_size, 3, 32, 32), torch.randint(0, 10, (batch_size,)))
            for _ in range(num_batches)
        ]
        # Create a mock loader with dataset attribute
        loader = MagicMock()
        loader.__iter__ = MagicMock(return_value=iter(data))
        loader.__len__ = MagicMock(return_value=num_batches)
        loader.dataset = MagicMock()
        loader.dataset.__len__ = MagicMock(return_value=batch_size * num_batches)
        return loader

    @pytest.fixture
    def model_setup(self, mock_loader):
        """Create model and training components."""
        model = CIFAR10CNN(num_classes=10)
        device = torch.device("cpu")
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        return model, mock_loader, criterion, optimizer, device

    def test_train_epoch(self, model_setup):
        """Test single training epoch."""
        model, train_loader, criterion, optimizer, device = model_setup

        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_validate(self, model_setup):
        """Test validation function."""
        model, val_loader, criterion, _, device = model_setup

        loss, acc = validate(model, val_loader, criterion, device)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_model_changes_with_training(self, model_setup):
        """Test that model weights change with training."""
        model, train_loader, criterion, optimizer, device = model_setup

        # Get initial weights
        initial_weights = model.fc1.weight.clone().detach()

        # Train for one epoch
        train_epoch(model, train_loader, criterion, optimizer, device)

        # Weights should have changed
        assert not torch.allclose(initial_weights, model.fc1.weight)


class TestCIFAR10Loaders:
    """Tests for CIFAR-10 data loader creation."""

    def test_create_loaders(self, tmp_path):
        """Test CIFAR-10 loader creation with download."""
        # This test downloads CIFAR-10, so we mock it for CI
        with patch("train.datasets.CIFAR10") as mock_cifar:
            # Create mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_cifar.return_value = mock_dataset

            train_loader, val_loader, class_names = create_cifar10_loaders(
                data_dir=str(tmp_path / "data"),
                batch_size=32,
                image_size=32,
                num_workers=0,
            )

            assert len(class_names) == 10
            assert "airplane" in class_names
            assert "automobile" in class_names
            assert mock_cifar.called


class TestFullTraining:
    """Tests for the full training function."""

    @pytest.fixture
    def mock_cifar_data(self, tmp_path):
        """Mock CIFAR-10 download for testing."""
        # Create synthetic data that mimics CIFAR-10
        batch_size = 8
        num_batches = 2

        def create_mock_loader():
            data = [
                (torch.randn(batch_size, 3, 32, 32), torch.randint(0, 10, (batch_size,)))
                for _ in range(num_batches)
            ]
            loader = MagicMock()
            loader.__iter__ = MagicMock(return_value=iter(data))
            loader.__len__ = MagicMock(return_value=num_batches)
            loader.dataset = MagicMock()
            loader.dataset.__len__ = MagicMock(return_value=batch_size * num_batches)
            return loader

        return create_mock_loader

    def test_train_function(self, tmp_path, mock_cifar_data):
        """Test the main train function."""
        output_dir = str(tmp_path / "models")

        with patch("train.create_cifar10_loaders") as mock_loaders:
            class_names = [
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
            mock_loaders.return_value = (
                mock_cifar_data(),
                mock_cifar_data(),
                class_names,
            )

            results = train(
                data_dir=str(tmp_path / "data"),
                output_dir=output_dir,
                epochs=2,
                batch_size=8,
                num_workers=0,
            )

            assert "best_accuracy" in results
            assert "final_accuracy" in results
            assert "epochs_trained" in results
            assert "class_names" in results
            assert "model_path" in results
            assert "history" in results
            assert results["epochs_trained"] == 2
            assert len(results["class_names"]) == 10

    def test_train_saves_model(self, tmp_path, mock_cifar_data):
        """Test that training saves model files."""
        output_dir = str(tmp_path / "models")

        with patch("train.create_cifar10_loaders") as mock_loaders:
            class_names = [
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
            mock_loaders.return_value = (
                mock_cifar_data(),
                mock_cifar_data(),
                class_names,
            )

            train(
                data_dir=str(tmp_path / "data"),
                output_dir=output_dir,
                epochs=1,
                batch_size=8,
                num_workers=0,
            )

            output_path = Path(output_dir)
            assert (output_path / "best_model.pt").exists()
            assert (output_path / "final_model.pt").exists()
            assert (output_path / "classes.txt").exists()

    def test_train_saves_classes(self, tmp_path, mock_cifar_data):
        """Test that class names are saved correctly."""
        output_dir = str(tmp_path / "models")

        with patch("train.create_cifar10_loaders") as mock_loaders:
            class_names = [
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
            mock_loaders.return_value = (
                mock_cifar_data(),
                mock_cifar_data(),
                class_names,
            )

            train(
                data_dir=str(tmp_path / "data"),
                output_dir=output_dir,
                epochs=1,
                batch_size=8,
                num_workers=0,
            )

            classes_file = Path(output_dir) / "classes.txt"
            with open(classes_file) as f:
                saved_classes = [line.strip() for line in f.readlines()]

            assert len(saved_classes) == 10
            assert "airplane" in saved_classes
            assert "dog" in saved_classes

    def test_train_history(self, tmp_path, mock_cifar_data):
        """Test that training history is recorded."""
        output_dir = str(tmp_path / "models")

        with patch("train.create_cifar10_loaders") as mock_loaders:
            class_names = [
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
            mock_loaders.return_value = (
                mock_cifar_data(),
                mock_cifar_data(),
                class_names,
            )

            results = train(
                data_dir=str(tmp_path / "data"),
                output_dir=output_dir,
                epochs=3,
                batch_size=8,
                num_workers=0,
            )

            history = results["history"]
            assert len(history["train_loss"]) == 3
            assert len(history["train_acc"]) == 3
            assert len(history["val_loss"]) == 3
            assert len(history["val_acc"]) == 3

    def test_train_with_seed(self, tmp_path, mock_cifar_data):
        """Test that training respects the random seed."""
        output_dir = str(tmp_path / "models")

        with patch("train.create_cifar10_loaders") as mock_loaders:
            class_names = [
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
            mock_loaders.return_value = (
                mock_cifar_data(),
                mock_cifar_data(),
                class_names,
            )

            results = train(
                data_dir=str(tmp_path / "data"),
                output_dir=output_dir,
                epochs=1,
                batch_size=8,
                num_workers=0,
                seed=123,
            )

            assert results is not None
