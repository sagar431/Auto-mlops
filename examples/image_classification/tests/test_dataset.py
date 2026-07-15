"""Tests for the dataset utilities."""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from dataset import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    ImageClassificationDataset,
    create_cifar10_loaders,
    create_data_loaders,
    create_synthetic_data,
    get_cifar10_transforms,
    get_transforms,
)


class TestTransforms:
    """Tests for image transforms."""

    def test_training_transforms(self):
        """Test training transforms produce correct output shape."""
        transform = get_transforms(image_size=224, is_training=True)
        image = Image.new("RGB", (100, 100), color="red")

        result = transform(image)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_eval_transforms(self):
        """Test evaluation transforms produce correct output shape."""
        transform = get_transforms(image_size=224, is_training=False)
        image = Image.new("RGB", (100, 100), color="blue")

        result = transform(image)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_transforms_normalization(self):
        """Test that transforms normalize the image."""
        transform = get_transforms(image_size=224, is_training=False)
        # Create a white image (all 255)
        image = Image.new("RGB", (100, 100), color=(255, 255, 255))

        result = transform(image)

        # After normalization, values should not be in [0, 1] range
        # ImageNet normalization: (pixel - mean) / std
        assert result.max() > 1.0 or result.min() < 0.0


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_create_synthetic_data(self, tmp_path):
        """Test synthetic data creation."""
        data_dir = str(tmp_path / "data")
        classes = create_synthetic_data(data_dir, num_samples=20, num_classes=2)

        assert len(classes) == 2
        assert "cat" in classes
        assert "dog" in classes

        # Check files were created
        train_dir = tmp_path / "data" / "train"
        assert train_dir.exists()

        cat_images = list((train_dir / "cat").glob("*.png"))
        dog_images = list((train_dir / "dog").glob("*.png"))

        assert len(cat_images) == 10  # 20 samples / 2 classes
        assert len(dog_images) == 10

    def test_synthetic_images_are_valid(self, tmp_path):
        """Test that synthetic images can be opened and read."""
        data_dir = str(tmp_path / "data")
        create_synthetic_data(data_dir, num_samples=4, num_classes=2)

        train_dir = tmp_path / "data" / "train" / "cat"
        images = list(train_dir.glob("*.png"))

        for img_path in images:
            img = Image.open(img_path)
            assert img.mode == "RGB"
            assert img.size == (64, 64)


class TestImageClassificationDataset:
    """Tests for ImageClassificationDataset."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset for testing."""
        data_dir = str(tmp_path / "data")
        create_synthetic_data(data_dir, num_samples=20, num_classes=2)
        return data_dir, tmp_path / "data"

    def test_dataset_length(self, sample_dataset):
        """Test dataset reports correct length."""
        data_dir, _ = sample_dataset
        transform = get_transforms(image_size=224, is_training=False)

        dataset = ImageClassificationDataset(data_dir, transform, split="train")

        assert len(dataset) == 20

    def test_dataset_classes(self, sample_dataset):
        """Test dataset identifies classes correctly."""
        data_dir, _ = sample_dataset
        transform = get_transforms(image_size=224, is_training=False)

        dataset = ImageClassificationDataset(data_dir, transform, split="train")

        assert len(dataset.classes) == 2
        assert "cat" in dataset.classes
        assert "dog" in dataset.classes

    def test_dataset_getitem(self, sample_dataset):
        """Test dataset returns correct item format."""
        data_dir, _ = sample_dataset
        transform = get_transforms(image_size=224, is_training=False)

        dataset = ImageClassificationDataset(data_dir, transform, split="train")
        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert label in [0, 1]

    def test_dataset_class_to_idx(self, sample_dataset):
        """Test class to index mapping."""
        data_dir, _ = sample_dataset
        transform = get_transforms(image_size=224, is_training=False)

        dataset = ImageClassificationDataset(data_dir, transform, split="train")

        assert len(dataset.class_to_idx) == 2
        for cls, idx in dataset.class_to_idx.items():
            assert cls in ["cat", "dog"]
            assert idx in [0, 1]


class TestDataLoaders:
    """Tests for data loader creation."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data for testing."""
        data_dir = str(tmp_path / "data")
        create_synthetic_data(data_dir, num_samples=40, num_classes=2)
        return data_dir

    def test_create_data_loaders(self, sample_data):
        """Test data loader creation."""
        train_loader, val_loader, class_names = create_data_loaders(
            sample_data, batch_size=8, train_split=0.8, num_workers=0
        )

        assert len(class_names) == 2
        assert len(train_loader.dataset) == 32  # 80% of 40
        assert len(val_loader.dataset) == 8  # 20% of 40

    def test_data_loader_iteration(self, sample_data):
        """Test iterating through data loader."""
        train_loader, _, _ = create_data_loaders(sample_data, batch_size=8, num_workers=0)

        batch = next(iter(train_loader))
        images, labels = batch

        assert images.shape[0] <= 8
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[0] == images.shape[0]


class TestCIFAR10Constants:
    """Tests for CIFAR-10 constants."""

    def test_cifar10_classes_count(self):
        """Test CIFAR-10 has 10 classes."""
        assert len(CIFAR10_CLASSES) == 10

    def test_cifar10_classes_content(self):
        """Test CIFAR-10 class names are correct."""
        expected = [
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
        assert CIFAR10_CLASSES == expected

    def test_cifar10_mean_values(self):
        """Test CIFAR-10 mean normalization values."""
        assert len(CIFAR10_MEAN) == 3
        # Check approximate expected values
        assert abs(CIFAR10_MEAN[0] - 0.4914) < 0.01
        assert abs(CIFAR10_MEAN[1] - 0.4822) < 0.01
        assert abs(CIFAR10_MEAN[2] - 0.4465) < 0.01

    def test_cifar10_std_values(self):
        """Test CIFAR-10 std normalization values."""
        assert len(CIFAR10_STD) == 3
        # Check approximate expected values
        assert abs(CIFAR10_STD[0] - 0.2470) < 0.01
        assert abs(CIFAR10_STD[1] - 0.2435) < 0.01
        assert abs(CIFAR10_STD[2] - 0.2616) < 0.01


class TestCIFAR10Transforms:
    """Tests for CIFAR-10 specific transforms."""

    def test_cifar10_training_transforms(self):
        """Test CIFAR-10 training transforms produce correct output shape."""
        transform = get_cifar10_transforms(image_size=32, is_training=True)
        image = Image.new("RGB", (32, 32), color="red")

        result = transform(image)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 32)

    def test_cifar10_eval_transforms(self):
        """Test CIFAR-10 evaluation transforms produce correct output shape."""
        transform = get_cifar10_transforms(image_size=32, is_training=False)
        image = Image.new("RGB", (32, 32), color="blue")

        result = transform(image)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 32)

    def test_cifar10_transforms_normalization(self):
        """Test that CIFAR-10 transforms normalize the image."""
        transform = get_cifar10_transforms(image_size=32, is_training=False)
        # Create a white image (all 255)
        image = Image.new("RGB", (32, 32), color=(255, 255, 255))

        result = transform(image)

        # After normalization, values should not be in [0, 1] range
        assert result.max() > 1.0 or result.min() < 0.0

    def test_cifar10_transforms_custom_size(self):
        """Test CIFAR-10 transforms with custom image size."""
        transform = get_cifar10_transforms(image_size=64, is_training=False)
        image = Image.new("RGB", (32, 32), color="green")

        result = transform(image)

        assert result.shape == (3, 64, 64)


class TestCIFAR10DataLoaders:
    """Tests for CIFAR-10 data loader creation."""

    @pytest.fixture(autouse=True)
    def _offline_cifar10(self, monkeypatch):
        """Exercise loader behavior without downloading the external dataset."""

        class OfflineCIFAR10(torch.utils.data.Dataset):
            def __init__(self, root, train, download, transform):
                self.root = root
                self.train = train
                self.download = download
                self.transform = transform
                self.sample_count = 50_000 if train else 10_000

            def __len__(self):
                return self.sample_count

            def __getitem__(self, index):
                return torch.zeros(3, 32, 32), index % len(CIFAR10_CLASSES)

        monkeypatch.setattr("dataset.datasets.CIFAR10", OfflineCIFAR10)

    def test_create_cifar10_loaders(self, tmp_path):
        """Test CIFAR-10 data loader creation."""
        data_dir = str(tmp_path / "cifar10_data")

        train_loader, val_loader, class_names = create_cifar10_loaders(
            data_dir, batch_size=64, image_size=32, num_workers=0, download=True
        )

        assert class_names == CIFAR10_CLASSES
        assert len(train_loader.dataset) == 50000  # CIFAR-10 train size
        assert len(val_loader.dataset) == 10000  # CIFAR-10 test size

    def test_cifar10_loader_iteration(self, tmp_path):
        """Test iterating through CIFAR-10 data loader."""
        data_dir = str(tmp_path / "cifar10_data")

        train_loader, _, _ = create_cifar10_loaders(
            data_dir, batch_size=32, num_workers=0, download=True
        )

        batch = next(iter(train_loader))
        images, labels = batch

        assert images.shape == (32, 3, 32, 32)
        assert labels.shape == (32,)
        assert labels.min() >= 0
        assert labels.max() <= 9

    def test_cifar10_loader_batch_size(self, tmp_path):
        """Test CIFAR-10 loader respects batch size."""
        data_dir = str(tmp_path / "cifar10_data")

        train_loader, _, _ = create_cifar10_loaders(
            data_dir, batch_size=16, num_workers=0, download=True
        )

        batch = next(iter(train_loader))
        images, _ = batch

        assert images.shape[0] == 16
