"""Dataset utilities for image classification."""

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# CIFAR-10 class names
CIFAR10_CLASSES = [
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

# CIFAR-10 normalization values (computed from dataset)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


class ImageClassificationDataset(Dataset):
    """Custom dataset for image classification.

    Expects directory structure:
    data/
      train/
        class_0/
          image1.jpg
          image2.jpg
        class_1/
          image3.jpg
          image4.jpg
    """

    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose | None = None,
        split: str = "train",
    ):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = sorted(
            [d for d in os.listdir(self.root_dir) if (self.root_dir / d).is_dir()]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((class_dir / img_name, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """Get image transforms for training or evaluation."""
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    train_split: float = 0.8,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, list]:
    """Create train and validation data loaders.

    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)

    # Check if we have separate train/val directories
    data_path = Path(data_dir)
    if (data_path / "train").exists() and (data_path / "val").exists():
        train_dataset = ImageClassificationDataset(data_dir, train_transform, "train")
        val_dataset = ImageClassificationDataset(data_dir, val_transform, "val")
        class_names = train_dataset.classes
    else:
        # Single directory - split into train/val
        full_dataset = ImageClassificationDataset(data_dir, train_transform, "train")
        class_names = full_dataset.classes

        train_size = int(len(full_dataset) * train_split)
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Apply val transform to validation set
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names


def create_synthetic_data(data_dir: str, num_samples: int = 100, num_classes: int = 2):
    """Create synthetic dataset for testing.

    Creates random colored images for each class.
    """
    import numpy as np

    data_path = Path(data_dir) / "train"
    class_names = ["cat", "dog"] if num_classes == 2 else [f"class_{i}" for i in range(num_classes)]

    for class_name in class_names:
        class_dir = data_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples // num_classes):
            # Create a random image with class-specific color bias
            img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

            # Add some class-specific pattern
            if class_name == "cat":
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 50, 0, 255)  # Red bias
            elif class_name == "dog":
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] + 50, 0, 255)  # Blue bias

            img = Image.fromarray(img_array)
            img.save(class_dir / f"{class_name}_{i:04d}.png")

    return class_names


def get_cifar10_transforms(image_size: int = 32, is_training: bool = True) -> transforms.Compose:
    """Get CIFAR-10 specific transforms for training or evaluation.

    Args:
        image_size: Target image size (default 32 for CIFAR-10).
        is_training: Whether to apply training augmentations.

    Returns:
        Composed transforms for CIFAR-10.
    """
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )


def create_cifar10_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 32,
    num_workers: int = 4,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create CIFAR-10 train and validation data loaders.

    Args:
        data_dir: Directory to store/load CIFAR-10 data.
        batch_size: Batch size for data loaders.
        image_size: Target image size (default 32 for CIFAR-10).
        num_workers: Number of workers for data loading.
        download: Whether to download the dataset if not found.

    Returns:
        Tuple of (train_loader, val_loader, class_names).
    """
    train_transform = get_cifar10_transforms(image_size, is_training=True)
    val_transform = get_cifar10_transforms(image_size, is_training=False)

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, CIFAR10_CLASSES
