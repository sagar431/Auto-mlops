"""Image classification example project for Auto-MLOps."""

from .dataset import (
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
from .inference import ImageClassifier
from .model import SimpleCNN, create_model, load_model

__all__ = [
    "SimpleCNN",
    "create_model",
    "load_model",
    "ImageClassificationDataset",
    "create_data_loaders",
    "create_cifar10_loaders",
    "create_synthetic_data",
    "get_transforms",
    "get_cifar10_transforms",
    "CIFAR10_CLASSES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "ImageClassifier",
]
