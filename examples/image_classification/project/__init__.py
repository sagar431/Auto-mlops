"""Image classification example project for Auto-MLOps."""

from .dataset import (
    ImageClassificationDataset,
    create_data_loaders,
    create_synthetic_data,
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
    "create_synthetic_data",
    "get_transforms",
    "ImageClassifier",
]
