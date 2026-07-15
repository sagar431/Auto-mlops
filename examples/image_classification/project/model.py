"""Model definitions and checkpoint loading for image classification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

GOLDEN_SCHEMA_VERSION = "golden-image-classifier.v1"
GOLDEN_ARCHITECTURE = "tiny_color_cnn_v1"


class CheckpointError(ValueError):
    """Raised when a model checkpoint violates the golden-slice contract."""


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


class TinyColorCNN(nn.Module):
    """Small CPU-friendly CNN used by the golden synthetic slice."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Validated model and metadata loaded from a canonical checkpoint."""

    model: nn.Module
    schema_version: str
    architecture: str
    class_names: tuple[str, ...]
    image_size: int
    normalization: dict[str, list[float]]
    training_config: dict[str, Any]
    metrics: dict[str, float]
    dataset_lineage: dict[str, Any]


def create_golden_model(num_classes: int) -> TinyColorCNN:
    """Build the only architecture accepted by the golden checkpoint schema."""
    if num_classes < 2:
        raise ValueError("Golden model requires at least two classes")
    return TinyColorCNN(num_classes=num_classes)


def _validate_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise CheckpointError("Checkpoint must be a dictionary")
    if checkpoint.get("schema_version") != GOLDEN_SCHEMA_VERSION:
        raise CheckpointError("Unsupported checkpoint schema version")
    if checkpoint.get("architecture") != GOLDEN_ARCHITECTURE:
        raise CheckpointError("Incompatible checkpoint architecture")

    class_names = checkpoint.get("class_names")
    num_classes = checkpoint.get("num_classes")
    if (
        not isinstance(class_names, list)
        or len(class_names) < 2
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
        or len(set(class_names)) != len(class_names)
        or num_classes != len(class_names)
    ):
        raise CheckpointError("Invalid checkpoint class metadata")

    image_size = checkpoint.get("image_size")
    normalization = checkpoint.get("normalization")
    if not isinstance(image_size, int) or image_size <= 0:
        raise CheckpointError("Invalid checkpoint image size")
    if not isinstance(normalization, dict):
        raise CheckpointError("Invalid checkpoint normalization metadata")
    for key in ("mean", "std"):
        values = normalization.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(not isinstance(value, (int, float)) for value in values)
        ):
            raise CheckpointError("Invalid checkpoint normalization metadata")
    if any(float(value) <= 0 for value in normalization["std"]):
        raise CheckpointError("Invalid checkpoint normalization standard deviation")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise CheckpointError("Checkpoint state dictionary is missing or invalid")
    if not isinstance(checkpoint.get("training_config"), dict):
        raise CheckpointError("Checkpoint training configuration is missing")
    if not isinstance(checkpoint.get("metrics"), dict):
        raise CheckpointError("Checkpoint metrics are missing")
    dataset_lineage = checkpoint.get("dataset_lineage")
    if (
        not isinstance(dataset_lineage, dict)
        or not isinstance(dataset_lineage.get("schema_version"), str)
        or not isinstance(dataset_lineage.get("source"), str)
        or not _is_sha256(dataset_lineage.get("dataset_checksum"))
    ):
        raise CheckpointError("Checkpoint dataset lineage is missing or invalid")
    if dataset_lineage["source"] == "dvc-materialized-image-files":
        if (
            not _is_sha256(dataset_lineage.get("manifest_checksum"))
            or not isinstance(dataset_lineage.get("file_checksums"), dict)
            or not dataset_lineage["file_checksums"]
            or any(
                not isinstance(path, str) or not path or not _is_sha256(checksum)
                for path, checksum in dataset_lineage.get("file_checksums", {}).items()
            )
            or not isinstance(dataset_lineage.get("sample_counts"), dict)
            or set(dataset_lineage.get("sample_counts", {})) != {"train", "validation"}
            or any(
                not isinstance(count, int) or count < 1
                for count in dataset_lineage.get("sample_counts", {}).values()
            )
        ):
            raise CheckpointError("Checkpoint file-backed dataset lineage is invalid")
    return checkpoint


def load_golden_checkpoint(path: str | Path, device: str = "cpu") -> LoadedCheckpoint:
    """Load and strictly validate a golden-slice checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Golden model checkpoint was not found")
    try:
        raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise CheckpointError("Unable to read golden model checkpoint") from exc
    checkpoint = _validate_checkpoint(raw_checkpoint)
    model = create_golden_model(checkpoint["num_classes"])
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError("Checkpoint state dictionary is incompatible") from exc
    model.to(torch.device(device))
    model.eval()
    return LoadedCheckpoint(
        model=model,
        schema_version=checkpoint["schema_version"],
        architecture=checkpoint["architecture"],
        class_names=tuple(checkpoint["class_names"]),
        image_size=checkpoint["image_size"],
        normalization={
            "mean": [float(value) for value in checkpoint["normalization"]["mean"]],
            "std": [float(value) for value in checkpoint["normalization"]["std"]],
        },
        training_config=dict(checkpoint["training_config"]),
        metrics={key: float(value) for key, value in checkpoint["metrics"].items()},
        dataset_lineage=dict(checkpoint["dataset_lineage"]),
    )


class ResNet18(nn.Module):
    """ResNet18 for image classification.

    Supports both CIFAR-10 (32x32) and ImageNet-sized (224x224) images.
    Can use pretrained ImageNet weights for transfer learning.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained ImageNet weights.
        input_size: Expected input image size (32 for CIFAR-10, 224 for ImageNet).
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = False,
        input_size: int = 32,
    ):
        super().__init__()
        from torchvision import models

        self.num_classes = num_classes
        self.input_size = input_size

        # Load pretrained ResNet18 or create from scratch
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.resnet = models.resnet18(weights=weights)
        else:
            self.resnet = models.resnet18(weights=None)

        # Modify first conv layer for CIFAR-10 (32x32 images)
        # Standard ResNet uses 7x7 conv with stride 2, which is too aggressive for small images
        if input_size <= 64:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove the max pooling layer for small images
            self.resnet.maxpool = nn.Identity()

        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


def create_model(
    num_classes: int = 10,
    pretrained: bool = False,
    input_size: int = 32,
    dropout: float = 0.5,  # Kept for backward compatibility, not used in ResNet18
) -> ResNet18:
    """Create and return a ResNet18 model.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained ImageNet weights.
        input_size: Expected input image size (32 for CIFAR-10, 224 for ImageNet).
        dropout: Unused, kept for backward compatibility with existing code.

    Returns:
        ResNet18 model instance.
    """
    return ResNet18(
        num_classes=num_classes,
        pretrained=pretrained,
        input_size=input_size,
    )


def load_model(
    path: str,
    num_classes: int = 10,
    device: str = "cpu",
    input_size: int = 32,
) -> ResNet18:
    """Load a trained model from a checkpoint file.

    Args:
        path: Path to the model checkpoint file.
        num_classes: Number of output classes.
        device: Device to load the model on.
        input_size: Expected input image size.

    Returns:
        Loaded ResNet18 model in eval mode.
    """
    model = ResNet18(num_classes=num_classes, pretrained=False, input_size=input_size)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
