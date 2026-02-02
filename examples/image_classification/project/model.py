"""ResNet18 model for image classification."""

import torch
import torch.nn as nn
from torchvision import models


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
