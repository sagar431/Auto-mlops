"""Evaluate trained image classification model.

This script evaluates a trained model on the test set and outputs metrics.
Used as the evaluation stage in the DVC pipeline.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from train import CIFAR10CNN

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_test_transforms(image_size: int = 32) -> transforms.Compose:
    """Get transforms for test data."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )


def load_model(model_path: str, num_classes: int = 10, dropout: float = 0.5) -> nn.Module:
    """Load trained model from checkpoint."""
    model = CIFAR10CNN(num_classes=num_classes, dropout=dropout)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def evaluate_model(
    model: nn.Module,
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 32,
) -> dict:
    """Evaluate model on test set.

    Args:
        model: Trained model
        data_dir: Directory containing CIFAR-10 data
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        image_size: Input image size

    Returns:
        dict with evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_transform = get_test_transforms(image_size)
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

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

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    # Per-class metrics
    class_correct = [0] * 10
    class_total = [0] * 10

    # Confusion matrix
    confusion_matrix = [[0] * 10 for _ in range(10)]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                confusion_matrix[label][pred] += 1

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total

    # Per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            per_class_accuracy[class_name] = class_correct[i] / class_total[i]
        else:
            per_class_accuracy[class_name] = 0.0

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix,
        "class_names": class_names,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained image classification model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to trained model checkpoint (default: models/best_model.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing CIFAR-10 data (default: data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.json",
        help="Path to write metrics JSON (default: metrics.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    args = parser.parse_args()

    log.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)

    log.info("Evaluating model on test set")
    metrics = evaluate_model(
        model,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    log.info(f"Test Loss: {metrics['test_loss']:.4f}")
    log.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    log.info("Per-class accuracy:")
    for class_name, acc in metrics["per_class_accuracy"].items():
        log.info(f"  {class_name}: {acc:.4f}")

    # Write metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
