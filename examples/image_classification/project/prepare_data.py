"""Prepare data for image classification training.

This script downloads CIFAR-10 dataset and prepares it for training.
Used as the first stage in the DVC pipeline.
"""

import argparse
import json
import logging
from pathlib import Path

from torchvision import datasets

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def prepare_cifar10(data_dir: str, output_file: str) -> dict:
    """Download and prepare CIFAR-10 dataset.

    Args:
        data_dir: Directory to store the dataset
        output_file: Path to write data info JSON

    Returns:
        dict with dataset information
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading CIFAR-10 to {data_path}")

    # Download training set
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)

    # Download test set
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)

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

    data_info = {
        "dataset": "cifar10",
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_size": [32, 32],
        "channels": 3,
        "data_dir": str(data_path.resolve()),
    }

    log.info(f"Training samples: {data_info['train_samples']}")
    log.info(f"Test samples: {data_info['test_samples']}")
    log.info(f"Classes: {class_names}")

    # Write data info
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data_info, f, indent=2)

    log.info(f"Data info written to {output_path}")

    return data_info


def main():
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 data for training")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store the dataset (default: data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/data_info.json",
        help="Path to write data info JSON (default: data/data_info.json)",
    )
    args = parser.parse_args()

    prepare_cifar10(args.data_dir, args.output)


if __name__ == "__main__":
    main()
