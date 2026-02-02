"""Prepare data for text classification training.

This script downloads dataset and prepares it for training.
Used as the first stage in the DVC pipeline.
"""

import argparse
import json
import logging
import os
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def download_imdb(data_dir: str, output_file: str) -> dict:
    """Download and prepare IMDB movie review dataset.

    Args:
        data_dir: Directory to store the dataset.
        output_file: Path to write data info JSON.

    Returns:
        dict with dataset information.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    archive_path = data_path / "aclImdb_v1.tar.gz"
    extracted_path = data_path / "aclImdb"

    # Download if not exists
    if not extracted_path.exists():
        if not archive_path.exists():
            log.info(f"Downloading IMDB dataset to {archive_path}")
            urlretrieve(IMDB_URL, archive_path)
            log.info("Download complete")

        log.info("Extracting archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(data_path)
        log.info("Extraction complete")

        # Move to expected structure
        # aclImdb/train/pos, aclImdb/train/neg -> data/train/pos, data/train/neg
        for split in ["train", "test"]:
            for label in ["pos", "neg"]:
                src = extracted_path / split / label
                dst = data_path / split / label
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(src, dst)

    # Count samples
    train_pos = len(list((data_path / "train" / "pos").glob("*.txt")))
    train_neg = len(list((data_path / "train" / "neg").glob("*.txt")))
    test_pos = len(list((data_path / "test" / "pos").glob("*.txt")))
    test_neg = len(list((data_path / "test" / "neg").glob("*.txt")))

    class_names = ["negative", "positive"]

    data_info = {
        "dataset": "imdb",
        "train_samples": train_pos + train_neg,
        "test_samples": test_pos + test_neg,
        "train_pos": train_pos,
        "train_neg": train_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
        "num_classes": len(class_names),
        "class_names": class_names,
        "data_dir": str(data_path.resolve()),
    }

    log.info(f"Training samples: {data_info['train_samples']} (pos: {train_pos}, neg: {train_neg})")
    log.info(f"Test samples: {data_info['test_samples']} (pos: {test_pos}, neg: {test_neg})")
    log.info(f"Classes: {class_names}")

    # Write data info
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data_info, f, indent=2)

    log.info(f"Data info written to {output_path}")

    return data_info


def create_synthetic_dataset(
    data_dir: str, output_file: str, num_train: int = 1000, num_test: int = 200
) -> dict:
    """Create synthetic dataset for testing without downloading.

    Args:
        data_dir: Directory to store the dataset.
        output_file: Path to write data info JSON.
        num_train: Number of training samples.
        num_test: Number of test samples.

    Returns:
        dict with dataset information.
    """
    import random

    data_path = Path(data_dir)

    # Create positive and negative templates
    positive_words = [
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "love",
        "best",
        "perfect",
        "brilliant",
    ]
    negative_words = [
        "bad",
        "terrible",
        "awful",
        "horrible",
        "worst",
        "hate",
        "boring",
        "waste",
        "poor",
    ]
    neutral_words = [
        "movie",
        "film",
        "story",
        "acting",
        "plot",
        "character",
        "scene",
        "director",
        "cast",
    ]

    def generate_review(is_positive: bool, length: int = 50) -> str:
        words = []
        sentiment_words = positive_words if is_positive else negative_words
        for _ in range(length):
            if random.random() < 0.3:
                words.append(random.choice(sentiment_words))
            else:
                words.append(random.choice(neutral_words))
        return " ".join(words)

    for split, num_samples in [("train", num_train), ("test", num_test)]:
        for label, is_positive in [("neg", False), ("pos", True)]:
            split_dir = data_path / split / label
            split_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_samples // 2):
                review = generate_review(is_positive, random.randint(30, 100))
                with open(split_dir / f"{i}.txt", "w") as f:
                    f.write(review)

    class_names = ["negative", "positive"]

    data_info = {
        "dataset": "synthetic",
        "train_samples": num_train,
        "test_samples": num_test,
        "num_classes": len(class_names),
        "class_names": class_names,
        "data_dir": str(data_path.resolve()),
    }

    log.info(f"Created synthetic dataset with {num_train} training and {num_test} test samples")

    # Write data info
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data_info, f, indent=2)

    log.info(f"Data info written to {output_path}")

    return data_info


def main():
    parser = argparse.ArgumentParser(description="Prepare data for text classification")
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
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create synthetic dataset instead of downloading IMDB",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=1000,
        help="Number of training samples for synthetic data (default: 1000)",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=200,
        help="Number of test samples for synthetic data (default: 200)",
    )
    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_dataset(
            args.data_dir,
            args.output,
            num_train=args.num_train,
            num_test=args.num_test,
        )
    else:
        download_imdb(args.data_dir, args.output)


if __name__ == "__main__":
    main()
