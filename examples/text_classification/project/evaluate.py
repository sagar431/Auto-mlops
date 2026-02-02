"""Evaluate trained text classification model.

This script evaluates a trained model on the test set and outputs metrics.
Used as the evaluation stage in the DVC pipeline.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from dataset import (
    SENTIMENT_CLASSES,
    TextClassificationDataset,
    Vocabulary,
    collate_fn,
    create_synthetic_data,
    load_imdb_data,
)
from model import create_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_model_for_eval(
    model_path: str,
    vocab_size: int,
    num_classes: int = 2,
    model_type: str = "textcnn",
    **model_kwargs,
) -> nn.Module:
    """Load trained model from checkpoint."""
    model = create_model(
        model_type=model_type,
        vocab_size=vocab_size,
        num_classes=num_classes,
        **model_kwargs,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def evaluate_model(
    model: nn.Module,
    vocab: Vocabulary,
    test_texts: list[str],
    test_labels: list[int],
    class_names: list[str],
    batch_size: int = 128,
    num_workers: int = 4,
    max_length: int = 256,
) -> dict:
    """Evaluate model on test set.

    Args:
        model: Trained model.
        vocab: Vocabulary for encoding.
        test_texts: List of test texts.
        test_labels: List of test labels.
        class_names: List of class names.
        batch_size: Batch size for evaluation.
        num_workers: Number of data loading workers.
        max_length: Maximum sequence length.

    Returns:
        dict with evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create test dataset and loader
    test_dataset = TextClassificationDataset(test_texts, test_labels, vocab, max_length)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx),
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    num_classes = len(class_names)

    # Per-class metrics
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # Confusion matrix
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
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

    # Precision, recall, F1
    precision = {}
    recall = {}
    f1 = {}
    for i, class_name in enumerate(class_names):
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[j][i] for j in range(num_classes)) - tp
        fn = sum(confusion_matrix[i][j] for j in range(num_classes)) - tp

        precision[class_name] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[class_name] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision[class_name] + recall[class_name] > 0:
            f1[class_name] = (
                2
                * precision[class_name]
                * recall[class_name]
                / (precision[class_name] + recall[class_name])
            )
        else:
            f1[class_name] = 0.0

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "per_class_accuracy": per_class_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix,
        "class_names": class_names,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained text classification model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to trained model checkpoint (default: models/best_model.pt)",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="models/vocab.json",
        help="Path to vocabulary file (default: models/vocab.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing data (default: data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.json",
        help="Path to write metrics JSON (default: metrics.json)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="textcnn",
        help="Model type (default: textcnn)",
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
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    args = parser.parse_args()

    # Load vocabulary
    log.info(f"Loading vocabulary from {args.vocab_path}")
    vocab = Vocabulary.load(args.vocab_path)

    # Load or create test data
    if args.use_synthetic:
        log.info("Using synthetic data")
        _, _, test_texts, test_labels = None, None, *create_synthetic_data(200, 2)
        class_names = ["class_0", "class_1"]
    else:
        data_path = Path(args.data_dir)
        if (data_path / "test").exists():
            log.info(f"Loading IMDB data from {args.data_dir}")
            _, _, test_texts, test_labels = load_imdb_data(args.data_dir)
            class_names = SENTIMENT_CLASSES
        else:
            log.warning(f"No data found in {args.data_dir}, using synthetic data")
            test_texts, test_labels = create_synthetic_data(200, 2)
            class_names = ["class_0", "class_1"]

    log.info(f"Loading model from {args.model_path}")
    model = load_model_for_eval(
        args.model_path,
        vocab_size=len(vocab),
        num_classes=len(class_names),
        model_type=args.model_type,
    )

    log.info("Evaluating model on test set")
    metrics = evaluate_model(
        model,
        vocab,
        test_texts,
        test_labels,
        class_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    log.info(f"Test Loss: {metrics['test_loss']:.4f}")
    log.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    log.info("Per-class metrics:")
    for class_name in class_names:
        log.info(
            f"  {class_name}: "
            f"accuracy={metrics['per_class_accuracy'][class_name]:.4f}, "
            f"precision={metrics['precision'][class_name]:.4f}, "
            f"recall={metrics['recall'][class_name]:.4f}, "
            f"f1={metrics['f1'][class_name]:.4f}"
        )

    # Write metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
