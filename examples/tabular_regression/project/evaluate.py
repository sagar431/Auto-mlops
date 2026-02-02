"""Evaluation script for tabular regression models."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from dataset import TabularDataset, load_data
from model import load_model
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute regression metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Dictionary of metrics
    """
    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))

    # R-squared (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Mean Absolute Percentage Error (avoid division by zero)
    mask = targets != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }


def evaluate(
    model_path: str,
    data_dir: str,
    batch_size: int = 128,
    device: str = None,
) -> dict:
    """Evaluate a trained model on test data.

    Args:
        model_path: Path to model checkpoint
        data_dir: Directory containing test.csv
        batch_size: Batch size for evaluation
        device: Device to use (auto-detected if None)

    Returns:
        Dictionary of metrics
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # Load model
    model = load_model(model_path, device=device)
    log.info("Loaded model from checkpoint")

    # Load test data
    features, targets, feature_names = load_data(data_dir, split="test")
    log.info(f"Loaded {len(features)} test samples")

    # Create dataloader
    dataset = TabularDataset(features, targets, feature_names)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions
    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_features, batch_targets in dataloader:
            batch_features = batch_features.to(device)
            predictions = model(batch_features).cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(batch_targets.numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    metrics["samples"] = len(targets)

    log.info("Evaluation results:")
    log.info(f"  RMSE: {metrics['rmse']:.4f}")
    log.info(f"  MAE: {metrics['mae']:.4f}")
    log.info(f"  R²: {metrics['r2']:.4f}")
    log.info(f"  MAPE: {metrics['mape']:.2f}%")

    return metrics


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(description="Evaluate tabular regression model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.json",
        help="Output path for metrics JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu or cuda)",
    )

    args = parser.parse_args()

    # Evaluate
    metrics = evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Save metrics
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
