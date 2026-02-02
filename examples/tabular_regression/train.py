#!/usr/bin/env python
"""Sklearn-based training script for tabular regression on California Housing dataset.

This script provides a simple sklearn-based alternative to the PyTorch training
in project/train.py. It uses scikit-learn models (Ridge, RandomForest, GradientBoosting)
for training on the California Housing dataset.

Usage:
    python train.py                          # Train with default settings
    python train.py --model gradient_boosting
    python train.py --model random_forest --n-estimators 200
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def load_california_housing(
    test_size: float = 0.2,
    seed: int = 42,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler | None]:
    """Load and prepare California Housing dataset.

    Args:
        test_size: Fraction of data for testing
        seed: Random seed for reproducibility
        normalize: Whether to normalize features

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    log.info("Loading California Housing dataset...")
    housing = fetch_california_housing()

    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    log.info(f"Dataset shape: {X.shape}")
    log.info(f"Features: {feature_names}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    log.info(f"Train samples: {len(X_train)}")
    log.info(f"Test samples: {len(X_test)}")

    # Normalize features
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        log.info("Applied feature normalization")

    return X_train, X_test, y_train, y_test, list(feature_names), scaler


def create_model(
    model_name: str,
    n_estimators: int = 100,
    max_depth: int | None = None,
    learning_rate: float = 0.1,
    alpha: float = 1.0,
    seed: int = 42,
):
    """Create a sklearn regression model.

    Args:
        model_name: One of 'ridge', 'random_forest', 'gradient_boosting'
        n_estimators: Number of trees (for ensemble methods)
        max_depth: Maximum tree depth (None for unlimited)
        learning_rate: Learning rate (for gradient boosting)
        alpha: Regularization strength (for ridge)
        seed: Random seed

    Returns:
        Sklearn regressor model
    """
    models = {
        "ridge": lambda: Ridge(alpha=alpha, random_state=seed),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth or 3,
            learning_rate=learning_rate,
            random_state=seed,
        ),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name]()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }


def train(
    model_name: str = "gradient_boosting",
    output_dir: str = "outputs",
    test_size: float = 0.2,
    normalize: bool = True,
    seed: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = None,
    learning_rate: float = 0.1,
    alpha: float = 1.0,
) -> dict:
    """Train a sklearn model on California Housing dataset.

    Args:
        model_name: Model type to train
        output_dir: Directory to save model and results
        test_size: Fraction of data for testing
        normalize: Whether to normalize features
        seed: Random seed
        n_estimators: Number of trees (for ensemble methods)
        max_depth: Maximum tree depth
        learning_rate: Learning rate (for gradient boosting)
        alpha: Regularization strength (for ridge)

    Returns:
        Dictionary with training results and metrics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test, feature_names, scaler = load_california_housing(
        test_size=test_size,
        seed=seed,
        normalize=normalize,
    )

    # Create model
    log.info(f"Creating {model_name} model...")
    model = create_model(
        model_name=model_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        alpha=alpha,
        seed=seed,
    )

    # Train
    log.info("Training model...")
    model.fit(X_train, y_train)
    log.info("Training complete")

    # Evaluate on train set
    y_train_pred = model.predict(X_train)
    train_metrics = compute_metrics(y_train, y_train_pred)
    log.info(f"Train RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    log.info(f"Test RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")

    # Save model
    model_path = output_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Saved model to {model_path}")

    # Save scaler if used
    if scaler is not None:
        scaler_path = output_path / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        log.info(f"Saved scaler to {scaler_path}")

    # Feature importance (for tree-based models)
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feature_importance = {name: float(imp) for name, imp in zip(feature_names, importance)}
        log.info("Feature importance:")
        for name, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
            log.info(f"  {name}: {imp:.4f}")

    # Compile results
    results = {
        "model_name": model_name,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "config": {
            "test_size": test_size,
            "normalize": normalize,
            "seed": seed,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "alpha": alpha,
        },
    }

    # Save results
    results_path = output_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved results to {results_path}")

    return results


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description="Train sklearn models on California Housing dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gradient_boosting",
        choices=["ridge", "random_forest", "gradient_boosting"],
        help="Model type to train",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save model and results",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip feature normalization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees (for ensemble methods)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (for gradient boosting)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularization strength (for ridge)",
    )

    args = parser.parse_args()

    results = train(
        model_name=args.model,
        output_dir=args.output_dir,
        test_size=args.test_size,
        normalize=not args.no_normalize,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
    )

    print("\n" + "=" * 50)
    print("Training Results Summary")
    print("=" * 50)
    print(f"Model: {results['model_name']}")
    print(f"Train samples: {results['train_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print("\nTest Metrics:")
    print(f"  RMSE: {results['test_metrics']['rmse']:.4f}")
    print(f"  MAE:  {results['test_metrics']['mae']:.4f}")
    print(f"  R²:   {results['test_metrics']['r2']:.4f}")
    print(f"  MAPE: {results['test_metrics']['mape']:.2f}%")


if __name__ == "__main__":
    main()
