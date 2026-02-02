"""Data preparation script for tabular regression."""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def prepare_california_housing(
    data_dir: str,
    test_size: float = 0.2,
    seed: int = 42,
    normalize: bool = True,
) -> dict:
    """Prepare California Housing dataset.

    Args:
        data_dir: Directory to save prepared data
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
        normalize: Whether to normalize features

    Returns:
        Dataset info dictionary
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Fetching California Housing dataset...")
    housing = fetch_california_housing()

    # Create DataFrame
    feature_names = housing.feature_names
    df = pd.DataFrame(housing.data, columns=feature_names)
    df["MedHouseVal"] = housing.target

    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Features: {feature_names}")
    log.info("Target: MedHouseVal (median house value in $100,000s)")

    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    log.info(f"Train samples: {len(train_df)}")
    log.info(f"Test samples: {len(test_df)}")

    # Normalize features
    scaler = None
    if normalize:
        scaler = StandardScaler()

        # Extract targets before modifying DataFrames
        train_targets = train_df["MedHouseVal"].values
        test_targets = test_df["MedHouseVal"].values

        # Fit on training data only and transform
        train_features_scaled = scaler.fit_transform(train_df[feature_names])
        test_features_scaled = scaler.transform(test_df[feature_names])

        # Create new DataFrames with scaled features
        train_df = pd.DataFrame(train_features_scaled, columns=feature_names)
        train_df["MedHouseVal"] = train_targets

        test_df = pd.DataFrame(test_features_scaled, columns=feature_names)
        test_df["MedHouseVal"] = test_targets

        # Save scaler
        with open(data_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        log.info("Saved feature scaler")

    # Save data
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    log.info(f"Saved train.csv ({len(train_df)} rows)")
    log.info(f"Saved test.csv ({len(test_df)} rows)")

    # Compute statistics
    info = {
        "dataset": "california_housing",
        "description": "California Housing dataset from sklearn",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "target_name": "MedHouseVal",
        "target_description": "Median house value in $100,000s",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "normalized": normalize,
        "test_size": test_size,
        "seed": seed,
        "feature_stats": {},
    }

    # Feature statistics (from original data)
    original_df = pd.DataFrame(housing.data, columns=feature_names)
    for col in feature_names:
        info["feature_stats"][col] = {
            "mean": float(original_df[col].mean()),
            "std": float(original_df[col].std()),
            "min": float(original_df[col].min()),
            "max": float(original_df[col].max()),
        }

    # Target statistics
    info["target_stats"] = {
        "mean": float(housing.target.mean()),
        "std": float(housing.target.std()),
        "min": float(housing.target.min()),
        "max": float(housing.target.max()),
    }

    # Save info
    with open(data_dir / "data_info.json", "w") as f:
        json.dump(info, f, indent=2)

    log.info("Saved data_info.json")

    return info


def prepare_synthetic_data(
    data_dir: str,
    n_train: int = 800,
    n_test: int = 200,
    n_features: int = 8,
    seed: int = 42,
) -> dict:
    """Prepare synthetic regression data for testing.

    Args:
        data_dir: Directory to save data
        n_train: Number of training samples
        n_test: Number of test samples
        n_features: Number of features
        seed: Random seed

    Returns:
        Dataset info dictionary
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    log.info(f"Generating synthetic data: {n_train} train, {n_test} test samples")

    # Generate features
    n_total = n_train + n_test
    features = np.random.randn(n_total, n_features).astype(np.float32)

    # Generate targets with nonlinear relationship
    weights = np.random.randn(n_features).astype(np.float32)
    targets = (
        np.dot(features, weights)
        + 0.5 * np.sin(features[:, 0] * 2)
        + 0.3 * features[:, 1] ** 2
        + 0.1 * np.random.randn(n_total)
    ).astype(np.float32)

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Split
    train_features = features[:n_train]
    train_targets = targets[:n_train]
    test_features = features[n_train:]
    test_targets = targets[n_train:]

    # Create DataFrames
    train_df = pd.DataFrame(train_features, columns=feature_names)
    train_df["target"] = train_targets
    test_df = pd.DataFrame(test_features, columns=feature_names)
    test_df["target"] = test_targets

    # Save
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    log.info(f"Saved train.csv ({n_train} rows)")
    log.info(f"Saved test.csv ({n_test} rows)")

    # Info
    info = {
        "dataset": "synthetic",
        "description": "Synthetic regression data for testing",
        "n_features": n_features,
        "feature_names": feature_names,
        "target_name": "target",
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
    }

    with open(data_dir / "data_info.json", "w") as f:
        json.dump(info, f, indent=2)

    log.info("Saved data_info.json")

    return info


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(description="Prepare data for tabular regression")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save prepared data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="california",
        choices=["california", "synthetic"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip feature normalization",
    )
    # Synthetic data options
    parser.add_argument(
        "--n-train",
        type=int,
        default=800,
        help="Number of training samples (synthetic only)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=200,
        help="Number of test samples (synthetic only)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=8,
        help="Number of features (synthetic only)",
    )

    args = parser.parse_args()

    if args.dataset == "california":
        info = prepare_california_housing(
            data_dir=args.data_dir,
            test_size=args.test_size,
            seed=args.seed,
            normalize=not args.no_normalize,
        )
    else:
        info = prepare_synthetic_data(
            data_dir=args.data_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            n_features=args.n_features,
            seed=args.seed,
        )

    log.info(f"Data preparation complete: {info['n_train']} train, {info['n_test']} test samples")


if __name__ == "__main__":
    main()
