"""Dataset utilities for tabular regression."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular regression data."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ):
        """Initialize dataset.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,), optional for inference
            feature_names: List of feature names
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        self.feature_names = feature_names

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return (self.features[idx],)


def load_data(
    data_dir: str,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load data from CSV files.

    Args:
        data_dir: Directory containing train.csv and test.csv
        split: One of 'train' or 'test'

    Returns:
        Tuple of (features, targets, feature_names)
    """
    data_dir = Path(data_dir)
    file_path = data_dir / f"{split}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Assume last column is target
    feature_names = df.columns[:-1].tolist()
    target_name = df.columns[-1]

    features = df[feature_names].values.astype(np.float32)
    targets = df[target_name].values.astype(np.float32)

    return features, targets, feature_names


def create_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader, dict]:
    """Create train and test dataloaders.

    Args:
        data_dir: Directory containing train.csv and test.csv
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, test_loader, info_dict)
    """
    # Load data
    train_features, train_targets, feature_names = load_data(data_dir, "train")
    test_features, test_targets, _ = load_data(data_dir, "test")

    # Create datasets
    train_dataset = TabularDataset(train_features, train_targets, feature_names)
    test_dataset = TabularDataset(test_features, test_targets, feature_names)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    info = {
        "input_dim": train_features.shape[1],
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "feature_names": feature_names,
    }

    return train_loader, test_loader, info


def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 8,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic regression data for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        seed: Random seed

    Returns:
        Tuple of (features, targets, feature_names)
    """
    np.random.seed(seed)

    # Generate features
    features = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate targets with nonlinear relationship
    weights = np.random.randn(n_features).astype(np.float32)
    targets = (
        np.dot(features, weights)
        + 0.5 * np.sin(features[:, 0] * 2)
        + noise * np.random.randn(n_samples)
    ).astype(np.float32)

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return features, targets, feature_names


def save_synthetic_data(
    data_dir: str,
    n_train: int = 800,
    n_test: int = 200,
    n_features: int = 8,
    seed: int = 42,
) -> dict:
    """Generate and save synthetic data for testing.

    Args:
        data_dir: Directory to save data
        n_train: Number of training samples
        n_test: Number of test samples
        n_features: Number of features
        seed: Random seed

    Returns:
        Data info dictionary
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    features, targets, feature_names = create_synthetic_data(
        n_samples=n_train + n_test,
        n_features=n_features,
        seed=seed,
    )

    # Split into train/test
    train_features = features[:n_train]
    train_targets = targets[:n_train]
    test_features = features[n_train:]
    test_targets = targets[n_train:]

    # Create DataFrames
    train_df = pd.DataFrame(train_features, columns=feature_names)
    train_df["target"] = train_targets
    test_df = pd.DataFrame(test_features, columns=feature_names)
    test_df["target"] = test_targets

    # Save to CSV
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    # Save info
    info = {
        "dataset": "synthetic",
        "n_features": n_features,
        "n_train": n_train,
        "n_test": n_test,
        "feature_names": feature_names,
        "target_name": "target",
    }
    with open(data_dir / "data_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return info
