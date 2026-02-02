"""Inference utilities for tabular regression models."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import load_model


class TabularRegressor:
    """Wrapper class for tabular regression inference."""

    def __init__(
        self,
        model_path: str,
        scaler_path: str | None = None,
        device: str = None,
    ):
        """Initialize the regressor.

        Args:
            model_path: Path to model checkpoint
            scaler_path: Path to feature scaler (optional)
            device: Device to use (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path, device=self.device)
        self.model.eval()

        # Load scaler if provided
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # Load model config for feature info
        config_path = Path(model_path).parent / "model_config.json"
        self.config = {}
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)

    def predict(self, features: dict | list | np.ndarray) -> float:
        """Make a single prediction.

        Args:
            features: Input features as dict, list, or array

        Returns:
            Predicted value
        """
        # Convert to numpy array
        if isinstance(features, dict):
            features = np.array(list(features.values()), dtype=np.float32)
        elif isinstance(features, list):
            features = np.array(features, dtype=np.float32)

        features = features.reshape(1, -1)

        # Apply scaler
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Convert to tensor
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Predict
        with torch.no_grad():
            prediction = self.model(tensor).item()

        return prediction

    def predict_batch(
        self,
        features: pd.DataFrame | np.ndarray,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Make batch predictions.

        Args:
            features: Input features as DataFrame or array
            batch_size: Batch size for processing

        Returns:
            Array of predictions
        """
        # Convert to numpy
        if isinstance(features, pd.DataFrame):
            features = features.values

        features = features.astype(np.float32)

        # Apply scaler
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Process in batches
        predictions = []
        n_samples = len(features)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = features[i : i + batch_size]
                tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                batch_pred = self.model(tensor).cpu().detach().tolist()
                predictions.extend(batch_pred)

        return np.array(predictions)


def main():
    """Run interactive inference demo."""
    parser = argparse.ArgumentParser(description="Run inference on tabular regression model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="data/scaler.pkl",
        help="Path to feature scaler",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV file for batch inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output path for batch predictions",
    )

    args = parser.parse_args()

    # Initialize regressor
    regressor = TabularRegressor(
        model_path=args.model_path,
        scaler_path=args.scaler_path if Path(args.scaler_path).exists() else None,
    )

    if args.data_path:
        # Batch inference
        print(f"Loading data from {args.data_path}...")
        df = pd.read_csv(args.data_path)

        # Remove target column if present
        target_cols = ["MedHouseVal", "target"]
        feature_cols = [c for c in df.columns if c not in target_cols]
        features = df[feature_cols]

        print(f"Running inference on {len(features)} samples...")
        predictions = regressor.predict_batch(features)

        # Save predictions
        output_df = df.copy()
        output_df["prediction"] = predictions
        output_df.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")

    else:
        # Interactive demo with California Housing features
        print("\nTabular Regression Inference Demo")
        print("=" * 40)
        print("\nUsing California Housing features:")
        print("  MedInc    - Median income in block group")
        print("  HouseAge  - Median house age in block group")
        print("  AveRooms  - Average rooms per household")
        print("  AveBedrms - Average bedrooms per household")
        print("  Population- Block group population")
        print("  AveOccup  - Average household members")
        print("  Latitude  - Block group latitude")
        print("  Longitude - Block group longitude")
        print("\nExample prediction (typical Bay Area home):")

        # Sample input (normalized values would be used internally)
        sample = {
            "MedInc": 8.3,
            "HouseAge": 41,
            "AveRooms": 6.98,
            "AveBedrms": 1.02,
            "Population": 322,
            "AveOccup": 2.56,
            "Latitude": 37.88,
            "Longitude": -122.23,
        }

        print("\nInput features:")
        for k, v in sample.items():
            print(f"  {k}: {v}")

        prediction = regressor.predict(sample)
        print(f"\nPredicted median house value: ${prediction * 100000:,.2f}")

        print("\n" + "-" * 40)
        print("For batch predictions, use: --data-path <csv_file>")


if __name__ == "__main__":
    main()
