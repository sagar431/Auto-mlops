"""Tests for sklearn-based training script."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing

# Add parent directory to path to import train module
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from train import (  # noqa: E402
    compute_metrics,
    create_model,
    load_california_housing,
    train,
)


class TestLoadCaliforniaHousing:
    """Tests for load_california_housing function."""

    def test_load_data_shape(self):
        """Test that data is loaded with correct shapes."""
        X_train, X_test, y_train, y_test, feature_names, scaler = load_california_housing(
            test_size=0.2, seed=42
        )

        # Check shapes
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1] == 8  # California Housing has 8 features

        # Check split ratio
        total = len(y_train) + len(y_test)
        assert abs(len(y_test) / total - 0.2) < 0.01

    def test_load_data_with_normalization(self):
        """Test that normalization is applied correctly."""
        X_train, _, _, _, _, scaler = load_california_housing(normalize=True, seed=42)

        assert scaler is not None
        # Normalized data should have mean close to 0 and std close to 1
        assert abs(X_train.mean()) < 0.1
        assert abs(X_train.std() - 1.0) < 0.1

    def test_load_data_without_normalization(self):
        """Test loading data without normalization."""
        X_train, _, _, _, _, scaler = load_california_housing(normalize=False, seed=42)

        assert scaler is None
        # Original California Housing data has different ranges
        housing = fetch_california_housing()
        assert np.allclose(X_train.mean(axis=0), housing.data.mean(axis=0), rtol=0.1)

    def test_feature_names(self):
        """Test that feature names are returned correctly."""
        _, _, _, _, feature_names, _ = load_california_housing()

        assert len(feature_names) == 8
        assert "MedInc" in feature_names
        assert "Latitude" in feature_names

    def test_reproducibility(self):
        """Test that same seed produces same split."""
        X1, _, y1, _, _, _ = load_california_housing(seed=42)
        X2, _, y2, _, _, _ = load_california_housing(seed=42)

        assert np.allclose(X1, X2)
        assert np.allclose(y1, y2)


class TestCreateModel:
    """Tests for create_model function."""

    def test_create_ridge(self):
        """Test creating Ridge model."""
        model = create_model("ridge", alpha=1.0, seed=42)
        assert model.__class__.__name__ == "Ridge"
        assert model.alpha == 1.0

    def test_create_random_forest(self):
        """Test creating RandomForest model."""
        model = create_model("random_forest", n_estimators=50, max_depth=5, seed=42)
        assert model.__class__.__name__ == "RandomForestRegressor"
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_create_gradient_boosting(self):
        """Test creating GradientBoosting model."""
        model = create_model(
            "gradient_boosting", n_estimators=50, max_depth=3, learning_rate=0.1, seed=42
        )
        assert model.__class__.__name__ == "GradientBoostingRegressor"
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("invalid_model")


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_prediction(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["mape"] == 0.0

    def test_known_values(self):
        """Test metrics with known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])

        metrics = compute_metrics(y_true, y_pred)

        # MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.25
        assert abs(metrics["mse"] - 0.25) < 1e-6
        # RMSE = sqrt(0.25) = 0.5
        assert abs(metrics["rmse"] - 0.5) < 1e-6
        # MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        assert abs(metrics["mae"] - 0.5) < 1e-6

    def test_metrics_types(self):
        """Test that metrics are correct types."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        metrics = compute_metrics(y_true, y_pred)

        assert isinstance(metrics["mse"], float)
        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["r2"], float)
        assert isinstance(metrics["mape"], float)


class TestTrain:
    """Tests for train function."""

    def test_train_ridge(self, tmp_path):
        """Test training Ridge model."""
        output_dir = str(tmp_path / "outputs")

        results = train(
            model_name="ridge",
            output_dir=output_dir,
            test_size=0.2,
            normalize=True,
            seed=42,
        )

        # Check results structure
        assert "model_name" in results
        assert "train_metrics" in results
        assert "test_metrics" in results
        assert results["model_name"] == "ridge"

        # Check test metrics are reasonable for California Housing
        assert 0 < results["test_metrics"]["rmse"] < 2.0  # RMSE should be reasonable
        assert results["test_metrics"]["r2"] > 0.5  # R² should be positive

        # Check files were saved
        assert Path(output_dir, "model.pkl").exists()
        assert Path(output_dir, "scaler.pkl").exists()
        assert Path(output_dir, "results.json").exists()

    def test_train_random_forest(self, tmp_path):
        """Test training RandomForest model."""
        output_dir = str(tmp_path / "outputs")

        results = train(
            model_name="random_forest",
            output_dir=output_dir,
            n_estimators=10,  # Small for faster test
            seed=42,
        )

        assert results["model_name"] == "random_forest"
        assert "feature_importance" in results
        assert results["feature_importance"] is not None
        assert len(results["feature_importance"]) == 8

    def test_train_gradient_boosting(self, tmp_path):
        """Test training GradientBoosting model."""
        output_dir = str(tmp_path / "outputs")

        results = train(
            model_name="gradient_boosting",
            output_dir=output_dir,
            n_estimators=10,  # Small for faster test
            seed=42,
        )

        assert results["model_name"] == "gradient_boosting"
        assert "feature_importance" in results
        assert results["feature_importance"] is not None

    def test_saved_model_can_be_loaded(self, tmp_path):
        """Test that saved model can be loaded and used."""
        output_dir = str(tmp_path / "outputs")

        train(
            model_name="ridge",
            output_dir=output_dir,
            seed=42,
        )

        # Load model
        with open(Path(output_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)

        # Load scaler
        with open(Path(output_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        # Make prediction
        sample_input = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        scaled_input = scaler.transform(sample_input)
        prediction = model.predict(scaled_input)

        assert prediction.shape == (1,)
        assert isinstance(prediction[0], (float, np.floating))

    def test_results_json_format(self, tmp_path):
        """Test that results.json has correct format."""
        output_dir = str(tmp_path / "outputs")

        train(
            model_name="ridge",
            output_dir=output_dir,
            seed=42,
        )

        with open(Path(output_dir, "results.json")) as f:
            results = json.load(f)

        # Check required keys
        assert "model_name" in results
        assert "train_samples" in results
        assert "test_samples" in results
        assert "train_metrics" in results
        assert "test_metrics" in results
        assert "feature_names" in results
        assert "config" in results

        # Check metrics keys
        for split in ["train_metrics", "test_metrics"]:
            assert "rmse" in results[split]
            assert "mae" in results[split]
            assert "r2" in results[split]
            assert "mape" in results[split]

    def test_train_without_normalization(self, tmp_path):
        """Test training without normalization."""
        output_dir = str(tmp_path / "outputs")

        results = train(
            model_name="ridge",
            output_dir=output_dir,
            normalize=False,
            seed=42,
        )

        # Scaler should not be saved
        assert not Path(output_dir, "scaler.pkl").exists()

        # Model should still work
        assert results["test_metrics"]["rmse"] > 0
