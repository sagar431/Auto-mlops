"""Tests for inference utilities."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

# Add project directory to path
project_dir = Path(__file__).parent.parent / "project"
sys.path.insert(0, str(project_dir))

from inference import TabularRegressor  # noqa: E402
from model import MLP  # noqa: E402


@pytest.fixture
def saved_model(tmp_model_dir):
    """Create and save a model for testing."""
    model = MLP(input_dim=8, hidden_dims=[16, 8])
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"model_type": "mlp", "input_dim": 8, "hidden_dims": [16, 8]},
    }
    model_path = tmp_model_dir / "model.pt"
    torch.save(checkpoint, model_path)

    # Save config
    with open(tmp_model_dir / "model_config.json", "w") as f:
        json.dump({"model_type": "mlp", "input_dim": 8, "hidden_dims": [16, 8]}, f)

    return model_path


@pytest.fixture
def saved_scaler(tmp_path):
    """Create and save a scaler for testing."""
    scaler = StandardScaler()
    scaler.fit(np.random.randn(100, 8))

    scaler_path = tmp_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return scaler_path


class TestTabularRegressor:
    """Tests for TabularRegressor class."""

    def test_init(self, saved_model):
        """Test regressor initialization."""
        regressor = TabularRegressor(str(saved_model))
        assert regressor.model is not None
        assert regressor.scaler is None

    def test_init_with_scaler(self, saved_model, saved_scaler):
        """Test regressor initialization with scaler."""
        regressor = TabularRegressor(str(saved_model), str(saved_scaler))
        assert regressor.scaler is not None

    def test_predict_dict(self, saved_model):
        """Test prediction with dict input."""
        regressor = TabularRegressor(str(saved_model))

        features = {f"feature_{i}": float(i) for i in range(8)}
        prediction = regressor.predict(features)

        assert isinstance(prediction, float)

    def test_predict_list(self, saved_model):
        """Test prediction with list input."""
        regressor = TabularRegressor(str(saved_model))

        features = [float(i) for i in range(8)]
        prediction = regressor.predict(features)

        assert isinstance(prediction, float)

    def test_predict_array(self, saved_model):
        """Test prediction with numpy array input."""
        regressor = TabularRegressor(str(saved_model))

        features = np.random.randn(8).astype(np.float32)
        prediction = regressor.predict(features)

        assert isinstance(prediction, float)

    def test_predict_batch_array(self, saved_model):
        """Test batch prediction with numpy array."""
        regressor = TabularRegressor(str(saved_model))

        features = np.random.randn(50, 8).astype(np.float32)
        predictions = regressor.predict_batch(features)

        assert predictions.shape == (50,)

    def test_predict_batch_dataframe(self, saved_model):
        """Test batch prediction with DataFrame."""
        regressor = TabularRegressor(str(saved_model))

        df = pd.DataFrame(np.random.randn(30, 8), columns=[f"f{i}" for i in range(8)])
        predictions = regressor.predict_batch(df)

        assert predictions.shape == (30,)

    def test_predict_with_scaler(self, saved_model, saved_scaler):
        """Test prediction with feature scaling."""
        regressor = TabularRegressor(str(saved_model), str(saved_scaler))

        features = np.random.randn(8).astype(np.float32)
        prediction = regressor.predict(features)

        assert isinstance(prediction, float)
