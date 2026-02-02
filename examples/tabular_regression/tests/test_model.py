"""Tests for model architectures."""

import json
import sys
from pathlib import Path

import pytest
import torch

# Add project directory to path
project_dir = Path(__file__).parent.parent / "project"
sys.path.insert(0, str(project_dir))

from model import MLP, TabNet, create_model, load_model  # noqa: E402


class TestMLP:
    """Tests for MLP model."""

    def test_forward_pass(self, sample_features):
        """Test MLP forward pass."""
        model = MLP(input_dim=8, hidden_dims=[32, 16])
        x = torch.tensor(sample_features)
        output = model(x)

        assert output.shape == (100,)
        assert not torch.isnan(output).any()

    def test_different_hidden_dims(self):
        """Test MLP with different hidden dimensions."""
        model = MLP(input_dim=10, hidden_dims=[64, 32, 16, 8])
        x = torch.randn(32, 10)
        output = model(x)

        assert output.shape == (32,)

    def test_different_activations(self):
        """Test MLP with different activation functions."""
        for activation in ["relu", "leaky_relu", "elu", "gelu", "selu"]:
            model = MLP(input_dim=8, hidden_dims=[16], activation=activation)
            x = torch.randn(10, 8)
            output = model(x)
            assert output.shape == (10,)

    def test_batch_norm_disabled(self):
        """Test MLP without batch normalization."""
        model = MLP(input_dim=8, hidden_dims=[32], use_batch_norm=False)
        x = torch.randn(10, 8)
        output = model(x)

        assert output.shape == (10,)


class TestTabNet:
    """Tests for TabNet model."""

    def test_forward_pass(self, sample_features):
        """Test TabNet forward pass."""
        model = TabNet(input_dim=8, n_steps=2, n_d=16, n_a=16)
        x = torch.tensor(sample_features)
        output = model(x)

        assert output.shape == (100,)
        assert not torch.isnan(output).any()

    def test_different_steps(self):
        """Test TabNet with different number of steps."""
        for n_steps in [1, 2, 3, 5]:
            model = TabNet(input_dim=8, n_steps=n_steps, n_d=16, n_a=16)
            x = torch.randn(32, 8)
            output = model(x)
            assert output.shape == (32,)


class TestCreateModel:
    """Tests for model factory function."""

    def test_create_mlp(self):
        """Test creating MLP model."""
        model = create_model("mlp", input_dim=8, hidden_dims=[32])
        assert isinstance(model, MLP)

    def test_create_tabnet(self):
        """Test creating TabNet model."""
        model = create_model("tabnet", input_dim=8, n_steps=2, n_d=16, n_a=16)
        assert isinstance(model, TabNet)

    def test_unknown_model_type(self):
        """Test error on unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("unknown", input_dim=8)


class TestLoadModel:
    """Tests for model loading."""

    def test_load_mlp(self, tmp_model_dir):
        """Test loading MLP model."""
        # Create and save model
        model = MLP(input_dim=8, hidden_dims=[32, 16])
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {"model_type": "mlp", "input_dim": 8, "hidden_dims": [32, 16]},
        }
        model_path = tmp_model_dir / "test_model.pt"
        torch.save(checkpoint, model_path)

        # Load model
        loaded = load_model(str(model_path))
        assert isinstance(loaded, MLP)

        # Verify same output
        x = torch.randn(10, 8)
        model.eval()
        with torch.no_grad():
            expected = model(x)
            actual = loaded(x)
        assert torch.allclose(expected, actual)

    def test_load_with_config_file(self, tmp_model_dir):
        """Test loading model with separate config file."""
        model = MLP(input_dim=8, hidden_dims=[32])
        torch.save(model.state_dict(), tmp_model_dir / "model.pt")

        config = {"model_type": "mlp", "input_dim": 8, "hidden_dims": [32]}
        with open(tmp_model_dir / "model_config.json", "w") as f:
            json.dump(config, f)

        loaded = load_model(str(tmp_model_dir / "model.pt"))
        assert isinstance(loaded, MLP)
