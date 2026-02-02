"""Tests for the image classification model."""

import sys
from pathlib import Path

import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from model import SimpleCNN, create_model, load_model


class TestSimpleCNN:
    """Tests for SimpleCNN model."""

    def test_model_creation(self):
        """Test that model can be created with default parameters."""
        model = create_model()
        assert isinstance(model, SimpleCNN)
        assert model.num_classes == 2

    def test_model_with_custom_classes(self):
        """Test model creation with custom number of classes."""
        model = create_model(num_classes=10)
        assert model.num_classes == 10

    def test_model_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = create_model(num_classes=2)
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)

        output = model(input_tensor)

        assert output.shape == (batch_size, 2)

    def test_model_output_values(self):
        """Test that output contains valid logits."""
        model = create_model(num_classes=2)
        input_tensor = torch.randn(1, 3, 224, 224)

        output = model(input_tensor)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_eval_mode(self):
        """Test model behaves correctly in eval mode."""
        model = create_model(num_classes=2)
        model.eval()

        input_tensor = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)

        # In eval mode, outputs should be deterministic
        assert torch.allclose(output1, output2)

    def test_model_save_load(self, tmp_path):
        """Test saving and loading model weights."""
        model = create_model(num_classes=2)
        model_path = tmp_path / "test_model.pt"

        # Save
        torch.save(model.state_dict(), model_path)

        # Load
        loaded_model = load_model(str(model_path), num_classes=2)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_model_dropout_effect(self):
        """Test that dropout has effect during training."""
        model = create_model(num_classes=2, dropout=0.5)
        model.train()

        input_tensor = torch.randn(1, 3, 224, 224)

        # Multiple forward passes should give different results due to dropout
        outputs = [model(input_tensor) for _ in range(5)]

        # At least some outputs should be different
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should cause different outputs during training"

    def test_model_gradients(self):
        """Test that gradients flow correctly."""
        model = create_model(num_classes=2)
        input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        target = torch.tensor([0])

        output = model(input_tensor)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
