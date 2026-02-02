"""Tests for the ResNet18 image classification model."""

import sys
from pathlib import Path

import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from model import ResNet18, create_model, load_model


class TestResNet18:
    """Tests for ResNet18 model."""

    def test_model_creation(self):
        """Test that model can be created with default parameters."""
        model = create_model()
        assert isinstance(model, ResNet18)
        assert model.num_classes == 10

    def test_model_with_custom_classes(self):
        """Test model creation with custom number of classes."""
        model = create_model(num_classes=5)
        assert model.num_classes == 5

    def test_model_forward_pass_cifar(self):
        """Test forward pass with CIFAR-10 input shape (32x32)."""
        model = create_model(num_classes=10, input_size=32)
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 32, 32)

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)

    def test_model_forward_pass_imagenet(self):
        """Test forward pass with ImageNet input shape (224x224)."""
        model = create_model(num_classes=10, input_size=224)
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)

    def test_model_output_values(self):
        """Test that output contains valid logits."""
        model = create_model(num_classes=10)
        input_tensor = torch.randn(1, 3, 32, 32)

        output = model(input_tensor)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_eval_mode(self):
        """Test model behaves correctly in eval mode."""
        model = create_model(num_classes=10)
        model.eval()

        input_tensor = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)

        # In eval mode, outputs should be deterministic
        assert torch.allclose(output1, output2)

    def test_model_save_load(self, tmp_path):
        """Test saving and loading model weights."""
        model = create_model(num_classes=10, input_size=32)
        model_path = tmp_path / "test_model.pt"

        # Save
        torch.save(model.state_dict(), model_path)

        # Load
        loaded_model = load_model(str(model_path), num_classes=10, input_size=32)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_model_gradients(self):
        """Test that gradients flow correctly."""
        model = create_model(num_classes=10)
        input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
        target = torch.tensor([0])

        output = model(input_tensor)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_model_different_batch_sizes(self):
        """Test model works with various batch sizes."""
        model = create_model(num_classes=10)
        model.eval()

        for batch_size in [1, 2, 8, 16]:
            input_tensor = torch.randn(batch_size, 3, 32, 32)
            output = model(input_tensor)
            assert output.shape == (batch_size, 10)

    def test_pretrained_model_structure(self):
        """Test that pretrained model has correct structure."""
        # Note: This test doesn't load actual pretrained weights to avoid network calls
        model = create_model(num_classes=5, pretrained=False, input_size=224)

        # Verify final layer has correct output size
        assert model.resnet.fc.out_features == 5

    def test_cifar_model_first_conv(self):
        """Test CIFAR model has modified first conv layer."""
        model = create_model(num_classes=10, input_size=32)

        # For CIFAR, conv1 should be 3x3 with stride 1
        conv1 = model.resnet.conv1
        assert conv1.kernel_size == (3, 3)
        assert conv1.stride == (1, 1)

    def test_imagenet_model_first_conv(self):
        """Test ImageNet model has standard first conv layer."""
        model = create_model(num_classes=10, input_size=224)

        # For ImageNet, conv1 should be 7x7 with stride 2
        conv1 = model.resnet.conv1
        assert conv1.kernel_size == (7, 7)
        assert conv1.stride == (2, 2)

    def test_backward_compatibility_dropout_param(self):
        """Test that dropout parameter is accepted for backward compatibility."""
        # Should not raise an error even though dropout is not used
        model = create_model(num_classes=10, dropout=0.5)
        assert isinstance(model, ResNet18)
