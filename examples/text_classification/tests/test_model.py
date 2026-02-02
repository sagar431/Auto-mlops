"""Tests for the text classification models."""

import sys
from pathlib import Path

import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from model import LSTMClassifier, TextCNN, create_model, load_model


class TestTextCNN:
    """Tests for TextCNN model."""

    def test_model_creation(self):
        """Test that model can be created with default parameters."""
        model = TextCNN()
        assert model.num_classes == 2

    def test_model_with_custom_params(self):
        """Test model creation with custom parameters."""
        model = TextCNN(
            vocab_size=10000,
            embedding_dim=64,
            num_classes=5,
            num_filters=50,
            kernel_sizes=[2, 3, 4],
            dropout=0.3,
        )
        assert model.num_classes == 5

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = TextCNN(vocab_size=1000, num_classes=2)
        batch_size = 4
        seq_len = 50
        x = torch.randint(0, 1000, (batch_size, seq_len))
        output = model(x)
        assert output.shape == (batch_size, 2)

    def test_forward_pass_different_classes(self):
        """Test forward pass with different number of classes."""
        model = TextCNN(vocab_size=1000, num_classes=5)
        batch_size = 8
        seq_len = 100
        x = torch.randint(0, 1000, (batch_size, seq_len))
        output = model(x)
        assert output.shape == (batch_size, 5)

    def test_output_values(self):
        """Test that output contains valid logits."""
        model = TextCNN(vocab_size=1000, num_classes=2)
        x = torch.randint(0, 1000, (1, 50))
        output = model(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_eval_mode(self):
        """Test model behaves correctly in eval mode."""
        model = TextCNN(vocab_size=1000, num_classes=2)
        model.eval()
        x = torch.randint(0, 1000, (1, 50))

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)

    def test_different_sequence_lengths(self):
        """Test model works with various sequence lengths."""
        model = TextCNN(vocab_size=1000, num_classes=2)
        model.eval()

        for seq_len in [10, 50, 100, 256]:
            x = torch.randint(0, 1000, (2, seq_len))
            output = model(x)
            assert output.shape == (2, 2)


class TestLSTMClassifier:
    """Tests for LSTMClassifier model."""

    def test_model_creation(self):
        """Test that model can be created with default parameters."""
        model = LSTMClassifier()
        assert model.num_classes == 2

    def test_model_with_custom_params(self):
        """Test model creation with custom parameters."""
        model = LSTMClassifier(
            vocab_size=10000,
            embedding_dim=64,
            hidden_dim=128,
            num_classes=5,
            num_layers=3,
            dropout=0.3,
            bidirectional=False,
        )
        assert model.num_classes == 5
        assert not model.bidirectional

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = LSTMClassifier(vocab_size=1000, num_classes=2)
        batch_size = 4
        seq_len = 50
        x = torch.randint(0, 1000, (batch_size, seq_len))
        output = model(x)
        assert output.shape == (batch_size, 2)

    def test_forward_pass_unidirectional(self):
        """Test forward pass with unidirectional LSTM."""
        model = LSTMClassifier(vocab_size=1000, num_classes=2, bidirectional=False)
        batch_size = 4
        seq_len = 50
        x = torch.randint(0, 1000, (batch_size, seq_len))
        output = model(x)
        assert output.shape == (batch_size, 2)


class TestCreateModel:
    """Tests for create_model factory function."""

    def test_create_textcnn(self):
        """Test creating TextCNN model."""
        model = create_model(model_type="textcnn", vocab_size=1000, num_classes=2)
        assert isinstance(model, TextCNN)

    def test_create_lstm(self):
        """Test creating LSTM model."""
        model = create_model(model_type="lstm", vocab_size=1000, num_classes=2)
        assert isinstance(model, LSTMClassifier)

    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        try:
            create_model(model_type="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown model type" in str(e)


class TestLoadModel:
    """Tests for model saving and loading."""

    def test_save_load_textcnn(self, tmp_path):
        """Test saving and loading TextCNN model."""
        model = create_model(model_type="textcnn", vocab_size=1000, num_classes=2)
        model_path = tmp_path / "test_model.pt"

        # Save
        torch.save(model.state_dict(), model_path)

        # Load
        loaded_model = load_model(
            str(model_path),
            model_type="textcnn",
            vocab_size=1000,
            num_classes=2,
        )

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_save_load_lstm(self, tmp_path):
        """Test saving and loading LSTM model."""
        model = create_model(model_type="lstm", vocab_size=1000, num_classes=2)
        model_path = tmp_path / "test_model.pt"

        # Save
        torch.save(model.state_dict(), model_path)

        # Load
        loaded_model = load_model(
            str(model_path),
            model_type="lstm",
            vocab_size=1000,
            num_classes=2,
        )

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_loaded_model_in_eval_mode(self, tmp_path):
        """Test that loaded model is in eval mode."""
        model = create_model(model_type="textcnn", vocab_size=1000, num_classes=2)
        model_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), model_path)

        loaded_model = load_model(
            str(model_path),
            model_type="textcnn",
            vocab_size=1000,
            num_classes=2,
        )

        assert not loaded_model.training


class TestModelGradients:
    """Tests for gradient flow."""

    def test_textcnn_gradients(self):
        """Test that gradients flow correctly in TextCNN."""
        model = create_model(model_type="textcnn", vocab_size=1000, num_classes=2)
        x = torch.randint(0, 1000, (2, 50))
        target = torch.tensor([0, 1])

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_lstm_gradients(self):
        """Test that gradients flow correctly in LSTM."""
        model = create_model(model_type="lstm", vocab_size=1000, num_classes=2)
        x = torch.randint(0, 1000, (2, 50))
        target = torch.tensor([0, 1])

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
