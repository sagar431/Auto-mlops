"""Tests for the training module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from model import TextCNN
from train import train, train_epoch, validate


class TestTrainingFunctions:
    """Tests for training helper functions."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock data loader for testing."""
        batch_size = 8
        seq_len = 50
        num_batches = 4
        vocab_size = 1000

        data = [
            (
                torch.randint(0, vocab_size, (batch_size, seq_len)),
                torch.randint(0, 2, (batch_size,)),
            )
            for _ in range(num_batches)
        ]

        loader = MagicMock()
        loader.__iter__ = MagicMock(return_value=iter(data))
        loader.__len__ = MagicMock(return_value=num_batches)
        loader.dataset = MagicMock()
        loader.dataset.__len__ = MagicMock(return_value=batch_size * num_batches)
        return loader

    @pytest.fixture
    def model_setup(self, mock_loader):
        """Create model and training components."""
        model = TextCNN(vocab_size=1000, num_classes=2)
        device = torch.device("cpu")
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        return model, mock_loader, criterion, optimizer, device

    def test_train_epoch(self, model_setup):
        """Test single training epoch."""
        model, train_loader, criterion, optimizer, device = model_setup

        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_validate(self, model_setup):
        """Test validation function."""
        model, val_loader, criterion, _, device = model_setup

        loss, acc = validate(model, val_loader, criterion, device)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_model_changes_with_training(self, model_setup):
        """Test that model weights change with training."""
        model, train_loader, criterion, optimizer, device = model_setup

        # Get initial weights
        initial_weights = model.fc.weight.clone().detach()

        # Train for one epoch
        train_epoch(model, train_loader, criterion, optimizer, device)

        # Weights should have changed
        assert not torch.allclose(initial_weights, model.fc.weight)


class TestFullTraining:
    """Tests for the full training function."""

    @pytest.fixture
    def mock_data_loaders(self):
        """Create mock data loaders."""
        batch_size = 8
        seq_len = 50
        num_batches = 2
        vocab_size = 500

        def create_mock_loader():
            data = [
                (
                    torch.randint(0, vocab_size, (batch_size, seq_len)),
                    torch.randint(0, 2, (batch_size,)),
                )
                for _ in range(num_batches)
            ]
            loader = MagicMock()
            loader.__iter__ = MagicMock(return_value=iter(data))
            loader.__len__ = MagicMock(return_value=num_batches)
            loader.dataset = MagicMock()
            loader.dataset.__len__ = MagicMock(return_value=batch_size * num_batches)
            return loader

        return create_mock_loader

    def test_train_function_synthetic(self, tmp_path):
        """Test the main train function with synthetic data."""
        output_dir = str(tmp_path / "models")

        results = train(
            data_dir=str(tmp_path / "data"),
            output_dir=output_dir,
            epochs=2,
            batch_size=8,
            vocab_size=500,
            max_length=50,
            num_workers=0,
            use_synthetic=True,
        )

        assert "best_accuracy" in results
        assert "final_accuracy" in results
        assert "epochs_trained" in results
        assert "class_names" in results
        assert "model_path" in results
        assert "vocab_size" in results
        assert "history" in results
        assert results["epochs_trained"] == 2

    def test_train_saves_model(self, tmp_path):
        """Test that training saves model files."""
        output_dir = str(tmp_path / "models")

        train(
            data_dir=str(tmp_path / "data"),
            output_dir=output_dir,
            epochs=1,
            batch_size=8,
            vocab_size=500,
            max_length=50,
            num_workers=0,
            use_synthetic=True,
        )

        output_path = Path(output_dir)
        assert (output_path / "best_model.pt").exists()
        assert (output_path / "final_model.pt").exists()
        assert (output_path / "vocab.json").exists()
        assert (output_path / "classes.txt").exists()

    def test_train_saves_classes(self, tmp_path):
        """Test that class names are saved correctly."""
        output_dir = str(tmp_path / "models")

        train(
            data_dir=str(tmp_path / "data"),
            output_dir=output_dir,
            epochs=1,
            batch_size=8,
            vocab_size=500,
            max_length=50,
            num_workers=0,
            use_synthetic=True,
        )

        classes_file = Path(output_dir) / "classes.txt"
        with open(classes_file) as f:
            saved_classes = [line.strip() for line in f.readlines()]

        assert len(saved_classes) == 2

    def test_train_history(self, tmp_path):
        """Test that training history is recorded."""
        output_dir = str(tmp_path / "models")

        results = train(
            data_dir=str(tmp_path / "data"),
            output_dir=output_dir,
            epochs=3,
            batch_size=8,
            vocab_size=500,
            max_length=50,
            num_workers=0,
            use_synthetic=True,
        )

        history = results["history"]
        assert len(history["train_loss"]) == 3
        assert len(history["train_acc"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["val_acc"]) == 3

    def test_train_with_seed(self, tmp_path):
        """Test that training respects the random seed."""
        output_dir = str(tmp_path / "models")

        results = train(
            data_dir=str(tmp_path / "data"),
            output_dir=output_dir,
            epochs=1,
            batch_size=8,
            vocab_size=500,
            max_length=50,
            num_workers=0,
            use_synthetic=True,
            seed=123,
        )

        assert results is not None

    def test_train_lstm_model(self, tmp_path):
        """Test training with LSTM model."""
        output_dir = str(tmp_path / "models")

        results = train(
            data_dir=str(tmp_path / "data"),
            output_dir=output_dir,
            epochs=1,
            batch_size=8,
            vocab_size=500,
            max_length=50,
            num_workers=0,
            use_synthetic=True,
            model_type="lstm",
        )

        assert results is not None
        assert results["epochs_trained"] == 1


class TestHuggingFaceIntegration:
    """Tests for HuggingFace integration."""

    def test_huggingface_text_dataset(self):
        """Test HuggingFaceTextDataset class."""
        from train import HuggingFaceTextDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        texts = ["This is a test.", "Another test sentence."]
        labels = [0, 1]

        dataset = HuggingFaceTextDataset(texts, labels, tokenizer, max_length=32)

        assert len(dataset) == 2

        input_ids, label = dataset[0]
        assert isinstance(input_ids, torch.Tensor)
        assert input_ids.shape == (32,)
        assert label == 0

    def test_create_huggingface_tokenizer(self):
        """Test HuggingFace tokenizer creation."""
        from train import create_huggingface_tokenizer

        tokenizer, vocab_size = create_huggingface_tokenizer(
            tokenizer_name="bert-base-uncased",
            max_length=128,
        )

        assert tokenizer is not None
        assert vocab_size == 30522  # bert-base-uncased vocab size

    def test_load_imdb_huggingface(self):
        """Test loading IMDB dataset from HuggingFace."""
        from train import load_imdb_huggingface

        # Load with limited samples for quick test
        train_texts, train_labels, test_texts, test_labels, class_names = load_imdb_huggingface(
            max_samples=10
        )

        assert len(train_texts) == 10
        assert len(train_labels) == 10
        assert len(test_texts) == 10
        assert len(test_labels) == 10
        assert class_names == ["negative", "positive"]

    def test_train_huggingface_mode(self, tmp_path):
        """Test training with HuggingFace tokenizer and IMDB data."""
        output_dir = str(tmp_path / "models")

        results = train(
            data_dir=str(tmp_path / "data"),
            output_dir=output_dir,
            epochs=1,
            batch_size=4,
            max_length=64,
            num_workers=0,
            use_synthetic=False,
            use_huggingface=True,
            tokenizer_name="bert-base-uncased",
            max_samples=20,  # Use very few samples for quick test
        )

        assert "best_accuracy" in results
        assert "final_accuracy" in results
        assert results["epochs_trained"] == 1

        # Check tokenizer was saved
        output_path = Path(output_dir)
        assert (output_path / "tokenizer").exists()
        assert (output_path / "tokenizer" / "tokenizer_config.json").exists()
