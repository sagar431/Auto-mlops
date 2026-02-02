"""Tests for the inference module."""

import sys
from pathlib import Path

import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from dataset import Vocabulary
from inference import TextClassifier
from model import create_model


class TestTextClassifier:
    """Tests for TextClassifier inference wrapper."""

    def test_classifier_creation(self, tmp_path):
        """Test creating classifier."""
        # Create and save vocab first to get the correct size
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build(["hello world", "goodbye world"])
        vocab_path = tmp_path / "vocab.json"
        vocab.save(str(vocab_path))

        # Create and save model with matching vocab size
        model = create_model(model_type="textcnn", vocab_size=len(vocab), num_classes=2)
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Create and save classes
        classes_path = tmp_path / "classes.txt"
        with open(classes_path, "w") as f:
            f.write("negative\npositive\n")

        classifier = TextClassifier(
            model_path=str(model_path),
            vocab_path=str(vocab_path),
            classes_file=str(classes_path),
            model_type="textcnn",
        )

        assert classifier.classes == ["negative", "positive"]

    def test_classifier_predict(self, tmp_path):
        """Test prediction."""
        # Create and save vocab first to get the correct size
        # Use longer texts to ensure proper tokenization for conv kernel sizes
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build(
            [
                "hello world this is a longer test sentence for building vocabulary",
                "goodbye world this is another test sentence to have more tokens",
            ]
        )
        vocab_path = tmp_path / "vocab.json"
        vocab.save(str(vocab_path))

        # Create and save model with matching vocab size
        model = create_model(model_type="textcnn", vocab_size=len(vocab), num_classes=2)
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        classifier = TextClassifier(
            model_path=str(model_path),
            vocab_path=str(vocab_path),
            model_type="textcnn",
        )

        # Use longer text to ensure sequence length > max kernel size (5)
        result = classifier.predict("hello world this is a test sentence")

        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["predicted_class"] in classifier.classes
        assert 0 <= result["confidence"] <= 1

    def test_classifier_predict_batch(self, tmp_path):
        """Test batch prediction."""
        # Create and save vocab first to get the correct size
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build(
            [
                "hello world this is a longer test sentence for building vocabulary",
                "goodbye world this is another test sentence to have more tokens",
            ]
        )
        vocab_path = tmp_path / "vocab.json"
        vocab.save(str(vocab_path))

        # Create and save model with matching vocab size
        model = create_model(model_type="textcnn", vocab_size=len(vocab), num_classes=2)
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        classifier = TextClassifier(
            model_path=str(model_path),
            vocab_path=str(vocab_path),
            model_type="textcnn",
        )

        # Use longer texts to ensure sequence length > max kernel size (5)
        texts = [
            "hello world this is a test sentence",
            "goodbye world this is another test",
            "test text that is a bit longer now",
        ]
        results = classifier.predict_batch(texts)

        assert len(results) == 3
        for result in results:
            assert "predicted_class" in result
            assert "confidence" in result

    def test_classifier_with_lstm(self, tmp_path):
        """Test classifier with LSTM model."""
        # Create and save vocab first to get the correct size
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build(["hello world", "goodbye world"])
        vocab_path = tmp_path / "vocab.json"
        vocab.save(str(vocab_path))

        # Create and save model with matching vocab size
        model = create_model(model_type="lstm", vocab_size=len(vocab), num_classes=2)
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        classifier = TextClassifier(
            model_path=str(model_path),
            vocab_path=str(vocab_path),
            model_type="lstm",
        )

        result = classifier.predict("hello world")
        assert "predicted_class" in result

    def test_classifier_truncates_long_text(self, tmp_path):
        """Test that long text is truncated in result."""
        # Create and save vocab first to get the correct size
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build(["hello world"])
        vocab_path = tmp_path / "vocab.json"
        vocab.save(str(vocab_path))

        # Create and save model with matching vocab size
        model = create_model(model_type="textcnn", vocab_size=len(vocab), num_classes=2)
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        classifier = TextClassifier(
            model_path=str(model_path),
            vocab_path=str(vocab_path),
            model_type="textcnn",
        )

        long_text = "hello " * 100
        result = classifier.predict(long_text)

        assert len(result["text"]) < len(long_text)
        assert result["text"].endswith("...")
