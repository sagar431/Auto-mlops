"""Tests for the inference module."""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from inference import ImageClassifier
from model import create_model


class TestImageClassifier:
    """Tests for ImageClassifier."""

    @pytest.fixture
    def trained_model(self, tmp_path):
        """Create and save a mock trained model."""
        model = create_model(num_classes=2, input_size=224)
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        classes_path = tmp_path / "classes.txt"
        classes_path.write_text("cat\ndog\n")

        return str(model_path), str(classes_path)

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample image for testing."""
        img_path = tmp_path / "test_image.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)
        return str(img_path)

    def test_classifier_initialization(self, trained_model):
        """Test classifier can be initialized."""
        model_path, classes_path = trained_model

        classifier = ImageClassifier(model_path=model_path, classes_file=classes_path)

        assert classifier.classes == ["cat", "dog"]
        assert classifier.model is not None

    def test_classifier_default_classes(self, trained_model):
        """Test classifier uses default classes if none provided."""
        model_path, _ = trained_model

        classifier = ImageClassifier(model_path=model_path)

        assert classifier.classes == ["cat", "dog"]

    def test_predict_from_path(self, trained_model, sample_image):
        """Test prediction from image path."""
        model_path, classes_path = trained_model

        classifier = ImageClassifier(model_path=model_path, classes_file=classes_path)

        result = classifier.predict(sample_image)

        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["predicted_class"] in ["cat", "dog"]
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 2

    def test_predict_from_pil_image(self, trained_model):
        """Test prediction from PIL Image."""
        model_path, classes_path = trained_model

        classifier = ImageClassifier(model_path=model_path, classes_file=classes_path)

        image = Image.new("RGB", (100, 100), color="blue")
        result = classifier.predict(image)

        assert "predicted_class" in result
        assert result["predicted_class"] in ["cat", "dog"]

    def test_predict_probabilities_sum_to_one(self, trained_model, sample_image):
        """Test that probabilities sum to approximately 1."""
        model_path, classes_path = trained_model

        classifier = ImageClassifier(model_path=model_path, classes_file=classes_path)

        result = classifier.predict(sample_image)
        total = sum(result["probabilities"].values())

        assert abs(total - 1.0) < 1e-5

    def test_predict_batch(self, trained_model, tmp_path):
        """Test batch prediction."""
        model_path, classes_path = trained_model

        # Create multiple test images
        image_paths = []
        for i in range(3):
            img_path = tmp_path / f"test_image_{i}.png"
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(img_path)
            image_paths.append(str(img_path))

        classifier = ImageClassifier(model_path=model_path, classes_file=classes_path)

        results = classifier.predict_batch(image_paths)

        assert len(results) == 3
        for result in results:
            assert "predicted_class" in result
            assert "confidence" in result

    def test_classifier_deterministic(self, trained_model, sample_image):
        """Test that classifier gives consistent results."""
        model_path, classes_path = trained_model

        classifier = ImageClassifier(model_path=model_path, classes_file=classes_path)

        result1 = classifier.predict(sample_image)
        result2 = classifier.predict(sample_image)

        assert result1["predicted_class"] == result2["predicted_class"]
        assert abs(result1["confidence"] - result2["confidence"]) < 1e-5
