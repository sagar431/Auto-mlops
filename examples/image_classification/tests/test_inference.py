"""Tests for canonical checkpoint-driven image inference."""

from pathlib import Path

import pytest
import torch
from golden_train import TrainingConfig, train_golden
from inference import ImageClassifier, InvalidImageError
from PIL import Image


@pytest.fixture
def golden_artifact(tmp_path: Path) -> Path:
    result = train_golden(
        tmp_path / "artifacts",
        TrainingConfig(epochs=2, train_samples=32, validation_samples=8, batch_size=8),
    )
    return Path(result["checkpoint_path"])


def test_classifier_loads_classes_and_preprocessing_from_checkpoint(golden_artifact):
    classifier = ImageClassifier(golden_artifact)
    assert classifier.classes == ["red", "blue"]
    assert classifier.image_size == 16
    assert classifier.normalization == {"mean": [0.5] * 3, "std": [0.5] * 3}
    assert str(classifier.device) == "cpu"


@pytest.mark.parametrize(
    ("color", "expected_class"),
    [((255, 0, 0), "red"), ((0, 0, 255), "blue")],
)
def test_real_prediction_uses_saved_model(golden_artifact, color, expected_class):
    classifier = ImageClassifier(golden_artifact)
    prediction = classifier.predict(Image.new("RGB", (24, 24), color=color))
    assert prediction["predicted_class"] == expected_class
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert set(prediction["probabilities"]) == {"red", "blue"}
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_prediction_from_png_path(golden_artifact, tmp_path):
    image_path = tmp_path / "red.png"
    Image.new("RGB", (16, 16), color="red").save(image_path)
    prediction = ImageClassifier(golden_artifact).predict(image_path)
    assert prediction["predicted_class"] == "red"


def test_missing_image_fails_clearly(golden_artifact, tmp_path):
    with pytest.raises(FileNotFoundError, match="Input image was not found"):
        ImageClassifier(golden_artifact).predict(tmp_path / "missing.png")


def test_invalid_image_fails_clearly(golden_artifact, tmp_path):
    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_text("not an image")
    with pytest.raises(InvalidImageError, match="valid PNG or JPEG"):
        ImageClassifier(golden_artifact).predict(invalid_path)


def test_classifier_rejects_non_cpu_device(golden_artifact):
    with pytest.raises(ValueError, match="CPU inference only"):
        ImageClassifier(golden_artifact, device="cuda")


def test_predictions_are_deterministic(golden_artifact):
    classifier = ImageClassifier(golden_artifact)
    image = Image.new("RGB", (16, 16), color="blue")
    assert classifier.predict(image) == classifier.predict(image)


def test_prediction_is_not_hard_coded(golden_artifact):
    classifier = ImageClassifier(golden_artifact)
    red = classifier.predict(Image.new("RGB", (16, 16), color="red"))
    blue = classifier.predict(Image.new("RGB", (16, 16), color="blue"))
    assert red["probabilities"] != blue["probabilities"]
    assert not torch.isclose(
        torch.tensor(red["probabilities"]["red"]),
        torch.tensor(blue["probabilities"]["red"]),
    )
