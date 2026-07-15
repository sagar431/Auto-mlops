"""FastAPI contract tests for the golden image-classification service."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from golden_train import TrainingConfig, train_golden
from PIL import Image
from serve import create_app


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    result = train_golden(
        tmp_path / "artifacts",
        TrainingConfig(epochs=2, train_samples=32, validation_samples=8, batch_size=8),
    )
    return Path(result["checkpoint_path"])


def _image_bytes(color: str, image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buffer, format=image_format)
    return buffer.getvalue()


def test_health_reports_loaded_checkpoint(checkpoint_path):
    with TestClient(create_app(checkpoint_path)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "schema_version": "golden-image-classifier.v1",
        "architecture": "tiny_color_cnn_v1",
        "class_names": ["red", "blue"],
    }


def test_predict_returns_real_probability_schema(checkpoint_path):
    with TestClient(create_app(checkpoint_path)) as client:
        response = client.post(
            "/predict",
            files={"file": ("red.png", _image_bytes("red"), "image/png")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "red"
    assert set(payload["probabilities"]) == {"red", "blue"}
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_accepts_jpeg(checkpoint_path):
    with TestClient(create_app(checkpoint_path)) as client:
        response = client.post(
            "/predict",
            files={"file": ("blue.jpg", _image_bytes("blue", "JPEG"), "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.json()["predicted_class"] == "blue"


def test_predict_rejects_invalid_content_type(checkpoint_path):
    with TestClient(create_app(checkpoint_path)) as client:
        response = client.post(
            "/predict", files={"file": ("data.txt", b"hello", "text/plain")}
        )
    assert response.status_code == 415


def test_predict_rejects_malformed_image(checkpoint_path):
    with TestClient(create_app(checkpoint_path)) as client:
        response = client.post(
            "/predict", files={"file": ("bad.png", b"not an image", "image/png")}
        )
    assert response.status_code == 400
    assert "valid PNG or JPEG" in response.json()["detail"]


def test_predict_rejects_oversized_upload(checkpoint_path):
    with TestClient(create_app(checkpoint_path)) as client:
        response = client.post(
            "/predict", files={"file": ("large.png", b"x" * 1_000_001, "image/png")}
        )
    assert response.status_code == 413


def test_model_not_loaded_is_not_healthy(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt")) as client:
        health = client.get("/health")
        prediction = client.post(
            "/predict",
            files={"file": ("red.png", _image_bytes("red"), "image/png")},
        )
    assert health.status_code == 503
    assert health.json() == {
        "status": "not_ready",
        "error": "model checkpoint could not be loaded",
    }
    assert prediction.status_code == 503
    assert "missing.pt" not in prediction.text
