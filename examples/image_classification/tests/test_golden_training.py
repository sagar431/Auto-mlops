"""Tests for deterministic bounded training and checkpoint validation."""

from pathlib import Path

import pytest
import torch
from golden_train import (
    CLASS_NAMES,
    MAX_EPOCHS,
    TrainingConfig,
    train_golden,
)
from model import (
    GOLDEN_ARCHITECTURE,
    GOLDEN_SCHEMA_VERSION,
    CheckpointError,
    load_golden_checkpoint,
)


def _small_config() -> TrainingConfig:
    return TrainingConfig(epochs=2, train_samples=32, validation_samples=8, batch_size=8)


def test_bounded_training_creates_complete_checkpoint(tmp_path):
    result = train_golden(tmp_path / "golden", _small_config())
    checkpoint_path = Path(result["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert result["status"] == "succeeded"
    assert result["duration_seconds"] < 10
    assert checkpoint_path.is_file()
    assert checkpoint["schema_version"] == GOLDEN_SCHEMA_VERSION
    assert checkpoint["architecture"] == GOLDEN_ARCHITECTURE
    assert checkpoint["class_names"] == list(CLASS_NAMES)
    assert checkpoint["num_classes"] == 2
    assert checkpoint["image_size"] == 16
    assert checkpoint["normalization"] == {"mean": [0.5] * 3, "std": [0.5] * 3}
    assert checkpoint["training_config"]["device"] == "cpu"
    assert checkpoint["training_config"]["train_samples"] == 32
    assert checkpoint["metrics"]["validation_accuracy"] == pytest.approx(1.0)
    assert checkpoint["state_dict"]
    assert Path(result["training_config_path"]).is_file()
    assert Path(result["metrics_path"]).is_file()
    assert Path(result["sample_image_path"]).is_file()


def test_training_is_deterministic(tmp_path):
    first = train_golden(tmp_path / "first", _small_config())
    second = train_golden(tmp_path / "second", _small_config())
    first_checkpoint = torch.load(first["checkpoint_path"], weights_only=True)
    second_checkpoint = torch.load(second["checkpoint_path"], weights_only=True)
    assert first_checkpoint["metrics"] == second_checkpoint["metrics"]
    assert first_checkpoint["training_config"] == second_checkpoint["training_config"]
    for name, tensor in first_checkpoint["state_dict"].items():
        assert torch.equal(tensor, second_checkpoint["state_dict"][name])


def test_training_rejects_unbounded_or_gpu_configuration(tmp_path):
    with pytest.raises(ValueError, match="epochs must be between"):
        train_golden(tmp_path, TrainingConfig(epochs=MAX_EPOCHS + 1))
    with pytest.raises(ValueError, match="CPU training only"):
        train_golden(tmp_path, TrainingConfig(device="cuda"))


def test_checkpoint_load_returns_validated_model(tmp_path):
    result = train_golden(tmp_path, _small_config())
    loaded = load_golden_checkpoint(result["checkpoint_path"])
    assert loaded.schema_version == GOLDEN_SCHEMA_VERSION
    assert loaded.architecture == GOLDEN_ARCHITECTURE
    assert loaded.class_names == CLASS_NAMES
    assert loaded.model.training is False


def test_missing_and_corrupt_checkpoints_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint was not found"):
        load_golden_checkpoint(tmp_path / "missing.pt")
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointError, match="Unable to read"):
        load_golden_checkpoint(corrupt)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "unknown", "Unsupported checkpoint schema"),
        ("architecture", "resnet18", "Incompatible checkpoint architecture"),
        ("class_names", ["red", "red"], "Invalid checkpoint class metadata"),
        ("state_dict", {}, "state dictionary is incompatible"),
    ],
)
def test_malformed_checkpoint_metadata_fails_clearly(tmp_path, field, value, message):
    result = train_golden(tmp_path / "valid", _small_config())
    checkpoint = torch.load(result["checkpoint_path"], weights_only=True)
    checkpoint[field] = value
    malformed = tmp_path / f"malformed-{field}.pt"
    torch.save(checkpoint, malformed)
    with pytest.raises(CheckpointError, match=message):
        load_golden_checkpoint(malformed)
