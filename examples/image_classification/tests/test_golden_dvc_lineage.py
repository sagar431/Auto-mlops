"""Tests for deterministic file-backed golden dataset lineage."""

import json
from pathlib import Path

import pytest
import torch
from golden_data import DatasetConfig, load_and_verify_manifest, prepare_golden_dataset
from golden_train import TrainingConfig, train_golden
from model import CheckpointError, load_golden_checkpoint


def _config() -> DatasetConfig:
    return DatasetConfig(seed=17, train_samples=16, validation_samples=8)


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        seed=17,
        epochs=2,
        train_samples=16,
        validation_samples=8,
        batch_size=4,
    )


def test_file_dataset_preparation_is_deterministic(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = prepare_golden_dataset(first_dir, _config())
    second = prepare_golden_dataset(second_dir, _config())

    assert first == second
    assert first["class_names"] == ["red", "blue"]
    assert first["sample_counts"] == {"train": 16, "validation": 8}
    assert len(first["files"]) == 24
    assert len(first["dataset_checksum"]) == 64
    for entry in first["files"]:
        assert (first_dir / entry["path"]).read_bytes() == (
            second_dir / entry["path"]
        ).read_bytes()


def test_manifest_verification_rejects_tampered_image(tmp_path):
    dataset_dir = tmp_path / "dataset"
    manifest = prepare_golden_dataset(dataset_dir, _config())
    tampered_path = dataset_dir / manifest["files"][0]["path"]
    tampered_path.write_bytes(b"not the declared image")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_and_verify_manifest(dataset_dir)


def test_training_reads_files_and_embeds_complete_lineage(tmp_path):
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "artifacts"
    manifest = prepare_golden_dataset(dataset_dir, _config())
    result = train_golden(
        artifact_dir,
        _training_config(),
        dataset_dir=dataset_dir,
        manifest_path=dataset_dir / "manifest.json",
    )

    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
    lineage = checkpoint["dataset_lineage"]
    assert checkpoint["metrics"]["validation_accuracy"] == pytest.approx(1.0)
    assert lineage["source"] == "dvc-materialized-image-files"
    assert lineage["dataset_checksum"] == manifest["dataset_checksum"]
    assert len(lineage["manifest_checksum"]) == 64
    assert len(lineage["file_checksums"]) == 24
    assert json.loads(Path(result["lineage_path"]).read_text()) == lineage

    loaded = load_golden_checkpoint(result["checkpoint_path"])
    assert loaded.dataset_lineage == lineage


def test_training_rejects_manifest_configuration_mismatch(tmp_path):
    dataset_dir = tmp_path / "dataset"
    prepare_golden_dataset(dataset_dir, _config())
    mismatched = TrainingConfig(
        seed=18,
        epochs=1,
        train_samples=16,
        validation_samples=8,
        batch_size=4,
    )

    with pytest.raises(ValueError, match="seed does not match"):
        train_golden(tmp_path / "artifacts", mismatched, dataset_dir=dataset_dir)


def test_file_backed_training_outputs_are_byte_deterministic(tmp_path):
    dataset_dir = tmp_path / "dataset"
    prepare_golden_dataset(dataset_dir, _config())
    first = train_golden(tmp_path / "first", _training_config(), dataset_dir=dataset_dir)
    second = train_golden(tmp_path / "second", _training_config(), dataset_dir=dataset_dir)

    for result_key in (
        "checkpoint_path",
        "training_config_path",
        "metrics_path",
        "lineage_path",
        "sample_image_path",
    ):
        assert Path(first[result_key]).read_bytes() == Path(second[result_key]).read_bytes()


def test_checkpoint_loader_rejects_invalid_file_lineage_checksum(tmp_path):
    dataset_dir = tmp_path / "dataset"
    prepare_golden_dataset(dataset_dir, _config())
    result = train_golden(tmp_path / "artifacts", _training_config(), dataset_dir=dataset_dir)
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
    first_path = next(iter(checkpoint["dataset_lineage"]["file_checksums"]))
    checkpoint["dataset_lineage"]["file_checksums"][first_path] = "invalid"
    malformed = tmp_path / "malformed.pt"
    torch.save(checkpoint, malformed)

    with pytest.raises(CheckpointError, match="file-backed dataset lineage is invalid"):
        load_golden_checkpoint(malformed)
