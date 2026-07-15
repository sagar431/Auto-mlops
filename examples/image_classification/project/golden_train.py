"""Bounded deterministic CPU training for the golden image-classification slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from .golden_data import load_and_verify_manifest
    from .model import GOLDEN_ARCHITECTURE, GOLDEN_SCHEMA_VERSION, create_golden_model
except ImportError:  # Support direct execution from the project directory.
    from golden_data import load_and_verify_manifest
    from model import GOLDEN_ARCHITECTURE, GOLDEN_SCHEMA_VERSION, create_golden_model

CLASS_NAMES = ("red", "blue")
IMAGE_SIZE = 16
NORMALIZATION = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
MAX_EPOCHS = 5
MAX_TRAIN_SAMPLES = 128
MAX_VALIDATION_SAMPLES = 64
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "golden"


@dataclass(frozen=True)
class TrainingConfig:
    """Effective bounded training controls persisted with the artifact."""

    seed: int = 17
    epochs: int = 3
    train_samples: int = 64
    validation_samples: int = 16
    batch_size: int = 8
    learning_rate: float = 0.05
    device: str = "cpu"
    image_size: int = IMAGE_SIZE

    def validate(self) -> None:
        if not 1 <= self.epochs <= MAX_EPOCHS:
            raise ValueError(f"epochs must be between 1 and {MAX_EPOCHS}")
        if not 2 <= self.train_samples <= MAX_TRAIN_SAMPLES:
            raise ValueError(f"train_samples must be between 2 and {MAX_TRAIN_SAMPLES}")
        if not 2 <= self.validation_samples <= MAX_VALIDATION_SAMPLES:
            raise ValueError(
                f"validation_samples must be between 2 and {MAX_VALIDATION_SAMPLES}"
            )
        if self.batch_size < 1 or self.batch_size > self.train_samples:
            raise ValueError("batch_size must be positive and no larger than train_samples")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.device != "cpu":
            raise ValueError("the golden slice supports CPU training only")


def set_deterministic_seed(seed: int) -> None:
    """Configure all random sources used by this training path."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def create_synthetic_dataset(sample_count: int, seed: int) -> TensorDataset:
    """Create deterministic red/blue tensors without network or filesystem access."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(sample_count, dtype=torch.long) % len(CLASS_NAMES)
    images = torch.rand(
        (sample_count, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator, dtype=torch.float32
    ) * 0.08
    for index, label in enumerate(labels.tolist()):
        images[index, label * 2, :, :] += 0.88
        images[index, 1, 4:12, 4:12] += 0.04
    images.clamp_(0.0, 1.0)
    mean = torch.tensor(NORMALIZATION["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZATION["std"]).view(1, 3, 1, 1)
    return TensorDataset((images - mean) / std, labels)


class ManifestImageDataset(Dataset):
    """Load only the image files declared by the verified dataset manifest."""

    def __init__(self, dataset_dir: Path, entries: list[dict[str, object]]):
        self.dataset_dir = dataset_dir
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[index]
        with Image.open(self.dataset_dir / str(entry["path"])) as image:
            rgb_image = image.convert("RGB")
            if rgb_image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError("Golden dataset contains an image with invalid dimensions")
            byte_values = bytearray(rgb_image.tobytes())
        tensor = torch.frombuffer(byte_values, dtype=torch.uint8).clone()
        tensor = tensor.view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1)
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
        mean = torch.tensor(NORMALIZATION["mean"], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(NORMALIZATION["std"], dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std, int(entry["class_index"])


def _synthetic_lineage(config: TrainingConfig) -> dict[str, object]:
    payload = {
        "source": "deterministic-synthetic-tensors",
        "seed": config.seed,
        "train_samples": config.train_samples,
        "validation_samples": config.validation_samples,
        "image_size": config.image_size,
    }
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"schema_version": "golden-synthetic-tensors.v1", **payload, "dataset_checksum": checksum}


def _evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    samples = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            samples += labels.size(0)
    return total_loss / samples, correct / samples


def train_golden(
    output_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    config: TrainingConfig | None = None,
    dataset_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Train, evaluate, and persist one canonical golden checkpoint."""
    effective = config or TrainingConfig()
    effective.validate()
    set_deterministic_seed(effective.seed)
    started = time.monotonic()

    if dataset_dir is None:
        if manifest_path is not None:
            raise ValueError("manifest_path requires dataset_dir")
        train_dataset = create_synthetic_dataset(effective.train_samples, effective.seed)
        validation_dataset = create_synthetic_dataset(
            effective.validation_samples, effective.seed + 1
        )
        dataset_lineage = _synthetic_lineage(effective)
    else:
        dataset_root = Path(dataset_dir).resolve()
        manifest, dataset_lineage = load_and_verify_manifest(dataset_root, manifest_path)
        generation_config = manifest.get("generation_config", {})
        sample_counts = manifest.get("sample_counts", {})
        if generation_config.get("seed") != effective.seed:
            raise ValueError("Training seed does not match the dataset manifest")
        if sample_counts.get("train") != effective.train_samples:
            raise ValueError("Training sample count does not match the dataset manifest")
        if sample_counts.get("validation") != effective.validation_samples:
            raise ValueError("Validation sample count does not match the dataset manifest")
        train_entries = [entry for entry in manifest["files"] if entry["split"] == "train"]
        validation_entries = [
            entry for entry in manifest["files"] if entry["split"] == "validation"
        ]
        train_dataset = ManifestImageDataset(dataset_root, train_entries)
        validation_dataset = ManifestImageDataset(dataset_root, validation_entries)
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(effective.seed),
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=effective.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = create_golden_model(num_classes=len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=effective.learning_rate)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, effective.epochs + 1):
        model.train()
        epoch_loss = 0.0
        samples = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * labels.size(0)
            samples += labels.size(0)
        validation_loss, validation_accuracy = _evaluate(model, validation_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / samples,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

    final_metrics = {
        "validation_loss": float(history[-1]["validation_loss"]),
        "validation_accuracy": float(history[-1]["validation_accuracy"]),
    }
    artifact_dir = Path(output_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "model.pt"
    config_path = artifact_dir / "training_config.json"
    metrics_path = artifact_dir / "metrics.json"
    lineage_path = artifact_dir / "lineage.json"
    sample_image_path = artifact_dir / "sample-red.png"
    checkpoint = {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "architecture": GOLDEN_ARCHITECTURE,
        "state_dict": model.state_dict(),
        "class_names": list(CLASS_NAMES),
        "num_classes": len(CLASS_NAMES),
        "image_size": IMAGE_SIZE,
        "normalization": NORMALIZATION,
        "training_config": asdict(effective),
        "metrics": final_metrics,
        "dataset_lineage": dataset_lineage,
    }
    torch.save(checkpoint, checkpoint_path)
    config_path.write_text(json.dumps(asdict(effective), indent=2, sort_keys=True) + "\n")
    metrics_payload = {"history": history, "final": final_metrics}
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n")
    lineage_path.write_text(json.dumps(dataset_lineage, indent=2, sort_keys=True) + "\n")
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="red").save(sample_image_path)

    result: dict[str, object] = {
        "status": "succeeded",
        "duration_seconds": round(time.monotonic() - started, 4),
        "checkpoint_path": str(checkpoint_path),
        "training_config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "lineage_path": str(lineage_path),
        "sample_image_path": str(sample_image_path),
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "architecture": GOLDEN_ARCHITECTURE,
        "class_names": list(CLASS_NAMES),
        "image_size": IMAGE_SIZE,
        "training_config": asdict(effective),
        "metrics": final_metrics,
        "dataset_lineage": {
            key: value for key, value in dataset_lineage.items() if key != "file_checksums"
        },
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--train-samples", type=int, default=TrainingConfig.train_samples)
    parser.add_argument(
        "--validation-samples", type=int, default=TrainingConfig.validation_samples
    )
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = train_golden(
            output_dir=args.output_dir,
            dataset_dir=args.dataset_dir,
            manifest_path=args.manifest,
            config=TrainingConfig(
                seed=args.seed,
                epochs=args.epochs,
                train_samples=args.train_samples,
                validation_samples=args.validation_samples,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            ),
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
