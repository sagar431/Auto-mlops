"""Deterministic file-backed dataset preparation for the golden slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

DATASET_SCHEMA_VERSION = "golden-red-blue-dataset.v1"
CLASS_NAMES = ("red", "blue")
IMAGE_SIZE = 16
MAX_TRAIN_SAMPLES = 128
MAX_VALIDATION_SAMPLES = 64


@dataclass(frozen=True)
class DatasetConfig:
    """Bounded controls used to generate the local image dataset."""

    seed: int = 17
    train_samples: int = 64
    validation_samples: int = 16
    image_size: int = IMAGE_SIZE

    def validate(self) -> None:
        if self.image_size != IMAGE_SIZE:
            raise ValueError(f"image_size must be {IMAGE_SIZE}")
        if not 2 <= self.train_samples <= MAX_TRAIN_SAMPLES:
            raise ValueError(f"train_samples must be between 2 and {MAX_TRAIN_SAMPLES}")
        if not 2 <= self.validation_samples <= MAX_VALIDATION_SAMPLES:
            raise ValueError(
                f"validation_samples must be between 2 and {MAX_VALIDATION_SAMPLES}"
            )
        if self.train_samples % len(CLASS_NAMES) != 0:
            raise ValueError("train_samples must be divisible by the class count")
        if self.validation_samples % len(CLASS_NAMES) != 0:
            raise ValueError("validation_samples must be divisible by the class count")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _canonical_checksum(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _image_bytes(class_index: int, sample_index: int, split: str, seed: int) -> bytes:
    split_offset = 0 if split == "train" else 1_000_000
    generator = random.Random(seed + split_offset + class_index * 10_000 + sample_index)
    pixels = bytearray()
    dominant_channel = class_index * 2
    for _ in range(IMAGE_SIZE * IMAGE_SIZE):
        channels = [generator.randrange(0, 21) for _ in range(3)]
        channels[dominant_channel] = generator.randrange(220, 256)
        channels[1] = min(255, channels[1] + 8)
        pixels.extend(channels)
    return bytes(pixels)


def prepare_golden_dataset(
    output_dir: str | Path,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    """Materialize deterministic PNG files and a content-addressed manifest."""
    effective = config or DatasetConfig()
    effective.validate()
    dataset_dir = Path(output_dir).resolve()
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    split_counts = {
        "train": effective.train_samples,
        "validation": effective.validation_samples,
    }
    for split, sample_count in split_counts.items():
        samples_per_class = sample_count // len(CLASS_NAMES)
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_dir = dataset_dir / split / class_name
            class_dir.mkdir(parents=True)
            for sample_index in range(samples_per_class):
                relative_path = Path(split) / class_name / f"{class_name}_{sample_index:03d}.png"
                image = Image.frombytes(
                    "RGB",
                    (IMAGE_SIZE, IMAGE_SIZE),
                    _image_bytes(class_index, sample_index, split, effective.seed),
                )
                image.save(dataset_dir / relative_path, format="PNG", optimize=False)
                entries.append(
                    {
                        "path": relative_path.as_posix(),
                        "split": split,
                        "class_name": class_name,
                        "class_index": class_index,
                        "sha256": sha256_file(dataset_dir / relative_path),
                    }
                )

    entries.sort(key=lambda item: item["path"])
    identity = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "class_names": list(CLASS_NAMES),
        "image_size": IMAGE_SIZE,
        "generation_config": asdict(effective),
        "files": entries,
    }
    manifest = {
        **identity,
        "dataset_id": f"golden-red-blue-seed-{effective.seed}",
        "dataset_checksum": _canonical_checksum(identity),
        "sample_counts": split_counts,
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_and_verify_manifest(
    dataset_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate manifest metadata and every file checksum before training."""
    root = Path(dataset_dir).resolve()
    path = Path(manifest_path).resolve() if manifest_path else root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Golden dataset manifest was not found")
    if path.parent != root:
        raise ValueError("Golden dataset manifest must be stored at the dataset root")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Golden dataset manifest is unreadable") from exc
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Unsupported golden dataset schema version")
    if manifest.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Golden dataset class metadata is invalid")
    if manifest.get("image_size") != IMAGE_SIZE:
        raise ValueError("Golden dataset image size is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Golden dataset file manifest is empty or invalid")

    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Golden dataset file metadata is invalid")
        relative_path = Path(entry["path"])
        split = entry.get("split")
        class_name = entry.get("class_name")
        class_index = entry.get("class_index")
        checksum = entry.get("sha256")
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 3
            or relative_path.suffix.lower() != ".png"
        ):
            raise ValueError("Golden dataset manifest contains an unsafe path")
        if (
            split not in {"train", "validation"}
            or class_name not in CLASS_NAMES
            or class_index != CLASS_NAMES.index(class_name)
            or relative_path.parts[:2] != (split, class_name)
            or not _is_sha256(checksum)
        ):
            raise ValueError("Golden dataset file metadata is invalid")
        if relative_path.as_posix() in expected_paths:
            raise ValueError("Golden dataset manifest contains duplicate files")
        image_path = root / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Golden dataset file is missing: {relative_path.as_posix()}")
        if sha256_file(image_path) != entry.get("sha256"):
            raise ValueError(f"Golden dataset checksum mismatch: {relative_path.as_posix()}")
        expected_paths.add(relative_path.as_posix())

    observed_counts = {
        split: sum(1 for entry in entries if entry["split"] == split)
        for split in ("train", "validation")
    }
    if observed_counts != manifest.get("sample_counts"):
        raise ValueError("Golden dataset sample counts are invalid")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_paths != expected_paths:
        raise ValueError("Golden dataset contents do not match the manifest")

    identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "class_names",
            "image_size",
            "generation_config",
            "files",
        )
    }
    if _canonical_checksum(identity) != manifest.get("dataset_checksum"):
        raise ValueError("Golden dataset aggregate checksum is invalid")
    lineage = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source": "dvc-materialized-image-files",
        "dataset_id": manifest.get("dataset_id"),
        "dataset_checksum": manifest["dataset_checksum"],
        "manifest_checksum": sha256_file(path),
        "manifest_file": path.name,
        "sample_counts": manifest.get("sample_counts"),
        "file_checksums": {entry["path"]: entry["sha256"] for entry in entries},
    }
    return manifest, lineage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DatasetConfig.seed)
    parser.add_argument("--train-samples", type=int, default=DatasetConfig.train_samples)
    parser.add_argument(
        "--validation-samples", type=int, default=DatasetConfig.validation_samples
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = prepare_golden_dataset(
            args.output_dir,
            DatasetConfig(
                seed=args.seed,
                train_samples=args.train_samples,
                validation_samples=args.validation_samples,
            ),
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "succeeded",
                "dataset_id": manifest["dataset_id"],
                "dataset_checksum": manifest["dataset_checksum"],
                "sample_counts": manifest["sample_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
