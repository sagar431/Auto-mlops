"""Canonical checkpoint-driven inference for the golden image-classification slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

try:
    from .model import LoadedCheckpoint, load_golden_checkpoint
except ImportError:  # Support direct execution from the project directory.
    from model import LoadedCheckpoint, load_golden_checkpoint


class InvalidImageError(ValueError):
    """Raised when an input is not a decodable PNG or JPEG image."""


class ImageClassifier:
    """Load one validated checkpoint and execute real CPU predictions."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        if device != "cpu":
            raise ValueError("The golden slice supports CPU inference only")
        self.device = torch.device(device)
        self.checkpoint: LoadedCheckpoint = load_golden_checkpoint(model_path, device=device)
        self.model = self.checkpoint.model
        self.classes = list(self.checkpoint.class_names)
        self.image_size = self.checkpoint.image_size
        self.normalization = self.checkpoint.normalization
        self.schema_version = self.checkpoint.schema_version
        self.architecture = self.checkpoint.architecture

    def _load_image(self, image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if not isinstance(image, (str, Path)):
            raise InvalidImageError("Image must be a filesystem path or PIL image")
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError("Input image was not found")
        try:
            with Image.open(image_path) as opened:
                opened.verify()
            with Image.open(image_path) as opened:
                return opened.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("Input is not a valid PNG or JPEG image") from exc

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        byte_values = bytearray(resized.tobytes())
        tensor = torch.frombuffer(byte_values, dtype=torch.uint8).clone()
        tensor = tensor.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1)
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
        mean = torch.tensor(self.normalization["mean"], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.normalization["std"], dtype=torch.float32).view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0).to(self.device)

    def predict(self, image: str | Path | Image.Image) -> dict[str, object]:
        """Return the canonical prediction schema for one image."""
        tensor = self._preprocess(self._load_image(image))
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_index = probabilities.max(dim=0)
        all_probabilities = {
            class_name: float(probability.item())
            for class_name, probability in zip(self.classes, probabilities, strict=True)
        }
        return {
            "predicted_class": self.classes[int(predicted_index.item())],
            "confidence": float(confidence.item()),
            "probabilities": all_probabilities,
        }

    def predict_batch(self, images: list[str | Path | Image.Image]) -> list[dict[str, object]]:
        return [self.predict(image) for image in images]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    try:
        classifier = ImageClassifier(args.model)
        for image_path in args.images:
            print(json.dumps(classifier.predict(image_path), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
