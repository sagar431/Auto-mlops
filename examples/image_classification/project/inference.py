"""Inference utilities for image classification model."""

from pathlib import Path

import torch
from model import load_model
from PIL import Image
from torchvision import transforms


class ImageClassifier:
    """Inference wrapper for the image classification model."""

    def __init__(
        self,
        model_path: str,
        classes_file: str | None = None,
        device: str | None = None,
        image_size: int = 224,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # Load class names
        if classes_file and Path(classes_file).exists():
            with open(classes_file) as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            self.classes = ["cat", "dog"]

        # Load model
        self.model = load_model(model_path, num_classes=len(self.classes), device=self.device)

        # Setup transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image: str | Path | Image.Image) -> dict:
        """Predict class for a single image.

        Args:
            image: Path to image file or PIL Image

        Returns:
            dict with predicted class, confidence, and all probabilities
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Transform and add batch dimension
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = probabilities.max(0)

        predicted_class = self.classes[predicted_idx.item()]
        all_probs = {cls: prob.item() for cls, prob in zip(self.classes, probabilities)}

        return {
            "predicted_class": predicted_class,
            "confidence": confidence.item(),
            "probabilities": all_probs,
        }

    def predict_batch(self, images: list) -> list:
        """Predict classes for multiple images.

        Args:
            images: List of image paths or PIL Images

        Returns:
            List of prediction dicts
        """
        return [self.predict(img) for img in images]


def main():
    """Example usage of the classifier."""
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("images", nargs="+", help="Path(s) to image files")
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Path to model")
    parser.add_argument(
        "--classes", type=str, default="models/classes.txt", help="Path to classes file"
    )

    args = parser.parse_args()

    classifier = ImageClassifier(model_path=args.model, classes_file=args.classes)

    for image_path in args.images:
        result = classifier.predict(image_path)
        print(f"\n{image_path}:")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  All probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()
