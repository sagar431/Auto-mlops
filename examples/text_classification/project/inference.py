"""Inference utilities for text classification model."""

from pathlib import Path

import torch
from dataset import Vocabulary
from model import load_model


class TextClassifier:
    """Inference wrapper for the text classification model."""

    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        classes_file: str | None = None,
        device: str | None = None,
        model_type: str = "textcnn",
        **model_kwargs,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load vocabulary
        self.vocab = Vocabulary.load(vocab_path)

        # Load class names
        if classes_file and Path(classes_file).exists():
            with open(classes_file) as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            self.classes = ["negative", "positive"]

        # Load model
        self.model = load_model(
            model_path,
            model_type=model_type,
            vocab_size=len(self.vocab),
            num_classes=len(self.classes),
            device=self.device,
            **model_kwargs,
        )

    def predict(self, text: str, max_length: int = 256) -> dict:
        """Predict class for a single text.

        Args:
            text: Input text string.
            max_length: Maximum sequence length.

        Returns:
            dict with predicted class, confidence, and all probabilities.
        """
        # Encode text
        encoded = self.vocab.encode(text, max_length=max_length)
        tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)

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
            "text": text[:100] + "..." if len(text) > 100 else text,
        }

    def predict_batch(self, texts: list[str], max_length: int = 256) -> list[dict]:
        """Predict classes for multiple texts.

        Args:
            texts: List of input text strings.
            max_length: Maximum sequence length.

        Returns:
            List of prediction dicts.
        """
        return [self.predict(text, max_length) for text in texts]


def main():
    """Example usage of the classifier."""
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on texts")
    parser.add_argument("texts", nargs="+", help="Text(s) to classify")
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Path to model")
    parser.add_argument("--vocab", type=str, default="models/vocab.json", help="Path to vocabulary")
    parser.add_argument(
        "--classes", type=str, default="models/classes.txt", help="Path to classes file"
    )
    parser.add_argument(
        "--model-type", type=str, default="textcnn", help="Model type (textcnn or lstm)"
    )

    args = parser.parse_args()

    classifier = TextClassifier(
        model_path=args.model,
        vocab_path=args.vocab,
        classes_file=args.classes,
        model_type=args.model_type,
    )

    for text in args.texts:
        result = classifier.predict(text)
        print(f"\nText: {result['text']}")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  All probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()
