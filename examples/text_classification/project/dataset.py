"""Dataset utilities for text classification."""

import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# Default class names for binary sentiment
SENTIMENT_CLASSES = ["negative", "positive"]


class Vocabulary:
    """Vocabulary for text tokenization.

    Maintains mappings between words and indices.
    """

    def __init__(
        self,
        max_size: int = 30000,
        min_freq: int = 2,
        special_tokens: list[str] | None = None,
    ):
        self.max_size = max_size
        self.min_freq = min_freq

        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]

        self.special_tokens = special_tokens
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}

        # Initialize special tokens
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        self.pad_idx = self.word2idx.get("<pad>", 0)
        self.unk_idx = self.word2idx.get("<unk>", 1)

    def build(self, texts: list[str]) -> None:
        """Build vocabulary from texts.

        Args:
            texts: List of text strings.
        """
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)

        # Filter by frequency and limit size
        filtered_words = [
            word for word, count in word_counts.most_common() if count >= self.min_freq
        ]

        # Add words to vocabulary
        current_idx = len(self.special_tokens)
        for word in filtered_words[: self.max_size - len(self.special_tokens)]:
            self.word2idx[word] = current_idx
            self.idx2word[current_idx] = word
            current_idx += 1

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple tokenization by splitting on whitespace and punctuation."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """Convert text to list of indices.

        Args:
            text: Input text string.
            max_length: Maximum sequence length (truncate if longer).

        Returns:
            List of token indices.
        """
        tokens = self.tokenize(text)
        if max_length is not None:
            tokens = tokens[:max_length]

        return [self.word2idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: list[int]) -> str:
        """Convert list of indices back to text.

        Args:
            indices: List of token indices.

        Returns:
            Decoded text string.
        """
        tokens = [self.idx2word.get(idx, "<unk>") for idx in indices if idx != self.pad_idx]
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str) -> None:
        """Save vocabulary to file."""
        data = {
            "word2idx": self.word2idx,
            "max_size": self.max_size,
            "min_freq": self.min_freq,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from file."""
        with open(path) as f:
            data = json.load(f)

        vocab = cls(
            max_size=data["max_size"],
            min_freq=data["min_freq"],
            special_tokens=data["special_tokens"],
        )
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(idx): word for word, idx in data["word2idx"].items()}
        return vocab


class TextClassificationDataset(Dataset):
    """Dataset for text classification.

    Args:
        texts: List of text strings.
        labels: List of integer labels.
        vocab: Vocabulary for encoding.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Encode text
        encoded = self.vocab.encode(text, max_length=self.max_length)

        return torch.tensor(encoded, dtype=torch.long), label


def collate_fn(
    batch: list[tuple[torch.Tensor, int]], pad_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader.

    Pads sequences to the same length within a batch.

    Args:
        batch: List of (sequence, label) tuples.
        pad_idx: Index used for padding.

    Returns:
        Tuple of (padded_sequences, labels).
    """
    sequences, labels = zip(*batch)

    # Find max length in batch
    max_len = max(len(seq) for seq in sequences)

    # Pad sequences
    padded = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq

    labels = torch.tensor(labels, dtype=torch.long)

    return padded, labels


def load_imdb_data(data_dir: str) -> tuple[list[str], list[int], list[str], list[int]]:
    """Load IMDB movie review dataset from local directory.

    Expects directory structure:
    data_dir/
      train/
        pos/
          *.txt
        neg/
          *.txt
      test/
        pos/
          *.txt
        neg/
          *.txt

    Args:
        data_dir: Path to IMDB data directory.

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels).
    """
    data_path = Path(data_dir)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for split, texts, labels in [
        ("train", train_texts, train_labels),
        ("test", test_texts, test_labels),
    ]:
        for label_idx, label_name in enumerate(["neg", "pos"]):
            label_dir = data_path / split / label_name
            if label_dir.exists():
                for file_path in label_dir.glob("*.txt"):
                    with open(file_path, encoding="utf-8") as f:
                        texts.append(f.read().strip())
                        labels.append(label_idx)

    return train_texts, train_labels, test_texts, test_labels


def create_synthetic_data(
    num_samples: int = 1000,
    num_classes: int = 2,
    vocab_size: int = 1000,
    max_length: int = 50,
) -> tuple[list[str], list[int]]:
    """Create synthetic data for testing.

    Args:
        num_samples: Number of samples to generate.
        num_classes: Number of classes.
        vocab_size: Size of vocabulary to use.
        max_length: Maximum sequence length.

    Returns:
        Tuple of (texts, labels).
    """
    import random

    words = [f"word{i}" for i in range(vocab_size)]
    texts = []
    labels = []

    for _ in range(num_samples):
        length = random.randint(10, max_length)
        text = " ".join(random.choices(words, k=length))
        texts.append(text)
        labels.append(random.randint(0, num_classes - 1))

    return texts, labels


def create_data_loaders(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    vocab: Vocabulary | None = None,
    batch_size: int = 32,
    max_length: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, Vocabulary]:
    """Create train and test data loaders.

    Args:
        train_texts: List of training texts.
        train_labels: List of training labels.
        test_texts: List of test texts.
        test_labels: List of test labels.
        vocab: Optional pre-built vocabulary.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, test_loader, vocab).
    """
    # Build vocabulary from training data if not provided
    if vocab is None:
        vocab = Vocabulary()
        vocab.build(train_texts)

    # Create datasets
    train_dataset = TextClassificationDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, vocab, max_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx),
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx),
        pin_memory=True,
    )

    return train_loader, test_loader, vocab
