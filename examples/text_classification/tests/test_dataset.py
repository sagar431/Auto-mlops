"""Tests for the dataset utilities."""

import sys
from pathlib import Path

import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))

from dataset import (
    TextClassificationDataset,
    Vocabulary,
    collate_fn,
    create_data_loaders,
    create_synthetic_data,
)


class TestVocabulary:
    """Tests for Vocabulary class."""

    def test_vocabulary_creation(self):
        """Test creating vocabulary with default parameters."""
        vocab = Vocabulary()
        assert len(vocab) == 2  # pad and unk tokens
        assert vocab.pad_idx == 0
        assert vocab.unk_idx == 1

    def test_vocabulary_build(self):
        """Test building vocabulary from texts."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        texts = ["hello world", "hello there", "world is great"]
        vocab.build(texts)

        assert "hello" in vocab.word2idx
        assert "world" in vocab.word2idx
        assert len(vocab) > 2

    def test_vocabulary_build_min_freq(self):
        """Test that min_freq filters rare words."""
        vocab = Vocabulary(max_size=100, min_freq=2)
        texts = ["hello world", "hello there", "world is great"]
        vocab.build(texts)

        # "hello" and "world" appear twice
        assert "hello" in vocab.word2idx
        assert "world" in vocab.word2idx
        # "there", "is", "great" appear once
        assert "there" not in vocab.word2idx

    def test_vocabulary_encode(self):
        """Test encoding text to indices."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        texts = ["hello world"]
        vocab.build(texts)

        encoded = vocab.encode("hello world")
        assert len(encoded) == 2
        assert all(isinstance(idx, int) for idx in encoded)

    def test_vocabulary_encode_unknown(self):
        """Test encoding with unknown words."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        texts = ["hello world"]
        vocab.build(texts)

        encoded = vocab.encode("hello unknown world")
        assert vocab.unk_idx in encoded

    def test_vocabulary_encode_max_length(self):
        """Test encoding with max_length truncation."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        texts = ["a b c d e"]
        vocab.build(texts)

        encoded = vocab.encode("a b c d e", max_length=3)
        assert len(encoded) == 3

    def test_vocabulary_decode(self):
        """Test decoding indices back to text."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        texts = ["hello world"]
        vocab.build(texts)

        encoded = vocab.encode("hello world")
        decoded = vocab.decode(encoded)
        assert "hello" in decoded
        assert "world" in decoded

    def test_vocabulary_save_load(self, tmp_path):
        """Test saving and loading vocabulary."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        texts = ["hello world", "hello there"]
        vocab.build(texts)

        vocab_path = tmp_path / "vocab.json"
        vocab.save(str(vocab_path))

        loaded_vocab = Vocabulary.load(str(vocab_path))

        assert len(loaded_vocab) == len(vocab)
        assert loaded_vocab.word2idx == vocab.word2idx

    def test_tokenize(self):
        """Test tokenization."""
        tokens = Vocabulary.tokenize("Hello, World! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens


class TestTextClassificationDataset:
    """Tests for TextClassificationDataset."""

    def test_dataset_creation(self):
        """Test creating dataset."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        texts = ["hello world", "goodbye world"]
        vocab.build(texts)

        dataset = TextClassificationDataset(
            texts=texts,
            labels=[0, 1],
            vocab=vocab,
            max_length=50,
        )

        assert len(dataset) == 2

    def test_dataset_getitem(self):
        """Test getting item from dataset."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        texts = ["hello world", "goodbye world"]
        vocab.build(texts)

        dataset = TextClassificationDataset(
            texts=texts,
            labels=[0, 1],
            vocab=vocab,
            max_length=50,
        )

        encoded, label = dataset[0]
        assert isinstance(encoded, torch.Tensor)
        assert isinstance(label, int)
        assert label == 0


class TestCollateFn:
    """Tests for collate function."""

    def test_collate_fn_padding(self):
        """Test that collate function pads sequences."""
        # Different length sequences
        batch = [
            (torch.tensor([1, 2, 3]), 0),
            (torch.tensor([4, 5]), 1),
            (torch.tensor([6, 7, 8, 9]), 0),
        ]

        padded, labels = collate_fn(batch, pad_idx=0)

        assert padded.shape == (3, 4)  # batch_size x max_len
        assert labels.shape == (3,)

    def test_collate_fn_preserves_content(self):
        """Test that collate function preserves content."""
        batch = [
            (torch.tensor([1, 2, 3]), 0),
            (torch.tensor([4, 5]), 1),
        ]

        padded, labels = collate_fn(batch, pad_idx=0)

        # First sequence should be [1, 2, 3]
        assert torch.equal(padded[0, :3], torch.tensor([1, 2, 3]))
        # Second sequence should be [4, 5, 0] (padded)
        assert torch.equal(padded[1, :2], torch.tensor([4, 5]))
        assert padded[1, 2] == 0  # Padding


class TestSyntheticData:
    """Tests for synthetic data creation."""

    def test_create_synthetic_data(self):
        """Test creating synthetic data."""
        texts, labels = create_synthetic_data(
            num_samples=100,
            num_classes=2,
            vocab_size=500,
            max_length=30,
        )

        assert len(texts) == 100
        assert len(labels) == 100
        assert all(0 <= label < 2 for label in labels)
        assert all(isinstance(text, str) for text in texts)

    def test_synthetic_data_multiclass(self):
        """Test creating synthetic data with multiple classes."""
        texts, labels = create_synthetic_data(
            num_samples=150,
            num_classes=3,
            vocab_size=500,
            max_length=30,
        )

        assert len(texts) == 150
        assert all(0 <= label < 3 for label in labels)


class TestCreateDataLoaders:
    """Tests for data loader creation."""

    def test_create_data_loaders(self):
        """Test creating data loaders."""
        train_texts = ["hello world", "goodbye world", "test text", "more text"]
        train_labels = [0, 1, 0, 1]
        test_texts = ["new hello", "new goodbye"]
        test_labels = [0, 1]

        train_loader, test_loader, vocab = create_data_loaders(
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            batch_size=2,
            max_length=50,
            num_workers=0,
        )

        assert len(vocab) > 2  # More than just special tokens

        # Test that we can iterate
        batch = next(iter(train_loader))
        assert len(batch) == 2  # texts and labels
        texts, labels = batch
        assert texts.shape[0] == 2  # batch size

    def test_create_data_loaders_with_vocab(self):
        """Test creating data loaders with pre-built vocabulary."""
        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build(["hello world", "goodbye world"])

        train_texts = ["hello world", "goodbye world"]
        train_labels = [0, 1]
        test_texts = ["new hello"]
        test_labels = [0]

        train_loader, test_loader, returned_vocab = create_data_loaders(
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            vocab=vocab,
            batch_size=2,
            max_length=50,
            num_workers=0,
        )

        assert returned_vocab is vocab
