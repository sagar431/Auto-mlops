"""Training script for text classification using HuggingFace Transformers and IMDB dataset."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def load_imdb_huggingface(max_samples: int | None = None) -> tuple:
    """Load IMDB dataset using HuggingFace datasets library.

    Args:
        max_samples: Optional limit on number of samples per split.

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels, class_names).
    """
    from datasets import load_dataset

    log.info("Loading IMDB dataset from HuggingFace...")
    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    if max_samples is not None:
        train_texts = train_texts[:max_samples]
        train_labels = train_labels[:max_samples]
        test_texts = test_texts[:max_samples]
        test_labels = test_labels[:max_samples]

    class_names = ["negative", "positive"]
    return train_texts, train_labels, test_texts, test_labels, class_names


def create_huggingface_tokenizer(
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 256,
):
    """Create a HuggingFace tokenizer.

    Args:
        tokenizer_name: Name of the pretrained tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Tokenizer instance and vocab size.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer, tokenizer.vocab_size


class HuggingFaceTextDataset(torch.utils.data.Dataset):
    """Dataset for text classification using HuggingFace tokenizer.

    Args:
        texts: List of text strings.
        labels: List of integer labels.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return encoding["input_ids"].squeeze(0), label


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc


def train(
    data_dir: str = "data",
    output_dir: str = "models",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_type: str = "textcnn",
    vocab_size: int = 30000,
    embedding_dim: int = 128,
    num_classes: int = 2,
    max_length: int = 256,
    dropout: float = 0.5,
    num_workers: int = 4,
    seed: int = 42,
    use_synthetic: bool = False,
    use_huggingface: bool = True,
    tokenizer_name: str = "bert-base-uncased",
    max_samples: int | None = None,
    **model_kwargs,
) -> dict:
    """Main training function.

    Args:
        data_dir: Directory containing the data.
        output_dir: Directory to save outputs.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        model_type: Type of model ('textcnn' or 'lstm').
        vocab_size: Maximum vocabulary size (used with custom vocab).
        embedding_dim: Embedding dimension.
        num_classes: Number of output classes.
        max_length: Maximum sequence length.
        dropout: Dropout probability.
        num_workers: Number of data loading workers.
        seed: Random seed.
        use_synthetic: Whether to use synthetic data for testing.
        use_huggingface: Whether to use HuggingFace datasets and tokenizer.
        tokenizer_name: Name of HuggingFace tokenizer to use.
        max_samples: Optional limit on number of samples (for quick testing).
        **model_kwargs: Additional model-specific arguments.

    Returns:
        dict with training results.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load or create data
    if use_synthetic:
        log.info("Using synthetic data for testing")
        from dataset import create_synthetic_data

        train_texts, train_labels = create_synthetic_data(
            num_samples=1000,
            num_classes=num_classes,
            vocab_size=1000,
            max_length=max_length,
        )
        test_texts, test_labels = create_synthetic_data(
            num_samples=200,
            num_classes=num_classes,
            vocab_size=1000,
            max_length=max_length,
        )
        class_names = [f"class_{i}" for i in range(num_classes)]
        use_huggingface = False  # Use simple tokenization for synthetic data
    elif use_huggingface:
        # Use HuggingFace datasets library to load IMDB
        train_texts, train_labels, test_texts, test_labels, class_names = load_imdb_huggingface(
            max_samples=max_samples
        )
    else:
        # Try to load IMDB data from local directory
        from dataset import SENTIMENT_CLASSES, load_imdb_data

        data_path = Path(data_dir)
        if (data_path / "train").exists():
            log.info(f"Loading IMDB data from {data_dir}")
            train_texts, train_labels, test_texts, test_labels = load_imdb_data(data_dir)
            class_names = SENTIMENT_CLASSES
        else:
            log.warning(f"No data found in {data_dir}, using synthetic data")
            from dataset import create_synthetic_data

            train_texts, train_labels = create_synthetic_data(
                num_samples=1000, num_classes=num_classes
            )
            test_texts, test_labels = create_synthetic_data(
                num_samples=200, num_classes=num_classes
            )
            class_names = [f"class_{i}" for i in range(num_classes)]

    log.info(f"Training samples: {len(train_texts)}")
    log.info(f"Test samples: {len(test_texts)}")

    # Build vocabulary or tokenizer
    if use_huggingface and not use_synthetic:
        # Use HuggingFace tokenizer
        log.info(f"Using HuggingFace tokenizer: {tokenizer_name}")
        tokenizer, actual_vocab_size = create_huggingface_tokenizer(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )
        vocab_size = actual_vocab_size

        # Create datasets with HuggingFace tokenizer
        train_dataset = HuggingFaceTextDataset(train_texts, train_labels, tokenizer, max_length)
        test_dataset = HuggingFaceTextDataset(test_texts, test_labels, tokenizer, max_length)

        # Collate function for HuggingFace tokenized data
        def collate_fn(batch):
            sequences, labels = zip(*batch)
            sequences = torch.stack(sequences)
            labels = torch.tensor(labels, dtype=torch.long)
            return sequences, labels

        # Save tokenizer info
        tokenizer.save_pretrained(str(output_path / "tokenizer"))
    else:
        # Use custom vocabulary
        from dataset import TextClassificationDataset, Vocabulary
        from dataset import collate_fn as dataset_collate_fn

        vocab = Vocabulary(max_size=vocab_size)
        vocab.build(train_texts)
        vocab_size = len(vocab)
        log.info(f"Vocabulary size: {vocab_size}")

        train_dataset = TextClassificationDataset(train_texts, train_labels, vocab, max_length)
        test_dataset = TextClassificationDataset(test_texts, test_labels, vocab, max_length)

        pad_idx = vocab.pad_idx

        def collate_fn(batch):
            return dataset_collate_fn(batch, pad_idx)

        # Save vocabulary
        vocab.save(str(output_path / "vocab.json"))

    log.info(f"Vocabulary size: {vocab_size}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    from model import create_model

    model = create_model(
        model_type=model_type,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        dropout=dropout,
        **model_kwargs,
    )
    model = model.to(device)
    log.info(f"Model type: {model_type}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    # Training loop
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log.info(f"Epoch {epoch + 1}/{epochs}")
        log.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        log.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_path / "best_model.pt")
            log.info(f"  Saved new best model with accuracy: {best_acc:.4f}")

    # Save final model
    torch.save(model.state_dict(), output_path / "final_model.pt")

    # Save class names
    with open(output_path / "classes.txt", "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    results = {
        "best_accuracy": best_acc,
        "final_accuracy": val_acc,
        "epochs_trained": epochs,
        "class_names": class_names,
        "model_path": str(output_path / "best_model.pt"),
        "vocab_size": vocab_size,
        "history": history,
    }

    log.info(f"Training complete! Best accuracy: {best_acc:.4f}")
    return results


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> dict:
    """Main entry point with Hydra configuration.

    Supports modular config structure with groups:
    - model: textcnn, lstm
    - training: default, fast, long
    - data: imdb, synthetic

    Run with experiment configs:
        python train.py +experiment=baseline
        python train.py +experiment=quick_test
        python train.py +experiment=lstm_baseline

    Override individual settings:
        python train.py model=lstm training.epochs=20
        python train.py training.learning_rate=0.01

    HuggingFace mode (default):
        python train.py  # Uses HuggingFace datasets and tokenizer

    Local data mode:
        python train.py data.use_huggingface=false  # Uses local IMDB data
    """
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Extract model-specific config
    model_kwargs = {}
    if cfg.model.name == "textcnn":
        model_kwargs["num_filters"] = cfg.model.get("num_filters", 100)
        model_kwargs["kernel_sizes"] = list(cfg.model.get("kernel_sizes", [3, 4, 5]))
    elif cfg.model.name == "lstm":
        model_kwargs["hidden_dim"] = cfg.model.get("hidden_dim", 256)
        model_kwargs["num_layers"] = cfg.model.get("num_layers", 2)
        model_kwargs["bidirectional"] = cfg.model.get("bidirectional", True)

    results = train(
        data_dir=cfg.data.data_dir,
        output_dir=cfg.paths.output_dir,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        model_type=cfg.model.name,
        vocab_size=cfg.data.vocab_size,
        embedding_dim=cfg.model.embedding_dim,
        num_classes=cfg.model.num_classes,
        max_length=cfg.data.max_length,
        dropout=cfg.model.dropout,
        num_workers=cfg.data.num_workers,
        seed=cfg.get("seed", 42),
        use_synthetic=cfg.data.get("use_synthetic", False),
        use_huggingface=cfg.data.get("use_huggingface", True),
        tokenizer_name=cfg.data.get("tokenizer_name", "bert-base-uncased"),
        max_samples=cfg.data.get("max_samples", None),
        **model_kwargs,
    )

    return results


if __name__ == "__main__":
    main()
