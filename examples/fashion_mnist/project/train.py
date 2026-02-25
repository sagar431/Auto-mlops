"""Training script for Fashion MNIST classifier using PyTorch."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionCNN(nn.Module):
    """Lightweight CNN for 28x28 grayscale Fashion MNIST images."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SyntheticFashionMNIST(torch.utils.data.Dataset):
    """Synthetic dataset mimicking Fashion MNIST when download isn't available."""

    def __init__(self, num_samples: int = 10000, num_classes: int = 10, seed: int = 42):
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(num_samples, 1, 28, 28, generator=gen) * 0.3 + 0.3
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=gen)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_loaders(data_dir: str = "data", batch_size: int = 64, num_workers: int = 0):
    """Create Fashion MNIST train/val data loaders.
    Falls back to synthetic data if download fails (e.g., no network).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    try:
        train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        val_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        log.info("Using real Fashion MNIST dataset")
    except Exception as e:
        log.warning(f"Could not download Fashion MNIST ({e}), using synthetic data")
        train_ds = SyntheticFashionMNIST(num_samples=10000, seed=42)
        val_ds = SyntheticFashionMNIST(num_samples=2000, seed=99)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader


def train(
    data_dir: str = "data",
    output_dir: str = "models",
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    dropout: float = 0.3,
    seed: int = 42,
) -> dict:
    """Train Fashion MNIST classifier. Returns dict with metrics."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_loaders(data_dir, batch_size)

    model = FashionCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                val_loss += criterion(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        log.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train: loss={train_loss/len(train_loader):.4f} acc={train_acc:.4f} - "
            f"Val: loss={val_loss/len(val_loader):.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")

    torch.save(model.state_dict(), f"{output_dir}/final_model.pt")
    with open(f"{output_dir}/classes.txt", "w") as f:
        f.write("\n".join(CLASS_NAMES))

    log.info(f"Done! Best accuracy: {best_acc:.4f}")
    return {
        "best_accuracy": best_acc,
        "final_accuracy": val_acc,
        "epochs_trained": epochs,
        "model_path": f"{output_dir}/best_model.pt",
        "class_names": CLASS_NAMES,
    }


if __name__ == "__main__":
    results = train()
    print(f"\nResults: {results}")
