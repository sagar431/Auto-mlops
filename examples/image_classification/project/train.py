"""Training script for image classification model using PyTorch, CIFAR-10, and Hydra."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets, transforms

log = logging.getLogger(__name__)


def get_cifar10_transforms(image_size: int = 32) -> tuple:
    """Get CIFAR-10 transforms for training and validation."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )
    return train_transform, val_transform


def create_cifar10_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 32,
    num_workers: int = 4,
) -> tuple:
    """Create CIFAR-10 train and validation data loaders.

    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    train_transform, val_transform = get_cifar10_transforms(image_size)

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    return train_loader, val_loader, class_names


class CIFAR10CNN(nn.Module):
    """CNN architecture optimized for CIFAR-10 (32x32 images).

    Architecture:
    - 3 convolutional layers with batch normalization
    - Max pooling after each conv layer
    - 2 fully connected layers with dropout
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        # After 3 pooling layers on 32x32: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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
) -> tuple:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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
    num_classes: int = 10,
    dropout: float = 0.5,
    image_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> dict:
    """Main training function.

    Returns:
        dict with training results including final accuracy and metrics
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

    # Create data loaders
    train_loader, val_loader, class_names = create_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )
    log.info(f"Classes: {class_names}")
    log.info(f"Training samples: {len(train_loader.dataset)}")
    log.info(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    model = CIFAR10CNN(num_classes=num_classes, dropout=dropout)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    # Training loop
    # Always persist the first completed epoch, even when its measured accuracy is zero.
    best_acc = -1.0
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
        "history": history,
    }

    log.info(f"Training complete! Best accuracy: {best_acc:.4f}")
    return results


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> dict:
    """Main entry point with Hydra configuration.

    Supports modular config structure with groups:
    - model: cifar10_cnn, resnet18
    - training: default, fast, long, sgd
    - data: cifar10, cifar10_minimal
    - paths: default

    Run with experiment configs:
        python train.py +experiment=baseline
        python train.py +experiment=quick_test
        python train.py +experiment=high_accuracy
        python train.py +experiment=resnet_baseline

    Override individual settings:
        python train.py model=resnet18 training=sgd
        python train.py training.epochs=20 training.learning_rate=0.01
    """
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Extract model config - handle both old flat structure and new modular structure
    num_classes = cfg.model.get("num_classes", 10)
    dropout = cfg.model.get("dropout", 0.5)

    results = train(
        data_dir=cfg.data.data_dir,
        output_dir=cfg.paths.output_dir,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        num_classes=num_classes,
        dropout=dropout,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.get("seed", 42),
    )

    return results


if __name__ == "__main__":
    main()
