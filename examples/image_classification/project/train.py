"""Training script for image classification model."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import create_data_loaders, create_synthetic_data
from model import create_model


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
    num_classes: int = 2,
    dropout: float = 0.5,
    image_size: int = 224,
    use_synthetic: bool = False,
) -> dict:
    """Main training function.

    Returns:
        dict with training results including final accuracy and metrics
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create synthetic data if needed
    if use_synthetic or not Path(data_dir).exists():
        print("Creating synthetic data for demonstration...")
        create_synthetic_data(data_dir, num_samples=200, num_classes=num_classes)

    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
    )
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    model = create_model(num_classes=len(class_names), dropout=dropout)
    model = model.to(device)

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

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_path / "best_model.pt")
            print(f"  Saved new best model with accuracy: {best_acc:.4f}")

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

    print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Path to save models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
        dropout=args.dropout,
        image_size=args.image_size,
        use_synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
