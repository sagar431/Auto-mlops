"""Training script for tabular regression with Hydra configuration."""

import json
import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from dataset import create_dataloaders, save_synthetic_data
from model import create_model
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple of (average loss, RMSE)
    """
    model.train()
    total_loss = 0.0
    total_se = 0.0
    n_samples = 0

    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_se += ((predictions - targets) ** 2).sum().item()
        n_samples += len(targets)

    avg_loss = total_loss / n_samples
    rmse = (total_se / n_samples) ** 0.5

    return avg_loss, rmse


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple of (average loss, RMSE)
    """
    model.eval()
    total_loss = 0.0
    total_se = 0.0
    n_samples = 0

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)
            loss = criterion(predictions, targets)

            total_loss += loss.item() * len(targets)
            total_se += ((predictions - targets) ** 2).sum().item()
            n_samples += len(targets)

    avg_loss = total_loss / n_samples
    rmse = (total_se / n_samples) ** 0.5

    return avg_loss, rmse


def train(
    cfg: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
) -> dict:
    """Main training function.

    Args:
        cfg: Hydra configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input feature dimension

    Returns:
        Training results dictionary
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # Create model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_type = model_cfg.pop("name")
    model_cfg["input_dim"] = input_dim

    model = create_model(model_type, **model_cfg)
    model.to(device)
    log.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} params")

    # Setup optimizer
    optimizer_name = cfg.training.get("optimizer", "adam").lower()
    lr = cfg.training.learning_rate
    weight_decay = cfg.training.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = cfg.training.get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Setup scheduler
    scheduler = None
    if hasattr(cfg.training, "scheduler"):
        sched_cfg = cfg.training.scheduler
        if sched_cfg.name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 5),
            )
        elif sched_cfg.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.epochs
            )

    # Loss function
    criterion = nn.MSELoss()

    # Early stopping
    early_stopping_cfg = cfg.training.get("early_stopping", {})
    patience = early_stopping_cfg.get("patience", 10)
    min_delta = early_stopping_cfg.get("min_delta", 0.0001)
    best_val_loss = float("inf")
    patience_counter = 0

    # Output paths
    output_dir = Path(cfg.paths.output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    history = {"train_loss": [], "train_rmse": [], "val_loss": [], "val_rmse": []}

    for epoch in range(cfg.training.epochs):
        # Train
        train_loss, train_rmse = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_rmse = validate(model, val_loader, criterion, device)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_rmse"].append(train_rmse)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log progress
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch + 1}/{cfg.training.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, LR: {current_lr:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {"model_type": model_type, **model_cfg},
                "epoch": epoch,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
            }
            torch.save(checkpoint, models_dir / "best_model.pt")
            log.info(f"Saved best model with val_loss={val_loss:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            log.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Save final model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"model_type": model_type, **model_cfg},
        "epoch": epoch,
        "val_loss": val_loss,
        "val_rmse": val_rmse,
    }
    torch.save(checkpoint, models_dir / "final_model.pt")

    # Save model config
    config_dict = {"model_type": model_type, **model_cfg}
    with open(models_dir / "model_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Results
    results = {
        "best_val_loss": best_val_loss,
        "best_val_rmse": min(history["val_rmse"]),
        "final_val_loss": val_loss,
        "final_val_rmse": val_rmse,
        "epochs_trained": epoch + 1,
        "model_type": model_type,
    }

    return results


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> dict | None:
    """Main entry point with Hydra configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Training results
    """
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Set seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # Get data directory
    data_dir = Path(cfg.data.data_dir)

    # Use synthetic data if specified
    if cfg.data.get("use_synthetic", False):
        log.info("Using synthetic data for testing")
        save_synthetic_data(
            data_dir,
            n_train=cfg.data.get("n_train", 800),
            n_test=cfg.data.get("n_test", 200),
            n_features=cfg.data.get("n_features", 8),
            seed=seed,
        )

    # Create dataloaders
    train_loader, val_loader, data_info = create_dataloaders(
        data_dir=str(data_dir),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.get("num_workers", 0),
    )

    log.info(f"Loaded data: {data_info['train_samples']} train, {data_info['test_samples']} test")
    log.info(f"Input dimension: {data_info['input_dim']}")

    # Train
    results = train(cfg, train_loader, val_loader, data_info["input_dim"])

    log.info(f"Training complete. Best validation RMSE: {results['best_val_rmse']:.4f}")

    return results


if __name__ == "__main__":
    main()
