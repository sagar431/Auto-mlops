import argparse
from pathlib import Path

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


def objective(trial, args, train_ds, val_ds):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=True, num_classes=len(train_ds.classes))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default="hpo_results.txt")
    args = parser.parse_args()

    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(args.val_dir, transform=val_tf)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, args, train_ds, val_ds), n_trials=args.trials)

    best = study.best_params
    Path(args.output).write_text(str(best))
    print("Best params:", best)


if __name__ == "__main__":
    main()
