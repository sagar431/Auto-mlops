import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from torch_lr_finder import LRFinder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-lr", type=float, default=1e-6)
    parser.add_argument("--end-lr", type=float, default=1)
    parser.add_argument("--num-iter", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default="lr_finder.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(args.train_dir, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=len(train_ds.classes))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr)
    criterion = nn.CrossEntropyLoss()

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=args.end_lr, num_iter=args.num_iter)
    lr_finder.plot(args.output)
    lr_finder.reset()
    print(f"LR finder plot saved to {args.output}")


if __name__ == "__main__":
    main()
