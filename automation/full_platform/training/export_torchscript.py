import argparse
from pathlib import Path

import timm
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", default="models/model.ts")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    example = torch.randn(1, 3, args.image_size, args.image_size)
    traced = torch.jit.trace(model, example)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    print(f"Saved TorchScript to {output}")


if __name__ == "__main__":
    main()
