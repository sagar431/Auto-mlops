import random
import shutil
from pathlib import Path

import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "datasets.yaml"
    return yaml.safe_load(config_path.read_text())


def split_class_images(class_dir: Path, out_train: Path, out_val: Path, out_test: Path, split):
    images = [p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(images)
    n = len(images)
    n_train = int(n * split["train"])
    n_val = int(n * split["val"])
    train_imgs = images[:n_train]
    val_imgs = images[n_train : n_train + n_val]
    test_imgs = images[n_train + n_val :]

    for img in train_imgs:
        shutil.copy2(img, out_train / img.name)
    for img in val_imgs:
        shutil.copy2(img, out_val / img.name)
    for img in test_imgs:
        shutil.copy2(img, out_test / img.name)


def prepare_dataset(ds):
    raw_dir = Path(ds["raw_dir"])
    train_dir = Path(ds["train_dir"])
    val_dir = Path(ds["val_dir"])
    test_dir = Path(ds["test_dir"])
    split = ds["split"]

    if train_dir.exists() and test_dir.exists():
        print(f"Split already exists: {train_dir}")
        return

    # Detect class folders directly under raw_dir
    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"No class directories found in {raw_dir}")
        return

    for class_dir in class_dirs:
        out_train = train_dir / class_dir.name
        out_val = val_dir / class_dir.name
        out_test = test_dir / class_dir.name
        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)
        out_test.mkdir(parents=True, exist_ok=True)
        split_class_images(class_dir, out_train, out_val, out_test, split)

    print(f"Split created for {raw_dir}")


def main():
    cfg = load_config()
    for _, ds in cfg.get("datasets", {}).items():
        prepare_dataset(ds)


if __name__ == "__main__":
    random.seed(42)
    main()
