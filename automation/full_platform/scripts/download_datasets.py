import os
import zipfile
from pathlib import Path

import yaml
from kaggle.api.kaggle_api_extended import KaggleApi


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "datasets.yaml"
    return yaml.safe_load(config_path.read_text())


def download_dataset(api: KaggleApi, slug: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {slug} to {output_dir}")
    api.dataset_download_files(slug, path=str(output_dir), unzip=True)


def main():
    api = KaggleApi()
    api.authenticate()

    cfg = load_config()
    for dataset_name, ds in cfg.get("datasets", {}).items():
        raw_dir = Path(ds["raw_dir"])
        slug = ds["kaggle_slug"]
        download_dataset(api, slug, raw_dir)
        # Unzip nested zip files if any
        for z in raw_dir.rglob("*.zip"):
            try:
                with zipfile.ZipFile(z, "r") as zip_ref:
                    zip_ref.extractall(z.parent)
            except Exception:
                pass
    print("Download complete")


if __name__ == "__main__":
    if not os.environ.get("KAGGLE_USERNAME"):
        raise SystemExit("Set KAGGLE_USERNAME and KAGGLE_KEY before running")
    main()
