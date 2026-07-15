# Image Classification Example

The verified path in this example is a small local CPU slice:

```text
deterministic synthetic tensors
→ real PyTorch checkpoint
→ FastAPI
→ local CPU Docker image
→ observed /health and /predict responses
```

It requires no external dataset, GPU, cloud service, registry, LLM key, DVC, or MLflow server. The canonical technical contract and the pre-change inconsistency audit are in [docs/golden_slice.md](docs/golden_slice.md).

## Install

From the repository root:

```bash
uv sync --extra dev --locked
```

PyTorch and TorchVision use the official CPU wheel index recorded in `pyproject.toml` and `uv.lock`.

## Fast Local API Verification

Generate the bounded artifact:

```bash
uv run python -m examples.image_classification.project.golden_train
```

The command uses seed 17, three epochs, 64 training samples, 16 validation samples, and CPU only. It writes ignored runtime artifacts under `examples/image_classification/project/artifacts/golden/` and prints structured JSON.

Run the focused contract tests:

```bash
uv run pytest examples/image_classification/tests/test_golden_training.py examples/image_classification/tests/test_inference.py examples/image_classification/tests/test_serve.py -q
```

Or run training and those tests with one safe command:

```bash
uv run python examples/image_classification/verify_golden.py
```

To inspect the API manually:

```bash
GOLDEN_MODEL_PATH=examples/image_classification/project/artifacts/golden/model.pt uv run uvicorn examples.image_classification.project.serve:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail --form "file=@examples/image_classification/project/artifacts/golden/sample-red.png;type=image/png" http://127.0.0.1:8000/predict
```

## Real Docker Verification

Check Docker and run the opt-in bounded verifier:

```bash
docker info || sudo -n docker info
uv run python examples/image_classification/verify_golden.py --docker
```

The verifier uses direct Docker access when available and otherwise tries bounded,
non-interactive `sudo -n`; it fails clearly if neither can reach the daemon.

It builds `auto-mlops-golden-image:local` from the narrow `project/` context, mounts the checkpoint directory read-only, chooses a local port, waits up to 30 seconds, exercises both endpoints, captures logs, and removes only its uniquely named container.

The equivalent manual commands are documented in `docs/golden_slice.md`. Do not omit the final container removal command when using the manual path.

## Layout

```text
examples/image_classification/
├── README.md
├── verify_golden.py
├── docs/
│   ├── golden_slice.md
│   └── walkthrough.md
├── project/
│   ├── golden_train.py
│   ├── model.py
│   ├── inference.py
│   ├── serve.py
│   ├── Dockerfile.golden
│   ├── .dockerignore
│   ├── requirements.golden.txt
│   ├── train.py
│   ├── evaluate.py
│   ├── prepare_data.py
│   ├── dataset.py
│   ├── requirements.txt
│   ├── dvc.yaml
│   └── configs/
└── tests/
    ├── test_golden_training.py
    ├── test_inference.py
    ├── test_serve.py
    ├── test_training.py
    ├── test_model.py
    ├── test_dataset.py
    ├── test_hydra_configs.py
    └── test_dvc_pipeline.py
```

## Historical Example Surface

The repository still contains the earlier CIFAR-10, Hydra, and DVC teaching surface so existing configuration and structural tests remain meaningful. It includes the `CIFAR10CNN` and `ResNet18` model options plus the `baseline`, `quick_test`, `high_accuracy`, and `resnet_baseline` experiment configurations.

That historical surface is not the verified golden path. Its CIFAR-10 commands may download an external dataset and its broader agent/deployment prose represented future ambitions. Do not use it as evidence that DVC/S3, MLflow/HPO, KServe, Lambda, Spaces, or a complete capstone is ready.

## Future Capstone Work

Later milestones may connect this verified serving foundation to dataset lineage, experiment tracking, tuning, production deployment, and operations. Those capabilities are explicitly deferred; this milestone proves only local bounded training, strict artifact loading, FastAPI inference, and local Docker execution.
