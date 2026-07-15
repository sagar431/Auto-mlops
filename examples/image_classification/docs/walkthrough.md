# Golden Slice Walkthrough

This walkthrough covers the one verified local image-classification path. Run every command from the repository root.

## 1. Synchronize the Locked Environment

```bash
uv sync --extra dev --locked
```

No `.env` file or API key is required.

## 2. Train a Real Bounded Model

```bash
uv run python -m examples.image_classification.project.golden_train
```

The trainer creates deterministic red/blue tensors in memory. It never downloads a dataset. Its fixed defaults are CPU, seed 17, three epochs, 64 training samples, 16 validation samples, and batches of eight. The structured result names:

- `project/artifacts/golden/model.pt`
- `project/artifacts/golden/training_config.json`
- `project/artifacts/golden/metrics.json`
- `project/artifacts/golden/sample-red.png`

The entire artifact directory is ignored by Git.

## 3. Verify Training, Loading, Inference, and FastAPI

```bash
uv run pytest examples/image_classification/tests/test_golden_training.py examples/image_classification/tests/test_inference.py examples/image_classification/tests/test_serve.py -q
```

These tests verify deterministic weights, checkpoint metadata and failure modes, real red/blue inference, probability normalization, model startup, health, prediction, content-type validation, malformed images, upload limits, and the not-loaded state.

The combined local command is:

```bash
uv run python examples/image_classification/verify_golden.py
```

## 4. Run the API Directly

```bash
GOLDEN_MODEL_PATH=examples/image_classification/project/artifacts/golden/model.pt uv run uvicorn examples.image_classification.project.serve:app --host 127.0.0.1 --port 8000
```

Probe it from another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail --form "file=@examples/image_classification/project/artifacts/golden/sample-red.png;type=image/png" http://127.0.0.1:8000/predict
```

The application loads the model once during startup. `/health` is successful only when that load completed. `/predict` performs the preprocessing recorded in the checkpoint and executes the loaded model.

## 5. Run the Real Docker Verification

```bash
docker info || sudo -n docker info
uv run python examples/image_classification/verify_golden.py --docker
```

Docker is opt-in because it builds an image and starts a local container. The verifier uses bounded timeouts and a unique name, mounts the model read-only, records the actual responses and recent logs, and stops and removes its own container in `finally` cleanup.

## 6. Interpret the Boundary

Passing this walkthrough proves a local synthetic CPU train-to-serve path. It does not prove external dataset handling, DVC/S3 lineage, MLflow/HPO, GPU use, Kubernetes/KServe, Helm, ArgoCD, Lambda, Hugging Face Spaces, registry push, self-healing, or complete capstone readiness. Those remain future milestones.
