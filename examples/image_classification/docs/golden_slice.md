# Golden Image-Classification Slice

## Baseline Audit

Before this contract was introduced, the example had no reproducible train-to-serve path:

- Training produced ten CIFAR-10 logits and wrote ten class names, while inference silently defaulted to `cat` and `dog`.
- Training saved a raw `CIFAR10CNN` state dictionary, while inference constructed a ResNet18 before loading it.
- Training used 32×32 images and CIFAR-10 normalization; inference defaulted to 224×224 and ImageNet normalization.
- The raw checkpoint did not identify its schema, architecture, classes, input shape, preprocessing, configuration, or metrics.
- The documented training path downloaded CIFAR-10. The setup script produced local files but did not establish a checkpoint-to-service contract.
- Tests modified `sys.path` to support top-level imports, while package execution required different imports.
- PyTorch, TorchVision, and Pillow were listed only in the example requirements and were absent from the repository's locked development environment.
- The available Docker template expected the whole repository as build context and targeted training, while deployment templates targeted future remote platforms.

## Canonical Contract

| Field | Contract |
| --- | --- |
| Supported task | Deterministic two-class RGB image classification using locally generated PNG files |
| Class names | `red`, `blue`, in that order |
| Input format | One PNG or JPEG upload; decoded as RGB |
| Input dimensions | Any positive source dimensions; resized to 16×16 |
| Preprocessing | Scale bytes to `[0, 1]`, then normalize each RGB channel with mean `0.5` and standard deviation `0.5` |
| Architecture | `tiny_color_cnn_v1`: 3→8 convolution, ReLU, adaptive average pooling, and a two-logit linear classifier |
| Device | CPU only |
| Checkpoint schema | `golden-image-classifier.v1` |
| Dataset schema | `golden-red-blue-dataset.v1`, with a SHA-256 manifest for every image |
| DVC stages | `prepare_golden_data` → `train_golden` |
| Checkpoint file | `project/artifacts/dvc-golden/model.pt` (generated, DVC-cached, and ignored by Git) |
| Other artifacts | `training_config.json`, `metrics.json`, `lineage.json`, and `sample-red.png` under the same ignored directory |
| API port | 8000 in the container |
| Upload limit | 1,000,000 bytes |

The checkpoint is a PyTorch dictionary containing `schema_version`, `architecture`, `state_dict`, `class_names`, `num_classes`, `image_size`, `normalization`, `training_config`, `metrics`, and `dataset_lineage`. File-backed lineage records the dataset ID, aggregate dataset checksum, manifest checksum, split sample counts, and every relative image path with its SHA-256 checksum. Loading is strict: missing files, unreadable files, unknown schemas, incompatible architectures, invalid class, preprocessing, or lineage metadata, and incompatible state dictionaries fail explicitly.

## Exact Commands

Run from the repository root after `uv sync --extra dev --locked`.

Canonical file-backed DVC reproduction:

```bash
cd examples/image_classification/project
uv run dvc repro
uv run dvc status
uv run dvc metrics show
cd ../../..
```

This creates 64 training and 16 validation PNG files under ignored `data/golden/`, verifies their checksums before training, and writes the ignored checkpoint under `artifacts/dvc-golden/`. `dvc.lock` records the exact stage dependencies, parameters, and output hashes. Repeating `uv run dvc repro` reports that the pipeline is up to date; forced reproductions produce byte-identical tracked outputs.

The original bounded in-memory command remains available for the fastest serving smoke test:

```bash
uv run python -m examples.image_classification.project.golden_train
```

Focused training, checkpoint, inference, and FastAPI tests:

```bash
uv run pytest examples/image_classification/tests/test_golden_dvc_lineage.py examples/image_classification/tests/test_golden_training.py examples/image_classification/tests/test_inference.py examples/image_classification/tests/test_serve.py -q
```

Single local verifier (training plus focused tests):

```bash
uv run python examples/image_classification/verify_golden.py
```

Start the API directly after training:

```bash
GOLDEN_MODEL_PATH=examples/image_classification/project/artifacts/dvc-golden/model.pt uv run uvicorn examples.image_classification.project.serve:app --host 127.0.0.1 --port 8000
```

`GET /health` returns HTTP 200 only after a checkpoint loads:

```json
{
  "status": "healthy",
  "schema_version": "golden-image-classifier.v1",
  "architecture": "tiny_color_cnn_v1",
  "class_names": ["red", "blue"]
}
```

`POST /predict` uses multipart form data with one field named `file` and a PNG or JPEG value. A successful response has this shape:

```json
{
  "predicted_class": "red",
  "confidence": 0.99,
  "probabilities": {"red": 0.99, "blue": 0.01}
}
```

The exact probability values depend on the trained checkpoint; they must be finite, between zero and one, and sum approximately to one.

## Docker Contract

The model is not copied into the image. Train it first, then mount the generated artifact directory read-only. On the verified host the Docker socket requires non-interactive `sudo`; omit `sudo -n` on hosts where the current user has direct Docker access:

```bash
sudo -n docker build --file examples/image_classification/project/Dockerfile.golden --tag auto-mlops-golden-image:local examples/image_classification/project
sudo -n docker run --detach --name auto-mlops-golden-manual --publish 127.0.0.1:8000:8000 --volume "$PWD/examples/image_classification/project/artifacts/golden:/models:ro" auto-mlops-golden-image:local
curl --fail http://127.0.0.1:8000/health
curl --fail --form "file=@examples/image_classification/project/artifacts/golden/sample-red.png;type=image/png" http://127.0.0.1:8000/predict
sudo -n docker rm --force auto-mlops-golden-manual
```

The bounded verifier owns a unique container name, waits at most 30 seconds for readiness, verifies both endpoints, captures recent logs, and removes only that container:

```bash
uv run python examples/image_classification/verify_golden.py --docker
```

## Expected Verification

- `dvc repro` prepares the declared PNG files, trains from them, and exits zero without network access.
- The dataset manifest, checkpoint, lineage JSON, metrics, and `dvc.lock` have deterministic content hashes.
- The checkpoint and JSON metadata files exist only under the ignored artifact directory.
- Validation accuracy is observed from real model execution.
- Focused tests pass without network, Docker, GPU, an external dataset, or an LLM key.
- Docker verification, when explicitly requested, records a successful build, observed health and prediction payloads, logs, and container removal.
- Repository registry/runtime tests, E2E tests, Ruff, and `git diff --check` remain green.

## Explicit Non-Goals

This slice does not provide CIFAR-10 or any external dataset download, a DVC remote or S3, MLflow or HPO, GPU/CUDA, Kubernetes or KServe, Helm, ArgoCD or GitOps, Lambda, Hugging Face Spaces, a container registry or image push, self-healing, production authentication, generalized model architecture discovery, or complete capstone readiness.
