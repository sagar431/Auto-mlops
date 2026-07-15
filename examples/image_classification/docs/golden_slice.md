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
| Supported task | Deterministic two-class RGB image classification using synthetic tensors |
| Class names | `red`, `blue`, in that order |
| Input format | One PNG or JPEG upload; decoded as RGB |
| Input dimensions | Any positive source dimensions; resized to 16×16 |
| Preprocessing | Scale bytes to `[0, 1]`, then normalize each RGB channel with mean `0.5` and standard deviation `0.5` |
| Architecture | `tiny_color_cnn_v1`: 3→8 convolution, ReLU, adaptive average pooling, and a two-logit linear classifier |
| Device | CPU only |
| Checkpoint schema | `golden-image-classifier.v1` |
| Checkpoint file | `project/artifacts/golden/model.pt` (generated and ignored) |
| Other artifacts | `training_config.json`, `metrics.json`, and optional `sample-red.png` under the same ignored directory |
| API port | 8000 in the container |
| Upload limit | 1,000,000 bytes |

The checkpoint is a PyTorch dictionary containing `schema_version`, `architecture`, `state_dict`, `class_names`, `num_classes`, `image_size`, `normalization`, `training_config`, and `metrics`. Loading is strict: missing files, unreadable files, unknown schemas, incompatible architectures, invalid class or preprocessing metadata, and incompatible state dictionaries fail explicitly.

## Exact Commands

Run from the repository root after `uv sync --extra dev --locked`.

Bounded training:

```bash
uv run python -m examples.image_classification.project.golden_train
```

Focused training, checkpoint, inference, and FastAPI tests:

```bash
uv run pytest examples/image_classification/tests/test_golden_training.py examples/image_classification/tests/test_inference.py examples/image_classification/tests/test_serve.py -q
```

Single local verifier (training plus focused tests):

```bash
uv run python examples/image_classification/verify_golden.py
```

Start the API directly after training:

```bash
GOLDEN_MODEL_PATH=examples/image_classification/project/artifacts/golden/model.pt uv run uvicorn examples.image_classification.project.serve:app --host 127.0.0.1 --port 8000
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

- Training exits zero and prints one structured JSON result.
- The checkpoint and JSON metadata files exist only under the ignored artifact directory.
- Validation accuracy is observed from real model execution.
- Focused tests pass without network, Docker, GPU, an external dataset, or an LLM key.
- Docker verification, when explicitly requested, records a successful build, observed health and prediction payloads, logs, and container removal.
- Repository registry/runtime tests, E2E tests, Ruff, and `git diff --check` remain green.

## Explicit Non-Goals

This slice does not provide CIFAR-10 or any external dataset download, DVC or S3, MLflow or HPO, GPU/CUDA, Kubernetes or KServe, Helm, ArgoCD or GitOps, Lambda, Hugging Face Spaces, a container registry or image push, self-healing, production authentication, generalized model architecture discovery, or complete capstone readiness.
