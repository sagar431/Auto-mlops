"""FastAPI service for the verified golden image-classification checkpoint."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

try:
    from .golden_train import DEFAULT_ARTIFACT_DIR
    from .inference import ImageClassifier, InvalidImageError
except ImportError:  # Support direct execution from the project directory.
    from golden_train import DEFAULT_ARTIFACT_DIR
    from inference import ImageClassifier, InvalidImageError

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}
MAX_UPLOAD_BYTES = 1_000_000
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "model.pt"


def _decode_upload(payload: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise InvalidImageError("Only PNG and JPEG images are supported")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Uploaded file is not a valid PNG or JPEG image") from exc


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create an application that loads exactly one checkpoint during startup."""
    configured_path = Path(
        model_path or os.environ.get("GOLDEN_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = None
        app.state.model_load_error = None
        try:
            app.state.classifier = ImageClassifier(configured_path)
        except Exception:
            app.state.model_load_error = "model checkpoint could not be loaded"
        yield
        app.state.classifier = None

    app = FastAPI(title="Golden Image Classification API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        classifier = app.state.classifier
        if classifier is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "error": app.state.model_load_error},
            )
        return {
            "status": "healthy",
            "schema_version": classifier.schema_version,
            "architecture": classifier.architecture,
            "class_names": classifier.classes,
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        classifier = app.state.classifier
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Only PNG and JPEG uploads are supported")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded image exceeds the size limit")
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")
        try:
            image = _decode_upload(payload)
            return classifier.predict(image)
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return app


app = create_app()
