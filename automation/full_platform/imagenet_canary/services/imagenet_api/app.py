import hashlib
import io
import os
import time

import redis
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from transformers import AutoImageProcessor, AutoModelForImageClassification

APP_NAME = "imagenet-api"

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "google/vit-base-patch16-224")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
CACHE_TTL = int(os.environ.get("CACHE_TTL", "3600"))

app = FastAPI(title=APP_NAME)
redis_client = redis.from_url(REDIS_URL)

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
)

processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
model.eval()


def cache_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@app.get("/health")
def health():
    return {"status": "ok", "model": HF_MODEL_ID}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    raw = await file.read()
    key = cache_key(raw)
    cached = redis_client.get(key)
    if cached:
        REQUEST_COUNT.labels(status="cached").inc()
        REQUEST_LATENCY.observe(time.perf_counter() - start)
        return {"cached": True, "prediction": cached.decode("utf-8")}

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        idx = int(torch.argmax(logits, dim=1)[0])

    label = model.config.id2label.get(idx, str(idx))
    redis_client.setex(key, CACHE_TTL, label)
    REQUEST_COUNT.labels(status="success").inc()
    REQUEST_LATENCY.observe(time.perf_counter() - start)
    return {"cached": False, "prediction": label}
