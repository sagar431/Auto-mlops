import hashlib
import io
import os
from pathlib import Path

import redis
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

APP_NAME = "model-api"

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.ts")
CLASSES_PATH = os.environ.get("CLASSES_PATH", "models/classes.txt")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))

app = FastAPI(title=APP_NAME)

redis_client = redis.from_url(REDIS_URL)

model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

if Path(CLASSES_PATH).exists():
    classes = Path(CLASSES_PATH).read_text().splitlines()
else:
    classes = []

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def cache_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    key = cache_key(raw)
    cached = redis_client.get(key)
    if cached:
        return {"cached": True, "prediction": cached.decode("utf-8")}

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        idx = int(torch.argmax(logits, dim=1)[0])

    label = classes[idx] if classes and idx < len(classes) else str(idx)
    redis_client.setex(key, 3600, label)
    return {"cached": False, "prediction": label}
