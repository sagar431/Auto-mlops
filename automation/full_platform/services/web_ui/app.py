import base64
import os
from typing import Optional

import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_NAME = "web-ui"
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://model-api:8000/predict")
MODEL_API_HEALTH_URL = os.environ.get("MODEL_API_HEALTH_URL", "http://model-api:8000/health")
REQUEST_TIMEOUT = float(os.environ.get("MODEL_API_TIMEOUT", "10"))

app = FastAPI(title=APP_NAME)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def check_api_health() -> Optional[str]:
    try:
        resp = requests.get(MODEL_API_HEALTH_URL, timeout=REQUEST_TIMEOUT)
        if resp.ok:
            return "healthy"
        return f"error ({resp.status_code})"
    except requests.RequestException:
        return "unreachable"


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "api_status": check_api_health(),
            "prediction": None,
            "detail": None,
            "error": None,
            "image_data": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    image_b64 = base64.b64encode(raw).decode("utf-8")

    try:
        resp = requests.post(
            MODEL_API_URL,
            files={"file": (file.filename or "image", raw, file.content_type)},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        prediction = data.get("prediction")
        cached = data.get("cached")
        detail = "cache hit" if cached else "fresh"
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "api_status": check_api_health(),
                "prediction": prediction,
                "detail": detail,
                "error": None,
                "image_data": image_b64,
            },
        )
    except requests.RequestException as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "api_status": check_api_health(),
                "prediction": None,
                "detail": None,
                "error": f"Failed to call model API: {exc}",
                "image_data": image_b64,
            },
        )
