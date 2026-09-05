"""Week 4 - FastAPI serving layer for the fraud XGBoost model.

Routes:
    GET  /health    liveness + model fingerprint (path, mtime, size, loaded_at)
    POST /predict   one transaction -> {fraud_probability, prediction, latency_ms}
    GET  /metrics   Prometheus text format (counters + latency histogram)

Model is loaded ONCE at startup via the lifespan handler (the @app.on_event
"startup" mechanism is deprecated in modern FastAPI), not per-request.
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from feature_encoder import encode_vector
from schemas import PredictionResponse, TransactionRequest

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/airflow/models/xgb_fraud_model.json")
THRESHOLD = 0.5  # == XGBoost default decision boundary; configurable later

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Model registry state (filled once by lifespan).
MODEL = {"model": None, "mtime": None, "size": None, "loaded_at": None}

# Prometheus instrumentation (scraped directly at /metrics - no separate exporter).
PREDICT_REQUESTS = Counter("predict_requests_total", "Total /predict requests")
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Prediction request latency in seconds",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICT_OUTCOME = Counter(
    "predict_outcome_total", "Predictions by outcome", ["outcome"]
)


def _load_model() -> xgb.XGBClassifier:
    """sklearn-wrapper reload: XGBClassifier().load_model() -> predict_proba() exists."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def _load_model_into_registry() -> None:
    MODEL["model"] = _load_model()
    stat = os.stat(MODEL_PATH)
    MODEL["mtime"] = stat.st_mtime
    MODEL["size"] = stat.st_size
    MODEL["loaded_at"] = time.time()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model_into_registry()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Serves the Week 3 XGBoost fraud model.",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    """Demo console UI - POST /predict playground."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if MODEL["model"] is not None else "model_not_loaded",
        "model_path": MODEL_PATH,
        "artifact_mtime": MODEL["mtime"],
        "artifact_size_bytes": MODEL["size"],
        "loaded_at": MODEL["loaded_at"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest) -> PredictionResponse:
    t0 = time.perf_counter()

    X = encode_vector(req).reshape(1, -1)
    fraud_probability = float(MODEL["model"].predict_proba(X)[0, 1])
    prediction = 1 if fraud_probability >= THRESHOLD else 0

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    PREDICT_REQUESTS.inc()
    PREDICT_OUTCOME.labels(outcome=str(prediction)).inc()
    PREDICT_LATENCY.observe(latency_ms / 1000)

    return PredictionResponse(
        fraud_probability=round(fraud_probability, 4),
        prediction=prediction,
        latency_ms=latency_ms,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)