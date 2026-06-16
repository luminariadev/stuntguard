"""FastAPI service for StuntGuard model prediction."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import load_model, predict_probability, predict_risk


MODEL_PATH = "models/best_model.joblib"
MODEL_NAME = "best_model"

app = FastAPI(title="StuntGuard Model API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    tahun: int = Field(..., ge=2014, le=2035)
    lag_1_stunting: float
    rolling_mean_3y: float
    trend_stunting: float


class PredictionResponse(BaseModel):
    risk_label: str
    probability: float | None
    probabilities: dict[str, float] | None
    model_name: str


def _get_model():
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Model belum tersedia. Jalankan `python -m src.pipeline`.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memuat model: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict stunting risk for one input row."""
    model = _get_model()
    input_data: dict[str, Any] = payload.model_dump()

    try:
        risk_label = str(predict_risk(model, input_data))
        probabilities = predict_probability(model, input_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {exc}") from exc

    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)
        if classes is None and hasattr(classifier, "label_encoder"):
            classes = classifier.label_encoder.classes_

    probabilities_map = None
    confidence = None
    if classes is not None:
        probabilities_map = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }
        confidence = probabilities_map.get(risk_label)
    elif probabilities is not None:
        confidence = float(max(probabilities))

    return PredictionResponse(
        risk_label=risk_label,
        probability=confidence,
        probabilities=probabilities_map,
        model_name=MODEL_NAME,
    )
