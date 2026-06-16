"""FastAPI service for StuntGuard model prediction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import load_model, predict_probability, predict_risk


MODEL_PATH = "models/best_model.joblib"
MODEL_NAME = "best_model"

app = FastAPI(title="StuntGuard Model API", version="0.1.0")

# CORS Configuration - Tambahkan semua port yang mungkin digunakan
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Request model untuk prediksi stunting."""
    tahun: int = Field(..., ge=2014, le=2035, description="Tahun prediksi (2014-2035)")
    lag_1_stunting: float = Field(..., description="Nilai stunting tahun sebelumnya (%)")
    rolling_mean_3y: float = Field(..., description="Rata-rata stunting 3 tahun terakhir (%)")
    trend_stunting: float = Field(..., description="Trend/pergerakan stunting (%)")


class PredictionResponse(BaseModel):
    """Response model untuk hasil prediksi."""
    risk_label: str = Field(..., description="Label risiko: Rendah/Sedang/Tinggi")
    probability: float | None = Field(None, description="Probabilitas untuk kelas yang diprediksi")
    probabilities: dict[str, float] | None = Field(None, description="Probabilitas untuk semua kelas")
    model_name: str = Field(..., description="Nama model yang digunakan")


def _get_model():
    """Load model dari file dengan error handling."""
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Model belum tersedia. Jalankan `python -m src.pipeline` terlebih dahulu.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memuat model: {str(exc)}",
        ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint untuk monitoring."""
    return {"status": "ok", "model_loaded": True}


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint dengan informasi API."""
    return {
        "message": "StuntGuard Model API",
        "version": "0.1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Prediksi risiko stunting",
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Endpoint untuk prediksi risiko stunting.
    
    Args:
        payload: Data input dengan tahun dan fitur stunting.
    
    Returns:
        PredictionResponse: Hasil prediksi dengan label risiko dan probabilitas.
    
    Raises:
        HTTPException: Jika model gagal dimuat atau prediksi gagal.
    """
    # Load model
    model = _get_model()
    
    # Konversi input ke DataFrame (ini langkah kritis!)
    input_dict: dict[str, Any] = payload.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    try:
        # Prediksi risiko (label)
        risk_label = predict_risk(model, input_df)
        # Pastikan risk_label adalah string
        if hasattr(risk_label, "iloc"):
            risk_label = str(risk_label.iloc[0])
        else:
            risk_label = str(risk_label)
        
        # Prediksi probabilitas
        probabilities = predict_probability(model, input_df)
        # Pastikan probabilities adalah array/list
        if hasattr(probabilities, "iloc"):
            probabilities = probabilities.iloc[0].tolist()
        elif hasattr(probabilities, "tolist"):
            probabilities = probabilities.tolist()
        elif isinstance(probabilities, (list, tuple)):
            probabilities = list(probabilities)
        else:
            probabilities = [float(probabilities)]
        
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediksi gagal: {str(exc)}",
        ) from exc
    
    # Dapatkan class labels dari model
    classes = None
    if hasattr(model, "classes_"):
        classes = model.classes_
    elif hasattr(model, "named_steps"):
        # Untuk pipeline
        classifier = model.named_steps.get("classifier")
        if classifier is not None:
            if hasattr(classifier, "classes_"):
                classes = classifier.classes_
            elif hasattr(classifier, "label_encoder"):
                classes = classifier.label_encoder.classes_
    
    # Buat mapping probabilitas
    probabilities_map = None
    confidence = None
    
    if classes is not None:
        # Pastikan jumlah classes sesuai dengan probabilities
        if len(classes) == len(probabilities):
            probabilities_map = {
                str(label): float(prob)
                for label, prob in zip(classes, probabilities)
            }
            # Ambil confidence untuk kelas yang diprediksi
            confidence = probabilities_map.get(risk_label)
    
    # Jika tidak ada classes, gunakan max probability
    if confidence is None and probabilities:
        confidence = float(max(probabilities))
    
    return PredictionResponse(
        risk_label=risk_label,
        probability=confidence,
        probabilities=probabilities_map,
        model_name=MODEL_NAME,
    )


@app.options("/predict")
async def options_predict():
    """
    Handler OPTIONS untuk CORS preflight.
    FastAPI secara otomatis menangani ini, tapi kita tambahkan untuk transparansi.
    """
    return {"message": "OK"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )