"""Prediction utilities for StuntGuard Jabar ML."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

try:
    from src.utils import resolve_project_path
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import resolve_project_path


def _as_dataframe(input_data: Any) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data
    if isinstance(input_data, pd.Series):
        return input_data.to_frame().T
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])
    return pd.DataFrame(input_data)


def load_model(path):
    """Load a serialized model from a project-relative or absolute path."""
    model_path = resolve_project_path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_risk(model, input_data):
    """Predict risk labels for one or more input rows."""
    features = _as_dataframe(input_data)
    predictions = model.predict(features)
    return predictions[0] if len(predictions) == 1 else predictions


def predict_probability(model, input_data):
    """Predict class probabilities when the model supports it."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support probability prediction")
    features = _as_dataframe(input_data)
    probabilities = model.predict_proba(features)
    return probabilities[0] if len(probabilities) == 1 else probabilities
