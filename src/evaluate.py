"""Model evaluation utilities for StuntGuard Jabar ML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

try:
    from src.utils import resolve_project_path
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import resolve_project_path

DEFAULT_REPORTS_DIR = resolve_project_path("reports")
DEFAULT_FIGURES_DIR = DEFAULT_REPORTS_DIR / "figures"
DEFAULT_TABLES_DIR = DEFAULT_REPORTS_DIR / "tables"
DEFAULT_MODEL_PATH = resolve_project_path("models/best_model.joblib")
DEFAULT_DATA_PATH = resolve_project_path("data/processed/stunting_jabar_features.csv")
DEFAULT_FEATURE_COLS = [
    "tahun",
    "lag_1_stunting",
    "rolling_mean_3y",
    "trend_stunting",
]
DEFAULT_TARGET_COL = "risk_label"


def _as_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values)


def _clean_feature_name(feature_name: str) -> str:
    return (
        feature_name.replace("numeric__", "")
        .replace("categorical__", "")
        .replace("remainder__", "")
    )


def _get_classifier(model):
    """Return the final estimator from a plain model or sklearn Pipeline."""
    if isinstance(model, Pipeline) and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
    else:
        classifier = model

    if hasattr(classifier, "model") and classifier.model is not None:
        return classifier.model
    return classifier


def _get_feature_names(model, feature_names: list[str]) -> list[str]:
    """Return transformed feature names when a pipeline exposes them."""
    if not isinstance(model, Pipeline) or "preprocessor" not in model.named_steps:
        return feature_names

    preprocessor = model.named_steps["preprocessor"]
    try:
        transformed_names = preprocessor.get_feature_names_out(feature_names)
        return [_clean_feature_name(str(name)) for name in transformed_names]
    except Exception:
        return feature_names


def evaluate_classification_model(model, X_test, y_test) -> dict[str, float]:
    """Calculate core classification metrics for a trained model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def generate_classification_report(y_test, y_pred) -> pd.DataFrame:
    """Return sklearn classification report as a dataframe."""
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose().reset_index(names="label")


def plot_confusion_matrix(
    y_test,
    y_pred,
    path: str | Path = DEFAULT_FIGURES_DIR / "confusion_matrix.png",
    labels: list[str] | None = None,
    normalize: str | None = None,
) -> Path:
    """Create and save a confusion matrix plot."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = _as_series(y_test)
    y_prediction = _as_series(y_pred)
    display_labels = labels or sorted(set(y_true.dropna()) | set(y_prediction.dropna()))
    matrix = confusion_matrix(
        y_true,
        y_prediction,
        labels=display_labels,
        normalize=normalize,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_labels,
    )
    display.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    """Extract feature importance or coefficient magnitude when available."""
    names = _get_feature_names(model, list(feature_names))
    classifier = _get_classifier(model)

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
        importance = abs(coef).mean(axis=0) if getattr(coef, "ndim", 1) > 1 else abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if len(importance) != len(names):
        names = [f"feature_{index}" for index in range(len(importance))]

    result = pd.DataFrame({"feature": names, "importance": importance})
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    path: str | Path = DEFAULT_FIGURES_DIR / "feature_importance.png",
    top_n: int = 15,
) -> Path:
    """Create and save a horizontal feature importance chart."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    if feature_importance.empty:
        ax.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        plot_data = feature_importance.head(top_n).iloc[::-1]
        ax.barh(plot_data["feature"], plot_data["importance"], color="#2f80ed")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_evaluation_result(result, path) -> Path:
    """Save evaluation result as CSV, JSON, or text based on file extension."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(result, pd.DataFrame):
        if output_path.suffix.lower() == ".json":
            result.to_json(output_path, orient="records", indent=2)
        else:
            result.to_csv(output_path, index=False)
        return output_path

    if isinstance(result, dict):
        if output_path.suffix.lower() == ".csv":
            pd.DataFrame([result]).to_csv(output_path, index=False)
        else:
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return output_path

    output_path.write_text(str(result), encoding="utf-8")
    return output_path


def evaluate_and_save(
    model,
    X_test,
    y_test,
    feature_names: list[str],
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> dict[str, Path]:
    """Evaluate a model and save all required report artifacts."""
    reports_path = resolve_project_path(reports_dir)
    figures_dir = reports_path / "figures"
    tables_dir = reports_path / "tables"

    y_pred = model.predict(X_test)
    metrics = evaluate_classification_model(model, X_test, y_test)
    report_df = generate_classification_report(y_test, y_pred)
    importance_df = get_feature_importance(model, feature_names)

    saved_paths = {
        "metrics": save_evaluation_result(
            metrics, tables_dir / "best_model_metrics.csv"
        ),
        "classification_report": save_evaluation_result(
            report_df, tables_dir / "classification_report.csv"
        ),
        "feature_importance_table": save_evaluation_result(
            importance_df, tables_dir / "feature_importance.csv"
        ),
        "confusion_matrix": plot_confusion_matrix(
            y_test, y_pred, figures_dir / "confusion_matrix.png"
        ),
        "feature_importance_plot": plot_feature_importance(
            importance_df, figures_dir / "feature_importance.png"
        ),
    }
    return saved_paths


def main() -> None:
    """Evaluate the default saved model against the processed dataset."""
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {DEFAULT_MODEL_PATH}")
    if not DEFAULT_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DEFAULT_DATA_PATH}")

    model = joblib.load(DEFAULT_MODEL_PATH)
    df = pd.read_csv(DEFAULT_DATA_PATH)
    feature_cols = [col for col in DEFAULT_FEATURE_COLS if col in df.columns]
    if DEFAULT_TARGET_COL not in df.columns or not feature_cols:
        raise ValueError(
            "Dataset must include risk_label and at least one default feature column"
        )

    X_test = df[feature_cols]
    y_test = df[DEFAULT_TARGET_COL]
    saved_paths = evaluate_and_save(model, X_test, y_test, feature_cols)
    for name, path in saved_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
