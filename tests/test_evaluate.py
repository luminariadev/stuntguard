from pathlib import Path

import pandas as pd

from src.evaluate import (
    evaluate_and_save,
    evaluate_classification_model,
    generate_classification_report,
    get_feature_importance,
    plot_confusion_matrix,
    save_evaluation_result,
)
from src.train import split_data, train_random_forest


def _sample_training_data():
    labels = ["Rendah", "Sedang", "Tinggi"] * 10
    return pd.DataFrame(
        {
            "tahun": list(range(2020, 2050)),
            "lag_1_stunting": [10, 20, 30] * 10,
            "rolling_mean_3y": [11, 21, 31] * 10,
            "trend_stunting": [0.5, 1.0, 1.5] * 10,
            "risk_label": labels,
        }
    )


def test_evaluate_classification_model_returns_metrics():
    df = _sample_training_data()
    feature_cols = ["tahun", "lag_1_stunting", "rolling_mean_3y", "trend_stunting"]
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, "risk_label")
    model = train_random_forest(X_train, y_train)

    metrics = evaluate_classification_model(model, X_test, y_test)

    assert {"accuracy", "precision_macro", "recall_macro", "f1_macro"}.issubset(
        metrics
    )


def test_generate_classification_report_returns_dataframe():
    report = generate_classification_report(
        ["Rendah", "Sedang", "Tinggi"],
        ["Rendah", "Tinggi", "Tinggi"],
    )

    assert {"label", "precision", "recall", "f1-score", "support"}.issubset(
        report.columns
    )


def test_plot_confusion_matrix_writes_png(tmp_path):
    output_path = tmp_path / "confusion_matrix.png"

    saved_path = plot_confusion_matrix(
        ["Rendah", "Sedang", "Tinggi"],
        ["Rendah", "Tinggi", "Tinggi"],
        output_path,
    )

    assert saved_path == output_path
    assert saved_path.exists()


def test_get_feature_importance_returns_dataframe():
    df = _sample_training_data()
    feature_cols = ["tahun", "lag_1_stunting", "rolling_mean_3y", "trend_stunting"]
    X_train, X_test, y_train, _ = split_data(df, feature_cols, "risk_label")
    model = train_random_forest(X_train, y_train)

    importance = get_feature_importance(model, feature_cols)

    assert {"feature", "importance"}.issubset(importance.columns)
    assert not importance.empty


def test_save_evaluation_result_writes_csv(tmp_path):
    output_path = tmp_path / "metrics.csv"

    saved_path = save_evaluation_result({"accuracy": 1.0}, output_path)

    assert saved_path == Path(output_path)
    assert saved_path.exists()


def test_evaluate_and_save_writes_report_artifacts(tmp_path):
    df = _sample_training_data()
    feature_cols = ["tahun", "lag_1_stunting", "rolling_mean_3y", "trend_stunting"]
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, "risk_label")
    model = train_random_forest(X_train, y_train)

    saved_paths = evaluate_and_save(model, X_test, y_test, feature_cols, tmp_path)

    assert set(saved_paths) == {
        "metrics",
        "classification_report",
        "feature_importance_table",
        "confusion_matrix",
        "feature_importance_plot",
    }
    assert all(path.exists() for path in saved_paths.values())
