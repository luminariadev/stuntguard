from pathlib import Path

import pandas as pd

from src.train import (
    select_best_model,
    split_data,
    train_and_save_best_model,
    train_decision_tree,
    train_logistic_regression,
    train_random_forest,
)


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


def test_split_data_returns_train_and_test_sets():
    df = _sample_training_data()

    X_train, X_test, y_train, y_test = split_data(
        df,
        ["tahun", "lag_1_stunting", "rolling_mean_3y", "trend_stunting"],
        "risk_label",
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_test) > 0


def test_train_baseline_models_can_predict():
    df = _sample_training_data()
    feature_cols = ["tahun", "lag_1_stunting", "rolling_mean_3y", "trend_stunting"]
    X_train, X_test, y_train, _ = split_data(df, feature_cols, "risk_label")

    for trainer in [
        train_decision_tree,
        train_logistic_regression,
        train_random_forest,
    ]:
        model = trainer(X_train, y_train)
        assert len(model.predict(X_test)) == len(X_test)


def test_select_best_model_returns_metrics_table():
    df = _sample_training_data()
    feature_cols = ["tahun", "lag_1_stunting", "rolling_mean_3y", "trend_stunting"]
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, "risk_label")
    models = {"decision_tree": train_decision_tree(X_train, y_train)}

    best_name, best_model, results = select_best_model(models, X_test, y_test)

    assert best_name == "decision_tree"
    assert best_model is not None
    assert "f1_macro" in results.columns


def test_train_and_save_best_model_writes_joblib(tmp_path):
    df = _sample_training_data()
    output_path = tmp_path / "best_model.joblib"

    best_name, saved_path, results = train_and_save_best_model(
        df,
        model_path=output_path,
    )

    assert best_name in set(results["model"])
    assert saved_path == Path(output_path)
    assert saved_path.exists()
