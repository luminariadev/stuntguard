"""Model training utilities for StuntGuard Jabar ML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from src.utils import resolve_project_path
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import resolve_project_path


DEFAULT_FEATURE_COLS = [
    "tahun",
    "lag_1_stunting",
    "rolling_mean_3y",
    "trend_stunting",
]
DEFAULT_TARGET_COL = "risk_label"
DEFAULT_MODEL_DIR = resolve_project_path("models")
DEFAULT_BEST_MODEL_PATH = DEFAULT_MODEL_DIR / "best_model.joblib"
RANDOM_STATE = 42


class XGBoostRiskClassifier(ClassifierMixin, BaseEstimator):
    """Small wrapper to train XGBoost with string risk labels."""

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.label_encoder = LabelEncoder()
        self.model = None
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostRiskClassifier":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        encoded_y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        default_params = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "mlogloss",
            "random_state": RANDOM_STATE,
        }
        default_params.update(self.params)

        self.model = XGBClassifier(**default_params)
        self.model.fit(X, encoded_y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        encoded_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(encoded_pred)

    def predict_proba(self, X: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        return self.model.predict_proba(X)

    def __sklearn_is_fitted__(self) -> bool:
        return self.model is not None


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Missing required column(s): {missing}")


def _can_stratify(y: pd.Series, test_size: float) -> bool:
    class_counts = y.value_counts()
    if len(class_counts) < 2 or (class_counts < 2).any():
        return False

    test_count = max(1, int(round(len(y) * test_size)))
    train_count = len(y) - test_count
    return test_count >= len(class_counts) and train_count >= len(class_counts)


def _build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical feature columns."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Split features and target, using stratify when class counts allow it."""
    _validate_columns(df, [*feature_cols, target_col])
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    data = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    X = data[feature_cols]
    y = data[target_col]
    stratify = y if _can_stratify(y, test_size) else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def train_decision_tree(X_train, y_train) -> Pipeline:
    """Train a Decision Tree classifier."""
    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor(X_train)),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model.fit(X_train, y_train)


def train_logistic_regression(X_train, y_train) -> Pipeline:
    """Train a Logistic Regression baseline classifier."""
    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor(X_train, scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model.fit(X_train, y_train)


def train_random_forest(X_train, y_train) -> Pipeline:
    """Train a Random Forest classifier."""
    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor(X_train)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return model.fit(X_train, y_train)


def train_xgboost(X_train, y_train) -> Pipeline:
    """Train an XGBoost classifier."""
    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor(X_train)),
            ("classifier", XGBoostRiskClassifier()),
        ]
    )
    return model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Evaluate a classifier with the project metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def train_all_models(X_train, y_train) -> dict[str, Any]:
    """Train all baseline and main models required by the README."""
    models = {
        "decision_tree": train_decision_tree(X_train, y_train),
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
    }

    try:
        models["xgboost"] = train_xgboost(X_train, y_train)
    except ImportError:
        models["xgboost"] = None

    return models


def select_best_model(
    trained_models: dict[str, Any],
    X_test,
    y_test,
    metric: str = "f1_macro",
) -> tuple[str, Any, pd.DataFrame]:
    """Select the best available model based on a validation metric."""
    rows = []
    best_name = None
    best_model = None
    best_score = float("-inf")

    for name, model in trained_models.items():
        if model is None:
            rows.append({"model": name, "status": "skipped"})
            continue

        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, "status": "trained", **metrics})
        score = metrics[metric]
        if score > best_score:
            best_name = name
            best_model = model
            best_score = score

    if best_name is None or best_model is None:
        raise ValueError("No model was trained successfully")

    return best_name, best_model, pd.DataFrame(rows)


def save_model(model, path) -> Path:
    """Save a trained model with joblib and return the output path."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def train_and_save_best_model(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = DEFAULT_TARGET_COL,
    model_path: str | Path = DEFAULT_BEST_MODEL_PATH,
) -> tuple[str, Path, pd.DataFrame]:
    """Train all models, save the best one, and return its metadata."""
    selected_features = feature_cols or [
        col for col in DEFAULT_FEATURE_COLS if col in df.columns
    ]
    if not selected_features:
        raise ValueError("No feature columns were provided or found in the dataframe")

    X_train, X_test, y_train, y_test = split_data(df, selected_features, target_col)
    trained_models = train_all_models(X_train, y_train)
    best_name, best_model, results = select_best_model(trained_models, X_test, y_test)
    output_path = save_model(best_model, model_path)
    return best_name, output_path, results


def main() -> None:
    """Train from the default processed dataset path."""
    data_path = resolve_project_path("data/processed/stunting_jabar_features.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Expected "
            "`data/processed/stunting_jabar_features.csv`."
        )

    df = pd.read_csv(data_path)
    best_name, output_path, results = train_and_save_best_model(df)
    print(f"Best model: {best_name}")
    print(f"Saved to: {output_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
