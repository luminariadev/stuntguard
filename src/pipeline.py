"""End-to-end project pipeline for preprocessing, training, and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from src.data_loader import load_csv, save_csv
    from src.evaluate import evaluate_and_save, save_evaluation_result
    from src.preprocessing import (
        REGION_COL,
        TARGET_COL,
        YEAR_COL,
        clean_column_names,
        normalize_region_name,
        preprocess_stunting_data,
    )
    from src.train import (
        DEFAULT_FEATURE_COLS,
        DEFAULT_TARGET_COL,
        save_model,
        select_best_model,
        split_data,
        train_all_models,
    )
    from src.utils import resolve_project_path
except ImportError:  # pragma: no cover - supports direct script execution
    from data_loader import load_csv, save_csv
    from evaluate import evaluate_and_save, save_evaluation_result
    from preprocessing import (
        REGION_COL,
        TARGET_COL,
        YEAR_COL,
        clean_column_names,
        normalize_region_name,
        preprocess_stunting_data,
    )
    from train import (
        DEFAULT_FEATURE_COLS,
        DEFAULT_TARGET_COL,
        save_model,
        select_best_model,
        split_data,
        train_all_models,
    )
    from utils import resolve_project_path


DEFAULT_PERCENTAGE_PATH = "data/raw/persentase_stunting_jabar.csv"
DEFAULT_COUNT_PATH = "data/raw/jumlah_stunting_jabar.csv"
DEFAULT_CLEAN_PATH = "data/processed/stunting_jabar_clean.csv"
DEFAULT_FEATURES_PATH = "data/processed/stunting_jabar_features.csv"
DEFAULT_MODEL_PATH = "models/best_model.joblib"
DEFAULT_REPORTS_DIR = "reports"


def _prepare_count_dataset(path: str | Path) -> pd.DataFrame:
    """Load and standardize the optional stunting count dataset."""
    count_df = load_csv(path)
    count_df = clean_column_names(count_df)
    count_df = normalize_region_name(count_df)

    required_cols = [REGION_COL, YEAR_COL, "jumlah_stunting"]
    missing_cols = [col for col in required_cols if col not in count_df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Count dataset missing required column(s): {missing}")

    return count_df[required_cols].drop_duplicates()


def build_processed_dataset(
    percentage_path: str | Path = DEFAULT_PERCENTAGE_PATH,
    count_path: str | Path | None = DEFAULT_COUNT_PATH,
    clean_path: str | Path = DEFAULT_CLEAN_PATH,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
) -> pd.DataFrame:
    """Create clean and feature datasets from raw stunting CSV files."""
    percentage_df = load_csv(percentage_path)
    clean_df = clean_column_names(percentage_df)
    clean_df = normalize_region_name(clean_df)

    if TARGET_COL not in clean_df.columns:
        raise ValueError(f"Percentage dataset must contain `{TARGET_COL}`")

    resolved_count_path = resolve_project_path(count_path) if count_path else None
    if resolved_count_path and resolved_count_path.exists():
        count_df = _prepare_count_dataset(resolved_count_path)
        merge_cols = [REGION_COL, YEAR_COL]
        missing_merge_cols = [col for col in merge_cols if col not in clean_df.columns]
        if missing_merge_cols:
            missing = ", ".join(missing_merge_cols)
            raise ValueError(
                "Percentage dataset cannot be merged with count dataset. "
                f"Missing column(s): {missing}"
            )
        clean_df = clean_df.merge(count_df, on=merge_cols, how="left")

    save_csv(clean_df, clean_path)
    features_df = preprocess_stunting_data(clean_df)
    save_csv(features_df, features_path)
    return features_df


def train_evaluate_and_save(
    df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> tuple[str, Path, pd.DataFrame]:
    """Train all available models, save the best model, and write reports."""
    feature_cols = [col for col in DEFAULT_FEATURE_COLS if col in df.columns]
    if not feature_cols:
        raise ValueError("Processed dataset does not contain training feature columns")
    if DEFAULT_TARGET_COL not in df.columns:
        raise ValueError(f"Processed dataset must contain `{DEFAULT_TARGET_COL}`")

    X_train, X_test, y_train, y_test = split_data(df, feature_cols, DEFAULT_TARGET_COL)
    trained_models = train_all_models(X_train, y_train)
    best_name, best_model, model_comparison = select_best_model(
        trained_models, X_test, y_test
    )
    output_model_path = save_model(best_model, model_path)

    reports_path = resolve_project_path(reports_dir)
    save_evaluation_result(model_comparison, reports_path / "tables/model_comparison.csv")
    evaluate_and_save(best_model, X_test, y_test, feature_cols, reports_path)
    return best_name, output_model_path, model_comparison


def run_pipeline(args: argparse.Namespace) -> tuple[str, Path, pd.DataFrame]:
    """Run preprocessing, training, evaluation, and artifact saving."""
    features_df = build_processed_dataset(
        percentage_path=args.percentage_path,
        count_path=args.count_path,
        clean_path=args.clean_path,
        features_path=args.features_path,
    )
    return train_evaluate_and_save(
        features_df,
        model_path=args.model_path,
        reports_dir=args.reports_dir,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the StuntGuard preprocessing, training, and evaluation pipeline."
    )
    parser.add_argument("--percentage-path", default=DEFAULT_PERCENTAGE_PATH)
    parser.add_argument("--count-path", default=DEFAULT_COUNT_PATH)
    parser.add_argument("--clean-path", default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--features-path", default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for the end-to-end pipeline."""
    best_name, model_path, model_comparison = run_pipeline(parse_args())
    print(f"Best model: {best_name}")
    print(f"Saved model: {model_path}")
    print(model_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
