from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "stunting_jabar_features.csv",
    PROJECT_ROOT / "data" / "processed" / "stunting_jabar_clean.csv",
]
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "best_model.joblib",
    PROJECT_ROOT / "models" / "best_model.pkl",
    PROJECT_ROOT / "models" / "random_forest_model.pkl",
    PROJECT_ROOT / "models" / "xgboost_model.pkl",
]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

DEFAULT_FEATURE_COLS = [
    "tahun",
    "lag_1_stunting",
    "rolling_mean_3y",
    "trend_stunting",
]
TARGET_COL = "risk_label"
REGION_COL = "nama_kabupaten_kota"
STUNTING_COL = "persentase_stunting"


def page_setup(title: str) -> None:
    """Configure a Streamlit page with the project naming convention."""
    st.set_page_config(page_title=f"{title} | StuntGuard Jabar ML", layout="wide")


def load_dataset() -> tuple[pd.DataFrame | None, Path | None, str | None]:
    """Load the first available processed dataset."""
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                return pd.read_csv(path), path, None
            except Exception as exc:
                return None, path, f"Gagal membaca dataset `{path.name}`: {exc}"
    return None, None, None


def load_csv(path: str) -> tuple[pd.DataFrame | None, str | None]:
    """Load a CSV file for report tables without crashing the dashboard."""
    file_path = Path(path)
    if not file_path.exists():
        return None, None
    try:
        return pd.read_csv(file_path), None
    except Exception as exc:
        return None, f"Gagal membaca `{file_path.name}`: {exc}"


def load_model() -> tuple[Any | None, Path | None, str | None]:
    """Load the first available model artifact."""
    try:
        import joblib
    except ImportError:
        return None, None, "Package joblib belum terpasang."

    for path in MODEL_CANDIDATES:
        if path.exists():
            try:
                return joblib.load(path), path, None
            except Exception as exc:
                return None, path, f"Gagal membaca model `{path.name}`: {exc}"
    return None, None, None


def project_relative(path: Path | None) -> str:
    """Format paths relative to the project root for display."""
    if path is None:
        return "-"
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def show_missing_data_message(error: str | None = None) -> None:
    """Show a consistent empty-state message for unavailable datasets."""
    st.warning("Dataset processed belum tersedia.")
    if error:
        st.caption(error)
    else:
        st.caption(
            "Dashboard tetap berjalan. Setelah pipeline preprocessing menghasilkan "
            "`data/processed/stunting_jabar_features.csv`, halaman ini akan terisi."
        )


def show_missing_model_message(error: str | None = None) -> None:
    """Show a consistent empty-state message for unavailable models."""
    st.warning("Model belum tersedia.")
    if error:
        st.caption(error)
    else:
        st.caption(
            "Train model terlebih dahulu agar file seperti "
            "`models/best_model.joblib` tersedia."
        )


def available_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return available baseline feature columns from a dataframe."""
    return [col for col in DEFAULT_FEATURE_COLS if col in df.columns]


def format_number(value: Any, digits: int = 2) -> str:
    """Format a scalar value for compact dashboard display."""
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}"


def risk_color(label: str) -> str:
    """Return a display color for a risk label."""
    colors = {
        "Rendah": "#1f9d55",
        "Sedang": "#d97706",
        "Tinggi": "#dc2626",
    }
    return colors.get(label, "#4b5563")
