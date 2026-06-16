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


def inject_design_system() -> None:
    """Apply the PulsePoint-inspired dark dashboard theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #0F172A;
            --surface: #1E293B;
            --surface-2: #334155;
            --navy: #1E3A5F;
            --steel: #8D99AE;
            --muted: #64748B;
            --text: #F1F5F9;
            --red: #E63946;
            --green: #22C55E;
            --amber: #F59E0B;
            --blue: #3B82F6;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg);
            color: var(--text);
            font-family: Inter, sans-serif;
        }

        [data-testid="stSidebar"] {
            background: #111C33;
            border-right: 1px solid #334155;
        }

        [data-testid="stHeader"] {
            background: rgba(15, 23, 42, 0.92);
            border-bottom: 1px solid #1E293B;
        }

        h1, h2, h3 {
            color: var(--text);
            letter-spacing: 0;
        }

        p, li, label, [data-testid="stMarkdownContainer"] {
            color: #CBD5E1;
        }

        code {
            color: #BFDBFE;
            background: #111827;
            border: 1px solid #334155;
            border-radius: 4px;
            padding: 2px 6px;
        }

        [data-testid="stMetric"],
        [data-testid="stDataFrame"],
        [data-testid="stAlert"],
        div[data-testid="stForm"] {
            background: var(--surface);
            border: 1px solid #334155;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35), 0 0 16px rgba(30, 58, 95, 0.18);
        }

        [data-testid="stMetric"] {
            padding: 12px 14px;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--steel);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 800;
        }

        .stButton button,
        .stDownloadButton button,
        [data-testid="stFormSubmitButton"] button {
            background: var(--navy);
            color: var(--text);
            border: 1px solid #3B82F6;
            border-radius: 4px;
            font-weight: 700;
            min-height: 34px;
            box-shadow: 0 0 12px rgba(59, 130, 246, 0.18);
        }

        .stButton button:hover,
        .stDownloadButton button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: #264A75;
            color: #FFFFFF;
            border-color: #60A5FA;
        }

        input, textarea, div[data-baseweb="select"] > div {
            background: #0F172A !important;
            color: var(--text) !important;
            border-color: #334155 !important;
            border-radius: 4px !important;
        }

        [data-testid="stAlert"] {
            border-left: 3px solid var(--blue);
        }

        [data-testid="stAlert"] p {
            color: #E2E8F0;
        }

        a {
            color: #60A5FA !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 4px;
            color: #CBD5E1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_setup(title: str) -> None:
    """Configure a Streamlit page with the project naming convention."""
    st.set_page_config(page_title=f"{title} | StuntGuard Jabar ML", layout="wide")
    inject_design_system()


def status_chip(label: str) -> str:
    """Return HTML for a compact risk status chip."""
    color = risk_color(label)
    return (
        f"<span style='display:inline-block;border:1px solid {color};"
        f"background:{color}22;color:{color};border-radius:4px;"
        "padding:2px 8px;font-size:11px;font-weight:800;"
        "text-transform:uppercase;'>"
        f"{label}</span>"
    )


def metric_panel(title: str, value: str, caption: str | None = None) -> None:
    """Render a compact dashboard metric panel."""
    caption_html = f"<div class='sg-caption'>{caption}</div>" if caption else ""
    st.markdown(
        f"""
        <div style="background:#1E293B;border:1px solid #334155;border-radius:6px;
        padding:14px 16px;box-shadow:0 0 16px rgba(30,58,95,.18);">
            <div style="font-size:11px;color:#8D99AE;font-weight:800;
            text-transform:uppercase;letter-spacing:.08em;">{title}</div>
            <div style="font-size:28px;color:#F1F5F9;font-weight:800;
            line-height:1.2;margin-top:4px;">{value}</div>
            <div style="font-size:12px;color:#64748B;margin-top:4px;">{caption_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
