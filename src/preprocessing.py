"""Preprocessing utilities for stunting data."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

try:
    from src.feature_engineering import build_features
except ImportError:  # pragma: no cover - supports direct script execution
    from feature_engineering import build_features


REGION_COL = "nama_kabupaten_kota"
TARGET_COL = "persentase_stunting"
YEAR_COL = "tahun"

COLUMN_ALIASES = {
    "nama_kabupaten_atau_kota": REGION_COL,
    "nama_kabupaten_kota": REGION_COL,
    "kabupaten_kota": REGION_COL,
    "kab_kota": REGION_COL,
    "nama_wilayah": REGION_COL,
    "wilayah": REGION_COL,
    "daerah": REGION_COL,
    "persentase_balita_stunting": TARGET_COL,
    "persentase_stunting": TARGET_COL,
    "prevalensi_stunting": TARGET_COL,
    "persen_stunting": TARGET_COL,
    "jumlah_balita_stunting": "jumlah_stunting",
    "jumlah_stunting": "jumlah_stunting",
    "kode_kabupaten_atau_kota": "kode_kabupaten_kota",
    "kode_kabupaten_kota": "kode_kabupaten_kota",
}


def _to_snake_case(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    text = re.sub(r"[%()/.-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _normalize_region_value(value: object) -> object:
    if pd.isna(value):
        return value

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?i)\bkab\.?\s+", "Kabupaten ", text)
    text = re.sub(r"(?i)\bkota\.?\s+", "Kota ", text)
    text = text.title()

    return text


def _parse_numeric_value(value: object) -> object:
    if pd.isna(value):
        return value

    text = str(value).strip().replace("%", "")
    text = re.sub(r"\s+", "", text)

    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")

    return text


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized snake_case column names."""
    result = df.copy()
    result.columns = [_to_snake_case(col) for col in result.columns]
    result = result.rename(columns=COLUMN_ALIASES)
    return result


def normalize_region_name(
    df: pd.DataFrame, region_col: str = REGION_COL
) -> pd.DataFrame:
    """Normalize kabupaten/kota names into a consistent title-case format."""
    result = df.copy()
    if region_col not in result.columns:
        return result

    result[region_col] = result[region_col].map(_normalize_region_value)
    return result


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_candidates = [
        YEAR_COL,
        TARGET_COL,
        "jumlah_stunting",
        "lag_1_stunting",
        "rolling_mean_3y",
        "trend_stunting",
    ]

    for col in numeric_candidates:
        if col not in result.columns:
            continue
        if result[col].dtype == "object":
            result[col] = result[col].map(_parse_numeric_value)
        result[col] = pd.to_numeric(result[col], errors="coerce")

    if YEAR_COL in result.columns:
        result[YEAR_COL] = result[YEAR_COL].astype("Int64")

    return result


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using simple, reproducible defaults."""
    result = df.copy()

    if REGION_COL in result.columns:
        result[REGION_COL] = result[REGION_COL].fillna("Tidak Diketahui")

    historical_feature_cols = [
        "lag_1_stunting",
        "rolling_mean_3y",
        "trend_stunting",
    ]
    for col in historical_feature_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    numeric_cols = result.select_dtypes(include="number").columns
    for col in numeric_cols:
        median_value = result[col].median()
        if pd.isna(median_value):
            median_value = 0
        result[col] = result[col].fillna(median_value)

    object_cols = result.select_dtypes(include="object").columns
    for col in object_cols:
        result[col] = result[col].fillna("Tidak Diketahui")

    return result


def create_risk_label(
    df: pd.DataFrame,
    target_column: str,
    output_col: str = "risk_label",
) -> pd.DataFrame:
    """Create risk labels based on Q1 and Q3 of stunting percentage."""
    if target_column not in df.columns:
        raise ValueError(f"Missing required column: {target_column}")

    result = df.copy()
    target = pd.to_numeric(result[target_column], errors="coerce")

    if target.dropna().empty:
        raise ValueError(f"Column {target_column} does not contain numeric values")

    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)

    def label_risk(value: float) -> str:
        if pd.isna(value):
            return "Tidak Diketahui"
        if value <= q1:
            return "Rendah"
        if value <= q3:
            return "Sedang"
        return "Tinggi"

    result[output_col] = target.map(label_risk)
    return result


def preprocess_stunting_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COL,
    region_col: str = REGION_COL,
) -> pd.DataFrame:
    """Run the baseline preprocessing pipeline for stunting data."""
    result = clean_column_names(df)
    result = normalize_region_name(result, region_col=region_col)
    result = _convert_numeric_columns(result)

    required_cols = [region_col, target_column]
    missing_cols = [col for col in required_cols if col not in result.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Missing required column(s): {missing}")

    result = result.drop_duplicates()
    if YEAR_COL in result.columns:
        result = result.sort_values([region_col, YEAR_COL]).reset_index(drop=True)

    result = build_features(result, group_col=region_col, target_col=target_column)
    result = handle_missing_values(result)
    result = create_risk_label(result, target_column=target_column)
    return result.reset_index(drop=True)
