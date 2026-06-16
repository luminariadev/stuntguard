import pandas as pd

from src.preprocessing import (
    clean_column_names,
    create_risk_label,
    normalize_region_name,
    preprocess_stunting_data,
)


def test_clean_column_names_normalizes_common_aliases():
    df = pd.DataFrame(
        {
            "Nama Kabupaten/Kota": ["KAB. BANDUNG"],
            "Persentase Balita Stunting (%)": ["12,5"],
        }
    )

    result = clean_column_names(df)

    assert "nama_kabupaten_kota" in result.columns
    assert "persentase_stunting" in result.columns


def test_normalize_region_name_expands_kabbreviation():
    df = pd.DataFrame({"nama_kabupaten_kota": ["kab. bandung"]})

    result = normalize_region_name(df)

    assert result.loc[0, "nama_kabupaten_kota"] == "Kabupaten Bandung"


def test_create_risk_label_uses_quartiles():
    df = pd.DataFrame({"persentase_stunting": [10, 20, 30, 40]})

    result = create_risk_label(df, "persentase_stunting")

    assert result["risk_label"].tolist() == ["Rendah", "Sedang", "Sedang", "Tinggi"]


def test_preprocess_stunting_data_builds_features_and_label():
    df = pd.DataFrame(
        {
            "Nama Kabupaten/Kota": ["kab. bandung", "kab. bandung", "kab. bandung"],
            "Tahun": [2022, 2023, 2024],
            "Persentase Stunting": ["10.5", "12,5", "15"],
        }
    )

    result = preprocess_stunting_data(df)

    assert result.loc[0, "nama_kabupaten_kota"] == "Kabupaten Bandung"
    assert result.loc[1, "lag_1_stunting"] == 10.5
    assert result.loc[2, "rolling_mean_3y"] == 11.5
    assert result.loc[2, "trend_stunting"] == 2.5
    assert "risk_label" in result.columns
