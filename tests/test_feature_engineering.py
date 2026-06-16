import pandas as pd

from src.feature_engineering import (
    add_lag_feature,
    add_rolling_mean,
    add_trend_feature,
    build_features,
)


def test_add_lag_feature_groups_by_region_and_sorts_by_year():
    df = pd.DataFrame(
        {
            "nama_kabupaten_kota": ["A", "A", "B", "B"],
            "tahun": [2024, 2023, 2023, 2024],
            "persentase_stunting": [14, 10, 20, 25],
        }
    )

    result = add_lag_feature(df, "nama_kabupaten_kota", "persentase_stunting")

    row = result[
        (result["nama_kabupaten_kota"] == "A") & (result["tahun"] == 2024)
    ].iloc[0]
    assert row["lag_1_stunting"] == 10


def test_add_rolling_mean_uses_previous_values_only():
    df = pd.DataFrame(
        {
            "nama_kabupaten_kota": ["A", "A", "A"],
            "tahun": [2022, 2023, 2024],
            "persentase_stunting": [10, 20, 30],
        }
    )

    result = add_rolling_mean(df, "nama_kabupaten_kota", "persentase_stunting")

    assert pd.isna(result.loc[result["tahun"] == 2022, "rolling_mean_3y"].iloc[0])
    assert result.loc[result["tahun"] == 2024, "rolling_mean_3y"].iloc[0] == 15


def test_add_trend_feature_calculates_year_over_year_change():
    df = pd.DataFrame(
        {
            "nama_kabupaten_kota": ["A", "A"],
            "tahun": [2023, 2024],
            "persentase_stunting": [12, 15],
        }
    )

    result = add_trend_feature(df, "nama_kabupaten_kota", "persentase_stunting")

    assert result.loc[result["tahun"] == 2024, "trend_stunting"].iloc[0] == 3


def test_build_features_adds_all_baseline_features():
    df = pd.DataFrame(
        {
            "nama_kabupaten_kota": ["A", "A"],
            "tahun": [2023, 2024],
            "persentase_stunting": [12, 15],
        }
    )

    result = build_features(df)

    assert {"lag_1_stunting", "rolling_mean_3y", "trend_stunting"}.issubset(
        result.columns
    )
