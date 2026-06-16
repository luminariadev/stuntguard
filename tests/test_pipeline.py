import pandas as pd

from src.pipeline import build_processed_dataset, train_evaluate_and_save


def test_build_processed_dataset_writes_clean_and_features(tmp_path):
    percentage_path = tmp_path / "persentase.csv"
    clean_path = tmp_path / "clean.csv"
    features_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "Nama Kabupaten/Kota": ["kab. bandung", "kab. bandung", "kota bogor"],
            "Tahun": [2022, 2023, 2022],
            "Persentase Stunting": [10, 12, 20],
        }
    ).to_csv(percentage_path, index=False)

    features = build_processed_dataset(
        percentage_path=percentage_path,
        count_path=None,
        clean_path=clean_path,
        features_path=features_path,
    )

    assert clean_path.exists()
    assert features_path.exists()
    assert {"lag_1_stunting", "rolling_mean_3y", "trend_stunting", "risk_label"}.issubset(
        features.columns
    )


def test_train_evaluate_and_save_writes_artifacts(tmp_path):
    labels = ["Rendah", "Sedang", "Tinggi"] * 10
    df = pd.DataFrame(
        {
            "tahun": list(range(2020, 2050)),
            "lag_1_stunting": [10, 20, 30] * 10,
            "rolling_mean_3y": [11, 21, 31] * 10,
            "trend_stunting": [0.5, 1.0, 1.5] * 10,
            "risk_label": labels,
        }
    )

    best_name, model_path, comparison = train_evaluate_and_save(
        df,
        model_path=tmp_path / "best_model.joblib",
        reports_dir=tmp_path / "reports",
    )

    assert best_name in set(comparison["model"])
    assert model_path.exists()
    assert (tmp_path / "reports" / "tables" / "model_comparison.csv").exists()
    assert (tmp_path / "reports" / "figures" / "confusion_matrix.png").exists()
