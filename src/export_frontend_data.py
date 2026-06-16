"""Export processed pipeline outputs as JSON for the React dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from src.utils import resolve_project_path
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import resolve_project_path


def _read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    file_path = resolve_project_path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def export_frontend_data(
    dataset_path: str | Path = "data/processed/stunting_jabar_features.csv",
    model_comparison_path: str | Path = "reports/tables/model_comparison.csv",
    feature_importance_path: str | Path = "reports/tables/feature_importance.csv",
    output_path: str | Path = "frontend/public/stuntguard-dashboard.json",
) -> Path:
    """Export processed data and report summaries for the React UI."""
    df = _read_csv_if_exists(dataset_path)
    comparison = _read_csv_if_exists(model_comparison_path)
    importance = _read_csv_if_exists(feature_importance_path)

    if df.empty:
        raise FileNotFoundError("Processed dataset is required before frontend export")

    summary = {
        "rows": int(len(df)),
        "regions": int(df["nama_kabupaten_kota"].nunique()),
        "year_min": int(df["tahun"].min()),
        "year_max": int(df["tahun"].max()),
        "average_stunting": float(df["persentase_stunting"].mean()),
    }
    if not comparison.empty and "f1_macro" in comparison.columns:
        best = comparison.sort_values("f1_macro", ascending=False).iloc[0]
        summary["best_model"] = best["model"]
        summary["best_macro_f1"] = float(best["f1_macro"])

    yearly_trend = (
        df.groupby("tahun", as_index=False)["persentase_stunting"]
        .mean()
        .sort_values("tahun")
        .to_dict(orient="records")
    )
    risk_distribution = (
        df["risk_label"]
        .value_counts()
        .rename_axis("risk_label")
        .reset_index(name="jumlah")
        .to_dict(orient="records")
    )

    payload = {
        "summary": summary,
        "yearly_trend": yearly_trend,
        "risk_distribution": risk_distribution,
        "model_comparison": comparison.to_dict(orient="records"),
        "feature_importance": importance.to_dict(orient="records"),
        "sample_rows": df[
            ["nama_kabupaten_kota", "tahun", "persentase_stunting", "risk_label"]
        ]
        .tail(12)
        .to_dict(orient="records"),
    }

    target = resolve_project_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def main() -> None:
    """CLI entry point for exporting React dashboard data."""
    print(export_frontend_data())


if __name__ == "__main__":
    main()
