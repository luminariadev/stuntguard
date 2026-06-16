"""Download raw StuntGuard datasets from Open Data Jabar."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

try:
    from src.utils import resolve_project_path
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import resolve_project_path


DATASETS = {
    "persentase_stunting_jabar.csv": "https://data.jabarprov.go.id/api-backend//bigdata/dinkes/od_17148_persentase_balita_stunting__kabupatenkota_v3?limit=500&skip=0",
    "jumlah_stunting_jabar.csv": "https://data.jabarprov.go.id/api-backend//bigdata/dinkes/od_17147_jumlah_balita_stunting_berdasarkan_kabupatenkota_v3?limit=500&skip=0",
}


def fetch_dataset(url: str) -> pd.DataFrame:
    """Fetch one Open Data Jabar API endpoint and return its data rows."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error") not in [0, None]:
        raise ValueError(payload.get("message", "Open Data Jabar returned an error"))
    data = payload.get("data", [])
    if not data:
        raise ValueError("Open Data Jabar response did not contain data rows")
    return pd.DataFrame(data)


def fetch_raw_datasets(output_dir: str | Path = "data/raw") -> dict[str, Path]:
    """Download all required raw datasets into the project."""
    target_dir = resolve_project_path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for filename, url in DATASETS.items():
        df = fetch_dataset(url)
        output_path = target_dir / filename
        df.to_csv(output_path, index=False)
        saved_paths[filename] = output_path
    return saved_paths


def main() -> None:
    """CLI entry point for downloading raw datasets."""
    saved_paths = fetch_raw_datasets()
    for filename, path in saved_paths.items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()
