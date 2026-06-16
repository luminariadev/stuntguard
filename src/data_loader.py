"""Data loading utilities for StuntGuard Jabar ML."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.utils import resolve_project_path
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import resolve_project_path


def load_csv(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV file from a project-relative or absolute path."""
    file_path = resolve_project_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path, **read_csv_kwargs)


def save_csv(df: pd.DataFrame, path: str | Path, **to_csv_kwargs) -> Path:
    """Save a dataframe to a project-relative or absolute CSV path."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, **to_csv_kwargs)
    return output_path
