"""Shared helper utilities for StuntGuard Jabar ML."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Return an absolute path rooted at the project directory.

    Absolute paths are returned unchanged. Relative paths are resolved against
    the repository root so scripts behave the same from notebooks, Streamlit,
    tests, or the command line.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
