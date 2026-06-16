"""Feature engineering utilities for stunting data."""

from __future__ import annotations

import pandas as pd


DEFAULT_GROUP_COL = "nama_kabupaten_kota"
DEFAULT_TARGET_COL = "persentase_stunting"
YEAR_COL = "tahun"


def _sort_by_group_and_year(
    df: pd.DataFrame, group_col: str = DEFAULT_GROUP_COL
) -> pd.DataFrame:
    """Return a copy sorted by region and year when those columns exist."""
    sort_cols = [col for col in [group_col, YEAR_COL] if col in df.columns]
    if not sort_cols:
        return df.copy()
    return df.sort_values(sort_cols).copy()


def _validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Missing required column(s): {missing}")


def add_lag_feature(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    lag: int = 1,
    output_col: str = "lag_1_stunting",
) -> pd.DataFrame:
    """Add previous-period stunting value for each region."""
    _validate_columns(df, [group_col, target_col])

    result = _sort_by_group_and_year(df, group_col)
    result[output_col] = result.groupby(group_col, dropna=False)[target_col].shift(lag)
    return result


def add_rolling_mean(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    window: int = 3,
    output_col: str = "rolling_mean_3y",
) -> pd.DataFrame:
    """Add rolling average from previous observations for each region."""
    _validate_columns(df, [group_col, target_col])
    if window < 1:
        raise ValueError("window must be at least 1")

    result = _sort_by_group_and_year(df, group_col)
    shifted_target = result.groupby(group_col, dropna=False)[target_col].shift(1)
    result[output_col] = (
        shifted_target.groupby(result[group_col], dropna=False)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return result


def add_trend_feature(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    output_col: str = "trend_stunting",
) -> pd.DataFrame:
    """Add year-over-year change in stunting value for each region."""
    _validate_columns(df, [group_col, target_col])

    result = _sort_by_group_and_year(df, group_col)
    result[output_col] = result.groupby(group_col, dropna=False)[target_col].diff()
    return result


def build_features(
    df: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COL,
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    """Build baseline historical features used by the ML pipeline."""
    _validate_columns(df, [group_col, target_col])

    result = add_lag_feature(df, group_col, target_col)
    result = add_rolling_mean(result, group_col, target_col, window=3)
    result = add_trend_feature(result, group_col, target_col)
    return result
