from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_utils import (
    REGION_COL,
    STUNTING_COL,
    TARGET_COL,
    load_dataset,
    metric_panel,
    page_setup,
    project_relative,
    show_missing_data_message,
)


page_setup("Overview")
st.title("Overview")

df, data_path, data_error = load_dataset()
if df is None:
    show_missing_data_message(data_error)
    st.stop()

st.caption(f"Dataset: `{project_relative(data_path)}`")

total_rows = len(df)
total_regions = df[REGION_COL].nunique() if REGION_COL in df.columns else 0
year_min = int(df["tahun"].min()) if "tahun" in df.columns and not df.empty else "-"
year_max = int(df["tahun"].max()) if "tahun" in df.columns and not df.empty else "-"
avg_stunting = (
    df[STUNTING_COL].mean() if STUNTING_COL in df.columns and not df.empty else None
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_panel("Jumlah Data", f"{total_rows:,}")
with col2:
    metric_panel("Kabupaten/Kota", f"{total_regions:,}")
with col3:
    metric_panel("Rentang Tahun", f"{year_min} - {year_max}")
with col4:
    metric_panel(
        "Rata-rata Stunting",
        "-" if avg_stunting is None or pd.isna(avg_stunting) else f"{avg_stunting:.2f}%",
    )

if "tahun" in df.columns and STUNTING_COL in df.columns and not df.empty:
    st.markdown("### Tren Tahunan")
    trend = (
        df.groupby("tahun", as_index=False)[STUNTING_COL]
        .mean()
        .sort_values("tahun")
        .set_index("tahun")
    )
    st.line_chart(trend)

if TARGET_COL in df.columns:
    st.markdown("### Distribusi Risiko")
    risk_counts = df[TARGET_COL].value_counts().rename_axis("risk_label").reset_index(
        name="jumlah"
    )
    st.bar_chart(risk_counts, x="risk_label", y="jumlah")

st.markdown("### Sampel Dataset")
st.dataframe(df.head(20), use_container_width=True)
