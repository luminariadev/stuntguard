from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_utils import (
    REGION_COL,
    load_dataset,
    page_setup,
    project_relative,
    show_missing_data_message,
)


page_setup("Dataset Explorer")
st.title("Dataset Explorer")

df, data_path, data_error = load_dataset()
if df is None:
    show_missing_data_message(data_error)
    st.stop()

st.caption(f"Dataset: `{project_relative(data_path)}`")
filtered = df.copy()

filter_col1, filter_col2 = st.columns(2)
if "tahun" in filtered.columns:
    years = sorted(filtered["tahun"].dropna().unique().tolist())
    selected_years = filter_col1.multiselect("Tahun", years, default=years)
    if selected_years:
        filtered = filtered[filtered["tahun"].isin(selected_years)]

if REGION_COL in filtered.columns:
    regions = sorted(filtered[REGION_COL].dropna().unique().tolist())
    selected_regions = filter_col2.multiselect("Kabupaten/Kota", regions)
    if selected_regions:
        filtered = filtered[filtered[REGION_COL].isin(selected_regions)]

st.metric("Data Terfilter", f"{len(filtered):,}")
st.dataframe(filtered, use_container_width=True, height=420)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="stunting_jabar_filtered.csv",
    mime="text/csv",
)
