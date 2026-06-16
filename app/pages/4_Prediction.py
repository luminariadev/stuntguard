from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_utils import (
    DEFAULT_FEATURE_COLS,
    REGION_COL,
    STUNTING_COL,
    available_feature_cols,
    load_dataset,
    load_model,
    page_setup,
    project_relative,
    show_missing_model_message,
)


page_setup("Prediction")
st.title("Prediction")

df, _, _ = load_dataset()
model, model_path, model_error = load_model()

if model is None:
    show_missing_model_message(model_error)

feature_cols = available_feature_cols(df) if df is not None else DEFAULT_FEATURE_COLS

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    if df is not None and REGION_COL in df.columns:
        regions = sorted(df[REGION_COL].dropna().unique().tolist())
        region = col1.selectbox("Kabupaten/Kota", regions)
    else:
        region = col1.text_input("Kabupaten/Kota", "Kabupaten Bandung")

    year_default = 2024
    if df is not None and "tahun" in df.columns and not df.empty:
        year_default = int(df["tahun"].max())
    tahun = col2.number_input("Tahun", min_value=2014, max_value=2035, value=year_default)

    inputs = {"tahun": tahun}
    for col in feature_cols:
        if col == "tahun":
            continue
        default_value = 0.0
        if df is not None and col in df.columns:
            default_value = float(df[col].median())
        inputs[col] = st.number_input(col, value=default_value)

    submitted = st.form_submit_button("Prediksi Risiko", disabled=model is None)

if submitted and model is not None:
    input_df = pd.DataFrame([{col: inputs.get(col, 0) for col in feature_cols}])
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Prediksi Risiko: {prediction}")
        st.caption(
            f"Wilayah: {region} | Tahun: {tahun} | "
            f"Model: `{project_relative(model_path)}`"
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                classifier = model.named_steps.get("classifier")
                classes = getattr(classifier, "classes_", None)
                if classes is None and hasattr(classifier, "label_encoder"):
                    classes = classifier.label_encoder.classes_
            if classes is not None:
                prob_df = pd.DataFrame(
                    {"risk_label": classes, "probability": probabilities}
                )
                st.dataframe(prob_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediksi gagal: {exc}")

if df is not None and STUNTING_COL in df.columns:
    st.markdown("### Referensi Data Terakhir")
    reference_cols = [
        col for col in [REGION_COL, "tahun", STUNTING_COL] if col in df.columns
    ]
    if reference_cols:
        st.dataframe(df[reference_cols].tail(10), use_container_width=True)
