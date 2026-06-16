from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_utils import (
    FIGURES_DIR,
    TABLES_DIR,
    load_csv,
    page_setup,
)


page_setup("Model Evaluation")
st.title("Model Evaluation")

model_comparison_path = TABLES_DIR / "model_comparison.csv"
classification_report_path = TABLES_DIR / "classification_report.csv"
feature_importance_table_path = TABLES_DIR / "feature_importance.csv"
confusion_matrix_path = FIGURES_DIR / "confusion_matrix.png"
feature_importance_path = FIGURES_DIR / "feature_importance.png"

model_comparison, model_comparison_error = load_csv(str(model_comparison_path))
classification_report, classification_report_error = load_csv(
    str(classification_report_path)
)
feature_importance, feature_importance_error = load_csv(
    str(feature_importance_table_path)
)

if model_comparison is None and classification_report is None:
    st.warning("Hasil evaluasi belum tersedia.")
    st.caption(
        "Setelah menjalankan pipeline evaluasi, tabel dan gambar akan dibaca dari "
        "`reports/tables/` dan `reports/figures/`."
    )

for error in [
    model_comparison_error,
    classification_report_error,
    feature_importance_error,
]:
    if error:
        st.caption(error)

if model_comparison is not None:
    st.markdown("### Perbandingan Model")
    st.dataframe(model_comparison, use_container_width=True)

if classification_report is not None:
    st.markdown("### Classification Report")
    st.dataframe(classification_report, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Confusion Matrix")
    if confusion_matrix_path.exists():
        st.image(str(confusion_matrix_path))
    else:
        st.info("`reports/figures/confusion_matrix.png` belum tersedia.")

with col2:
    st.markdown("### Feature Importance")
    if feature_importance_path.exists():
        st.image(str(feature_importance_path))
    else:
        st.info("`reports/figures/feature_importance.png` belum tersedia.")

if feature_importance is not None:
    st.markdown("### Tabel Feature Importance")
    st.dataframe(feature_importance, use_container_width=True)
