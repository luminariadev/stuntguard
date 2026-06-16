import streamlit as st


st.set_page_config(
    page_title="StuntGuard Jabar ML",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("StuntGuard Jabar ML")
st.subheader("Klasifikasi risiko stunting kabupaten/kota di Jawa Barat")

st.write(
    "Dashboard ini menyiapkan alur analisis stunting Jawa Barat dari dataset "
    "kabupaten/kota, preprocessing, training model klasifikasi risiko, hingga "
    "evaluasi dan prediksi."
)

col1, col2, col3 = st.columns(3)
col1.metric("Wilayah", "Jawa Barat")
col2.metric("Level Data", "Kab/Kota")
col3.metric("Rentang", "2014-2024")

st.markdown("### Tujuan")
st.markdown(
    """
- Mengolah dataset stunting Jawa Barat.
- Membuat label risiko: Rendah, Sedang, dan Tinggi.
- Membandingkan Decision Tree, Logistic Regression, Random Forest, dan XGBoost.
- Menyajikan hasil analisis melalui dashboard Streamlit.
"""
)

st.markdown("### Alur Sistem")
st.markdown(
    """
1. Dataset mentah disimpan di `data/raw/`.
2. Pipeline preprocessing membuat fitur historis dan `risk_label`.
3. Pipeline training memilih model terbaik berdasarkan Macro F1-Score.
4. Dashboard membaca dataset, model, dan laporan evaluasi dari folder project.
"""
)

st.markdown("### SDG Terkait")
sdg1, sdg2 = st.columns(2)
sdg1.info("SDG 2: Zero Hunger")
sdg2.info("SDG 3: Good Health and Well-Being")

st.caption(
    "Aplikasi tetap dapat dibuka meskipun dataset processed, model, atau laporan "
    "evaluasi belum tersedia."
)
