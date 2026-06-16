import streamlit as st


st.set_page_config(
    page_title="StuntGuard Jabar ML",
    page_icon="bar_chart",
    layout="wide",
)

st.title("StuntGuard Jabar ML")
st.subheader("Klasifikasi risiko stunting kabupaten/kota di Jawa Barat")

st.write(
    "StuntGuard Jabar ML adalah project machine learning untuk membantu "
    "analisis dan prediksi risiko stunting pada wilayah kabupaten/kota di "
    "Provinsi Jawa Barat. Versi awal ini menyiapkan struktur project dan "
    "entry point dashboard Streamlit."
)

st.markdown("### Tujuan")
st.markdown(
    """
- Mengolah dataset stunting Jawa Barat tahun 2014-2024.
- Membuat label risiko stunting: rendah, sedang, dan tinggi.
- Menyiapkan pipeline eksperimen model machine learning.
- Menampilkan hasil analisis melalui dashboard Streamlit.
"""
)

st.markdown("### Dataset")
st.markdown(
    """
- Persentase balita stunting berdasarkan kabupaten/kota di Jawa Barat.
- Jumlah balita stunting berdasarkan kabupaten/kota di Jawa Barat.
"""
)

st.markdown("### SDG Terkait")
st.markdown(
    """
- SDG 2: Zero Hunger
- SDG 3: Good Health and Well-Being
"""
)

st.info("Model dan dataset belum dibuat pada tahap setup awal ini.")
