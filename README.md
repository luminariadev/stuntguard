# StuntGuard Jabar ML

StuntGuard Jabar ML adalah project machine learning untuk klasifikasi risiko stunting kabupaten/kota di Provinsi Jawa Barat. Project ini menggunakan data Open Data Jabar, membuat fitur historis stunting, melatih beberapa model klasifikasi, dan menyajikan hasilnya lewat dashboard Streamlit.

## Features

- Fetch dataset resmi dari Open Data Jabar
- Preprocessing dan feature engineering otomatis
- Label risiko berbasis kuartil: `Rendah`, `Sedang`, `Tinggi`
- Training Decision Tree, Logistic Regression, Random Forest, dan XGBoost
- Evaluasi accuracy, precision, recall, F1-score, confusion matrix, dan feature importance
- Dashboard Streamlit multipage

## Tech Stack

- Python
- pandas, scikit-learn, XGBoost
- Streamlit
- Matplotlib, Plotly, Seaborn
- pytest

## Quickstart

Clone repository:

```bash
git clone https://github.com/luminariadev/stuntguard.git
cd stuntguard
```

Buat dan aktifkan virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download dataset:

```bash
python -m src.fetch_data
```

Jalankan pipeline end-to-end:

```bash
python -m src.pipeline
```

Jalankan dashboard:

```bash
streamlit run app/Home.py
```

Buka:

```text
http://localhost:8501
```

## Output Pipeline

Pipeline akan membuat file berikut:

```text
data/raw/persentase_stunting_jabar.csv
data/raw/jumlah_stunting_jabar.csv
data/processed/stunting_jabar_clean.csv
data/processed/stunting_jabar_features.csv
models/best_model.joblib
reports/tables/model_comparison.csv
reports/tables/classification_report.csv
reports/tables/feature_importance.csv
reports/figures/confusion_matrix.png
reports/figures/feature_importance.png
```

## Manual Prediction

```python
import pandas as pd
from src.predict import load_model, predict_risk, predict_probability

model = load_model("models/best_model.joblib")

sample = pd.DataFrame([{
    "tahun": 2024,
    "lag_1_stunting": 8.11,
    "rolling_mean_3y": 12.07,
    "trend_stunting": 8.97,
}])

print(predict_risk(model, sample))
print(predict_probability(model, sample))
```

## Project Structure

```text
app/                Streamlit dashboard
data/raw/           Raw datasets
data/processed/     Processed datasets
models/             Trained model artifacts
reports/            Evaluation tables and figures
src/                Reusable ML pipeline code
tests/              Unit tests
```

## Dashboard Pages

- Home
- Overview
- Dataset Explorer
- Model Training
- Prediction
- Model Evaluation

## Data Source

- [Persentase Balita Stunting Berdasarkan Kabupaten/Kota di Jawa Barat](https://opendata.jabarprov.go.id/id/dataset/persentase-balita-stunting-berdasarkan-kabupatenkota-di-jawa-barat)
- [Jumlah Balita Stunting Berdasarkan Kabupaten/Kota di Jawa Barat](https://opendata.jabarprov.go.id/id/dataset/jumlah-balita-stunting-berdasarkan-kabupatenkota-di-jawa-barat)

## Test

```bash
pytest
```

## License

MIT License
