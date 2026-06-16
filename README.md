# StuntGuard Jabar ML

**StuntGuard Jabar ML** adalah project machine learning untuk melakukan klasifikasi risiko stunting pada wilayah kabupaten/kota di Provinsi Jawa Barat. Project ini menggunakan data stunting tahun 2014–2024 dan dikembangkan sebagai model prediksi serta prototype dashboard berbasis Streamlit.

Project ini dibuat untuk mendukung analisis berbasis data terhadap isu stunting di Indonesia, khususnya Jawa Barat, serta selaras dengan Sustainable Development Goals (SDGs), terutama:

- **SDG 2: Zero Hunger**
- **SDG 3: Good Health and Well-Being**

## 1. Tujuan Project

Tujuan utama project ini adalah membangun sistem sederhana yang dapat:

1. Mengolah dataset stunting kabupaten/kota di Jawa Barat.
2. Membuat label risiko stunting wilayah: rendah, sedang, dan tinggi.
3. Melatih model machine learning untuk klasifikasi risiko stunting.
4. Membandingkan performa beberapa algoritma machine learning.
5. Menampilkan hasil analisis dan prediksi melalui dashboard Streamlit.
6. Menyediakan struktur project yang rapi untuk kebutuhan portfolio, paper, dan pengembangan lanjutan.

## 2. Scope Project

Scope awal project:

| Komponen       | Detail                                                                 |
| -------------- | ---------------------------------------------------------------------- |
| Wilayah        | Provinsi Jawa Barat                                                    |
| Level Data     | Kabupaten/Kota                                                         |
| Rentang Tahun  | 2014–2024                                                              |
| Task ML        | Klasifikasi risiko stunting                                            |
| Model Utama    | Random Forest dan XGBoost                                              |
| Baseline Model | Decision Tree dan Logistic Regression                                  |
| Prototype      | Streamlit Dashboard                                                    |
| Output Akhir   | Model `.pkl` / `.joblib`, dashboard, laporan evaluasi, dan dokumentasi |

## 3. Sumber Dataset

Dataset utama yang digunakan:

1. **Persentase Balita Stunting Berdasarkan Kabupaten/Kota di Jawa Barat**
   - Sumber: Open Data Jabar
   - Rentang: 2014–2024
   - Fungsi: target utama untuk klasifikasi risiko

2. **Jumlah Balita Stunting Berdasarkan Kabupaten/Kota di Jawa Barat**
   - Sumber: Open Data Jabar
   - Rentang: 2014–2024
   - Fungsi: fitur tambahan dan analisis jumlah kasus

Dataset tambahan opsional:

1. **Indeks Khusus Penanganan Stunting (IKPS)**
   - Sumber: BPS
   - Fungsi: indikator intervensi stunting

2. **Indikator sosial-ekonomi dan lingkungan**
   - Kemiskinan
   - IPM
   - Sanitasi layak
   - Akses air minum layak
   - Rata-rata lama sekolah

Untuk versi pertama, fokus pada dataset stunting utama terlebih dahulu. Dataset tambahan dapat ditambahkan setelah pipeline awal stabil.

## 4. Problem Statement

Stunting merupakan salah satu masalah kesehatan publik yang berdampak pada kualitas sumber daya manusia. Data stunting sering tersedia dalam bentuk laporan statistik, tetapi belum selalu dimanfaatkan untuk membangun sistem prediksi risiko wilayah.

Project ini mencoba menjawab pertanyaan:

1. Bagaimana membangun model machine learning untuk mengklasifikasikan risiko stunting kabupaten/kota di Jawa Barat?
2. Algoritma mana yang memberikan performa terbaik untuk klasifikasi risiko stunting?
3. Bagaimana hasil model dapat disajikan dalam bentuk dashboard yang mudah dipahami?

## 5. Output Model

Model menghasilkan output berupa kelas risiko stunting:

| Label  | Keterangan                                                          |
| ------ | ------------------------------------------------------------------- |
| Rendah | Wilayah dengan risiko stunting relatif rendah                       |
| Sedang | Wilayah dengan risiko stunting menengah                             |
| Tinggi | Wilayah dengan risiko stunting tinggi dan perlu prioritas perhatian |

Contoh output prediksi:

```text
Wilayah: Kabupaten Bandung
Tahun: 2024
Prediksi Risiko: Sedang
Probabilitas: 0.67
Model: Random Forest
```

## 6. Strategi Label Risiko

Label risiko dibuat menggunakan pendekatan kuartil terhadap nilai persentase stunting.

Contoh aturan:

```text
Persentase <= Q1       : Rendah
Q1 < Persentase <= Q3  : Sedang
Persentase > Q3        : Tinggi
```

Alasan menggunakan kuartil:

1. Lebih adaptif terhadap distribusi data.
2. Mengurangi risiko class imbalance.
3. Cocok untuk dataset wilayah dengan jumlah data terbatas.

## 7. Fitur Model

Fitur awal yang digunakan:

| Fitur                 | Keterangan                                        |
| --------------------- | ------------------------------------------------- |
| `tahun`               | Tahun observasi                                   |
| `kode_kabupaten_kota` | Kode wilayah jika tersedia                        |
| `nama_kabupaten_kota` | Nama wilayah                                      |
| `persentase_stunting` | Persentase balita stunting                        |
| `jumlah_stunting`     | Jumlah balita stunting                            |
| `lag_1_stunting`      | Nilai persentase stunting tahun sebelumnya        |
| `rolling_mean_3y`     | Rata-rata persentase stunting tiga tahun terakhir |
| `trend_stunting`      | Perubahan nilai stunting dari tahun sebelumnya    |
| `risk_label`          | Label risiko hasil preprocessing                  |

Catatan:

- `risk_label` adalah target klasifikasi.
- `persentase_stunting` dapat digunakan untuk proses labeling, tetapi perlu hati-hati agar tidak terjadi data leakage.
- Untuk eksperimen utama, gunakan fitur historis seperti `lag_1_stunting`, `rolling_mean_3y`, dan `trend_stunting`.
- Jika dataset tambahan sudah tersedia, tambahkan fitur sosial-ekonomi dan lingkungan.

## 8. Model Machine Learning

Model yang wajib dibuat:

1. Decision Tree Classifier
2. Logistic Regression
3. Random Forest Classifier
4. XGBoost Classifier

Model utama:

1. Random Forest
2. XGBoost

Random Forest digunakan karena kuat untuk data tabular dan relatif mudah diinterpretasikan. XGBoost digunakan sebagai pembanding karena sering memberikan performa tinggi pada data tabular.

## 9. Evaluasi Model

Metrik evaluasi:

| Metrik           | Fungsi                                            |
| ---------------- | ------------------------------------------------- |
| Accuracy         | Mengukur akurasi keseluruhan                      |
| Precision        | Mengukur ketepatan prediksi tiap kelas            |
| Recall           | Mengukur kemampuan model menangkap kelas tertentu |
| F1-Score         | Metrik utama untuk klasifikasi                    |
| Macro F1-Score   | Cocok untuk kelas tidak seimbang                  |
| Confusion Matrix | Melihat pola kesalahan model                      |

Metrik utama yang harus diperhatikan:

```text
Macro F1-Score
Recall untuk kelas Tinggi
```

Alasan:

Dalam kasus stunting, kesalahan memprediksi wilayah risiko tinggi sebagai rendah lebih berbahaya dibanding salah memprediksi wilayah rendah sebagai sedang.

## 10. Struktur Project

Gunakan struktur folder berikut:

```text
stuntguard-jabar-ml/
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Dataset_Explorer.py
│       ├── 3_Model_Training.py
│       ├── 4_Prediction.py
│       └── 5_Model_Evaluation.py
│
├── data/
│   ├── raw/
│   │   ├── persentase_stunting_jabar.csv
│   │   └── jumlah_stunting_jabar.csv
│   ├── processed/
│   │   ├── stunting_jabar_clean.csv
│   │   └── stunting_jabar_features.csv
│   └── external/
│
├── models/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── stunting_trend.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   ├── tables/
│   │   └── model_comparison.csv
│   └── paper_notes.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

## 11. Penjelasan Folder

### `app/`

Berisi aplikasi Streamlit.

Halaman yang perlu dibuat:

1. `Home.py`
   - Ringkasan project
   - Tujuan
   - Dataset
   - SDG terkait

2. `pages/1_Overview.py`
   - Statistik umum dataset
   - Jumlah wilayah
   - Rentang tahun
   - Ringkasan distribusi stunting

3. `pages/2_Dataset_Explorer.py`
   - Tabel dataset
   - Filter tahun
   - Filter kabupaten/kota
   - Download dataset processed

4. `pages/3_Model_Training.py`
   - Pilih model
   - Tombol training
   - Tampilkan metrik hasil training

5. `pages/4_Prediction.py`
   - Input wilayah dan tahun
   - Input fitur numerik jika diperlukan
   - Tampilkan prediksi risiko

6. `pages/5_Model_Evaluation.py`
   - Tabel perbandingan model
   - Confusion matrix
   - Classification report
   - Feature importance

### `data/`

Berisi data mentah dan data hasil preprocessing.

Jangan hapus data mentah agar proses bisa direproduksi.

### `models/`

Berisi model hasil training.

Format model:

```text
.pkl
.joblib
```

### `notebooks/`

Berisi notebook eksperimen.

Notebook harus diberi nomor agar alurnya jelas.

### `src/`

Berisi kode utama yang reusable.

Semua fungsi penting harus dipindahkan dari notebook ke folder `src/`.

### `reports/`

Berisi gambar, tabel, dan catatan untuk paper.

### `tests/`

Berisi unit test sederhana.

## 12. Instalasi

Clone repository:

```bash
git clone https://github.com/username/stuntguard-jabar-ml.git
cd stuntguard-jabar-ml
```

Buat virtual environment:

```bash
python -m venv venv
```

Aktifkan virtual environment.

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 13. Requirements

Isi file `requirements.txt`:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
plotly
seaborn
streamlit
joblib
openpyxl
pytest
```

Catatan:

- Gunakan `matplotlib` atau `plotly` untuk visualisasi.
- Gunakan `joblib` untuk menyimpan model.
- Gunakan `pytest` untuk testing sederhana.

## 14. Cara Menjalankan Streamlit

Jalankan aplikasi:

```bash
streamlit run app/Home.py
```

Aplikasi akan berjalan di:

```text
http://localhost:8501
```

## 15. Data Preprocessing

File utama:

```text
src/preprocessing.py
```

Fungsi yang perlu dibuat:

```python
def clean_column_names(df):
    pass

def normalize_region_name(df):
    pass

def handle_missing_values(df):
    pass

def create_risk_label(df, target_column):
    pass

def preprocess_stunting_data(df):
    pass
```

Tahapan preprocessing:

1. Load dataset mentah.
2. Bersihkan nama kolom.
3. Samakan format nama kabupaten/kota.
4. Ubah tipe data tahun dan nilai numerik.
5. Tangani missing value.
6. Buat fitur historis:
   - `lag_1_stunting`
   - `rolling_mean_3y`
   - `trend_stunting`

7. Buat label risiko:
   - Rendah
   - Sedang
   - Tinggi

8. Simpan hasil ke `data/processed/`.

## 16. Feature Engineering

File utama:

```text
src/feature_engineering.py
```

Fitur yang perlu dibuat:

```python
def add_lag_feature(df, group_col, target_col):
    pass

def add_rolling_mean(df, group_col, target_col, window=3):
    pass

def add_trend_feature(df, group_col, target_col):
    pass

def build_features(df):
    pass
```

Fitur minimal:

```text
lag_1_stunting
rolling_mean_3y
trend_stunting
tahun
```

Target:

```text
risk_label
```

## 17. Training Model

File utama:

```text
src/train.py
```

Fungsi yang perlu dibuat:

```python
def split_data(df, feature_cols, target_col):
    pass

def train_decision_tree(X_train, y_train):
    pass

def train_logistic_regression(X_train, y_train):
    pass

def train_random_forest(X_train, y_train):
    pass

def train_xgboost(X_train, y_train):
    pass

def save_model(model, path):
    pass
```

Model yang harus dilatih:

1. Decision Tree
2. Logistic Regression
3. Random Forest
4. XGBoost

Gunakan `train_test_split` dengan `stratify=y` jika memungkinkan.

Jika ingin lebih kuat untuk paper, gunakan pembagian berbasis waktu:

```text
Training: 2014–2021
Validation: 2022
Testing: 2023–2024
```

## 18. Evaluasi Model

File utama:

```text
src/evaluate.py
```

Fungsi yang perlu dibuat:

```python
def evaluate_classification_model(model, X_test, y_test):
    pass

def generate_classification_report(y_test, y_pred):
    pass

def plot_confusion_matrix(y_test, y_pred):
    pass

def get_feature_importance(model, feature_names):
    pass

def save_evaluation_result(result, path):
    pass
```

Output evaluasi:

```text
reports/tables/model_comparison.csv
reports/figures/confusion_matrix.png
reports/figures/feature_importance.png
```

## 19. Prediction

File utama:

```text
src/predict.py
```

Fungsi yang perlu dibuat:

```python
def load_model(path):
    pass

def predict_risk(model, input_data):
    pass

def predict_probability(model, input_data):
    pass
```

Output prediksi:

```python
{
    "risk_label": "Sedang",
    "probability": 0.67,
    "model_name": "Random Forest"
}
```

## 20. Dashboard Streamlit

Dashboard harus memiliki fitur:

### Home

Menampilkan:

- Nama project
- Deskripsi singkat
- SDG terkait
- Scope project
- Alur sistem

### Overview

Menampilkan:

- Jumlah data
- Jumlah kabupaten/kota
- Rentang tahun
- Rata-rata persentase stunting
- Grafik tren tahunan

### Dataset Explorer

Menampilkan:

- Tabel dataset
- Filter tahun
- Filter wilayah
- Tombol download CSV

### Model Training

Menampilkan:

- Pilihan model
- Tombol train model
- Hasil metrik
- Simpan model

### Prediction

Menampilkan:

- Form input wilayah
- Form input tahun
- Form input fitur
- Hasil prediksi risiko
- Probabilitas prediksi

### Model Evaluation

Menampilkan:

- Tabel perbandingan model
- Confusion matrix
- Classification report
- Feature importance

## 21. Git Workflow

Gunakan branch berikut:

```text
main
develop
feature/project-setup
feature/data-preprocessing
feature/eda
feature/model-training
feature/model-evaluation
feature/streamlit-dashboard
feature/documentation
```

Aturan branch:

1. `main` hanya untuk versi stabil.
2. `develop` untuk integrasi fitur.
3. Semua fitur dibuat dari `develop`.
4. Setiap fitur selesai dibuat Pull Request ke `develop`.
5. Setelah semua fitur stabil, merge `develop` ke `main`.

Contoh workflow:

```bash
git checkout -b develop
git push -u origin develop

git checkout -b feature/project-setup
git add .
git commit -m "init project structure"
git push -u origin feature/project-setup
```

Setelah itu, buat Pull Request:

```text
feature/project-setup -> develop
```

## 22. Commit Convention

Gunakan format commit berikut:

```text
init: initial project setup
data: add raw stunting dataset
prep: clean stunting dataset
feat: add risk label generation
feat: add feature engineering pipeline
model: add random forest classifier
model: add xgboost classifier
eval: add model evaluation report
app: add streamlit home page
app: add prediction page
docs: update readme
fix: handle missing values
refactor: simplify preprocessing pipeline
```

Contoh commit:

```bash
git commit -m "prep: clean stunting dataset columns"
git commit -m "feat: add quantile based risk label"
git commit -m "model: train random forest classifier"
git commit -m "eval: add classification metrics"
git commit -m "app: add prediction dashboard"
```

## 23. Issue Template

Buat GitHub issues berikut:

### Issue 1 — Project Setup

```text
Title: Setup initial project structure

Tasks:
- Create folder structure
- Add .gitignore
- Add requirements.txt
- Add README.md
- Add initial Streamlit app entry point
```

### Issue 2 — Data Preprocessing

```text
Title: Build preprocessing pipeline

Tasks:
- Load raw stunting dataset
- Clean column names
- Normalize region names
- Handle missing values
- Create risk labels
- Save processed dataset
```

### Issue 3 — Feature Engineering

```text
Title: Add feature engineering for stunting data

Tasks:
- Add lag feature
- Add rolling mean feature
- Add trend feature
- Prepare feature matrix and target
```

### Issue 4 — Model Training

```text
Title: Train baseline and main ML models

Tasks:
- Train Decision Tree
- Train Logistic Regression
- Train Random Forest
- Train XGBoost
- Save trained models
```

### Issue 5 — Model Evaluation

```text
Title: Evaluate classification models

Tasks:
- Generate accuracy, precision, recall, F1-score
- Generate confusion matrix
- Generate model comparison table
- Generate feature importance
```

### Issue 6 — Streamlit Dashboard

```text
Title: Build Streamlit dashboard

Tasks:
- Create Home page
- Create Overview page
- Create Dataset Explorer page
- Create Model Training page
- Create Prediction page
- Create Model Evaluation page
```

### Issue 7 — Documentation

```text
Title: Complete project documentation

Tasks:
- Update README
- Add dataset explanation
- Add model explanation
- Add running instructions
- Add screenshots
- Add paper notes
```

## 24. Pull Request Template

Gunakan template PR berikut:

```text
## Summary

Describe what this pull request adds or changes.

## Changes

-
-
-

## Checklist

- [ ] Code runs without error
- [ ] Data path is correct
- [ ] Function names are clear
- [ ] README updated if needed
- [ ] Screenshots added if UI changed

## Notes

Add additional notes here.
```

## 25. Target Output Akhir

Output akhir project:

```text
1. Dataset bersih
2. Notebook EDA
3. Model Random Forest
4. Model XGBoost
5. Tabel evaluasi model
6. Confusion matrix
7. Feature importance
8. Streamlit dashboard
9. README lengkap
10. Paper notes
```

## 26. Acceptance Criteria

Project dianggap selesai jika:

1. Dataset berhasil dimuat dan dibersihkan.
2. Label risiko berhasil dibuat.
3. Minimal 4 model berhasil dilatih.
4. Random Forest dan XGBoost berhasil dibandingkan.
5. Model terbaik berhasil disimpan.
6. Dashboard Streamlit dapat dijalankan.
7. Halaman prediksi menghasilkan output risiko.
8. README menjelaskan cara install dan menjalankan project.
9. Semua fitur utama sudah masuk ke branch `main`.
10. Repository memiliki riwayat commit dan PR yang rapi.

## 27. Future Development

Pengembangan lanjutan:

1. Menambahkan dataset kemiskinan, IPM, sanitasi, dan air minum layak.
2. Menambahkan interpretasi model menggunakan SHAP.
3. Menambahkan peta interaktif wilayah Jawa Barat.
4. Membuat backend FastAPI untuk model serving.
5. Membuat frontend Astro atau Next.js.
6. Membuat aplikasi mobile Flutter.
7. Menambahkan database Supabase PostgreSQL.
8. Deploy dashboard ke Streamlit Community Cloud.

## 28. Catatan untuk Codex

Saat mengembangkan project ini, ikuti aturan berikut:

1. Jangan membuat semua fitur sekaligus.
2. Kerjakan berdasarkan branch dan issue.
3. Mulai dari struktur project.
4. Lanjutkan ke preprocessing.
5. Setelah data bersih, buat model training.
6. Setelah model stabil, buat dashboard.
7. Simpan fungsi reusable di folder `src/`.
8. Notebook hanya untuk eksplorasi dan eksperimen.
9. Hindari hardcoded path yang sulit dipindahkan.
10. Gunakan error handling sederhana pada file loader dan prediction.
11. Pastikan aplikasi Streamlit tetap bisa berjalan meskipun model belum tersedia.
12. Gunakan komentar secukupnya untuk fungsi penting.
13. Pastikan kode mudah dibaca dan cocok untuk portfolio.

## 29. License

Project ini menggunakan lisensi MIT untuk kebutuhan pembelajaran, portfolio, dan pengembangan akademik.
