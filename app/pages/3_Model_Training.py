from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_utils import (
    TARGET_COL,
    available_feature_cols,
    load_dataset,
    page_setup,
    project_relative,
    show_missing_data_message,
)


page_setup("Model Training")
st.title("Model Training")

df, data_path, data_error = load_dataset()
if df is None:
    show_missing_data_message(data_error)
    st.stop()

feature_cols = available_feature_cols(df)
if TARGET_COL not in df.columns or not feature_cols:
    st.warning("Dataset belum memiliki fitur training lengkap.")
    st.caption(
        "Pastikan dataset memiliki `risk_label` dan minimal satu fitur seperti "
        "`tahun`, `lag_1_stunting`, `rolling_mean_3y`, atau `trend_stunting`."
    )
    st.stop()

st.caption(f"Dataset: `{project_relative(data_path)}`")

selected_models = st.multiselect(
    "Model",
    ["Decision Tree", "Logistic Regression", "Random Forest", "XGBoost"],
    default=["Decision Tree", "Logistic Regression", "Random Forest", "XGBoost"],
)
st.write("Fitur yang digunakan:")
st.code(", ".join(feature_cols))

if st.button("Train Model", type="primary", disabled=not selected_models):
    with st.spinner("Melatih model..."):
        try:
            from src.train import (
                select_best_model,
                save_model,
                split_data,
                train_decision_tree,
                train_logistic_regression,
                train_random_forest,
                train_xgboost,
            )
            from src.evaluate import evaluate_and_save, save_evaluation_result
            from src.utils import resolve_project_path

            X_train, X_test, y_train, y_test = split_data(
                df, feature_cols, TARGET_COL
            )
            trained_models = {}
            if "Decision Tree" in selected_models:
                trained_models["decision_tree"] = train_decision_tree(X_train, y_train)
            if "Logistic Regression" in selected_models:
                trained_models["logistic_regression"] = train_logistic_regression(
                    X_train, y_train
                )
            if "Random Forest" in selected_models:
                trained_models["random_forest"] = train_random_forest(X_train, y_train)
            if "XGBoost" in selected_models:
                try:
                    trained_models["xgboost"] = train_xgboost(X_train, y_train)
                except ImportError as exc:
                    trained_models["xgboost"] = None
                    st.warning(str(exc))

            best_name, best_model, results = select_best_model(
                trained_models, X_test, y_test
            )
            model_path = save_model(best_model, "models/best_model.joblib")
            reports_dir = resolve_project_path("reports")
            save_evaluation_result(
                results, reports_dir / "tables" / "model_comparison.csv"
            )
            evaluate_and_save(best_model, X_test, y_test, feature_cols, reports_dir)

            st.success(f"Model terbaik: {best_name}")
            st.caption(f"Model disimpan ke `{model_path}`")
            st.caption("Laporan evaluasi disimpan ke `reports/`.")
            st.dataframe(results, use_container_width=True)
        except Exception as exc:
            st.error(f"Training gagal: {exc}")
else:
    st.info("Training akan menyimpan model terbaik ke `models/best_model.joblib`.")
