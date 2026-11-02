import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from modeling import MODEL_FILES, calculate_metrics, get_positive_scores


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
METRICS_PATH = PROJECT_DIR / "metrics.json"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def evaluate_model(model, features, actual):
    predictions = model.predict(features)
    scores = get_positive_scores(model, features)
    return calculate_metrics(actual, predictions, scores), predictions


def main():
    st.set_page_config(page_title="Smoke Detection Model Lab",
                       page_icon="SD", layout="wide")
    st.title("Smoke Detection Model Lab")
    st.caption("Compare five trained classifiers on labeled sensor readings.")

    if not METRICS_PATH.exists():
        st.error("Model metadata is missing. Run `python train_models.py` first.")
        st.stop()

    metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]

    source = st.radio(
        "Evaluation data",
        ["Bundled test split", "Upload a CSV"],
        horizontal=True,
    )
    if source == "Bundled test split":
        test_path = PROJECT_DIR / "test_data.csv"
        if not test_path.exists():
            st.error("Bundled test_data.csv is missing.")
            st.stop()
        test_data = pd.read_csv(test_path)
    else:
        uploaded_file = st.file_uploader(
            "Upload labeled test data (CSV)", type=["csv"])
        if uploaded_file is None:
            st.info(
                "Choose a CSV containing all 13 sensor features and the Fire Alarm target.")
            st.stop()
        try:
            test_data = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Could not read the CSV: {error}")
            st.stop()

    required_columns = feature_columns + [target_column]
    missing_columns = [
        column for column in required_columns if column not in test_data.columns]
    if missing_columns:
        st.error(
            f"CSV is missing required columns: {', '.join(missing_columns)}")
        st.stop()
    if test_data.empty:
        st.error("The CSV contains no rows.")
        st.stop()

    features = test_data[feature_columns]
    actual = test_data[target_column]
    if features.isna().any().any() or actual.isna().any():
        st.error(
            "The required feature and target columns cannot contain missing values.")
        st.stop()
    if set(actual.unique()) != {0, 1}:
        st.error("The Fire Alarm target must contain both labels 0 and 1.")
        st.stop()

    model_names = list(MODEL_FILES)
    selected_models = st.multiselect(
        "Models to compare", model_names, default=model_names
    )
    if not selected_models:
        st.info("Select at least one model to evaluate.")
        st.stop()

    metrics_by_model = {}
    predictions_by_model = {}
    model_errors = []
    for model_name in selected_models:
        model_path = MODEL_DIR / MODEL_FILES[model_name]
        if not model_path.exists():
            model_errors.append(f"{model_name}: model file not found")
            continue
        try:
            model = load_model(str(model_path))
            metrics, predictions = evaluate_model(model, features, actual)
            metrics_by_model[model_name] = metrics
            predictions_by_model[model_name] = predictions
        except Exception as error:
            model_errors.append(f"{model_name}: {error}")

    for error in model_errors:
        st.error(error)
    if not metrics_by_model:
        st.stop()

    st.subheader("Evaluation metrics")
    comparison = pd.DataFrame.from_dict(metrics_by_model, orient="index")
    st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

    detail_model = st.selectbox(
        "Confusion matrix and report", list(metrics_by_model))
    predictions = predictions_by_model[detail_model]
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{detail_model}: confusion matrix**")
        matrix = confusion_matrix(actual, predictions, labels=[0, 1])
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual: No alarm", "Actual: Fire alarm"],
                columns=["Predicted: No alarm", "Predicted: Fire alarm"],
            ),
            use_container_width=True,
        )
    with right:
        st.markdown(f"**{detail_model}: classification report**")
        report = classification_report(
            actual,
            predictions,
            labels=[0, 1],
            target_names=["No alarm", "Fire alarm"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4),
                     use_container_width=True)


if __name__ == "__main__":
    main()
