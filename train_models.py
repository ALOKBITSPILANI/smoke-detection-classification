import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from modeling import MODEL_FILES, build_models, calculate_metrics, get_positive_scores


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "smoke_detection_iot.csv"
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
METRICS_PATH = PROJECT_DIR / "metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if data.shape[1] < 4:
        raise ValueError(
            "Expected two removable columns, at least one feature, and a target.")

    dropped_columns = list(data.columns[:2])
    target_column = data.columns[-1]
    feature_columns = list(data.columns[2:-1])
    if target_column != "Fire Alarm":
        raise ValueError(
            f"Expected 'Fire Alarm' as the last column, found {target_column!r}.")
    if data[feature_columns + [target_column]].isna().any().any():
        raise ValueError(
            "Features and target must not contain missing values.")
    if set(data[target_column].unique()) != {0, 1}:
        raise ValueError("Fire Alarm must contain the binary labels 0 and 1.")

    features = data[feature_columns]
    target = data[target_column]
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    for model_name, model in build_models().items():
        model.fit(train_features, train_target)
        predictions = model.predict(test_features)
        scores = get_positive_scores(model, test_features)
        results[model_name] = {
            **calculate_metrics(test_target, predictions, scores),
            "confusion_matrix": confusion_matrix(
                test_target, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                test_target,
                predictions,
                labels=[0, 1],
                target_names=["No alarm", "Fire alarm"],
                output_dict=True,
                zero_division=0,
            ),
        }
        joblib.dump(model, MODEL_DIR / MODEL_FILES[model_name])
        print(f"{model_name}: {results[model_name]}")

    test_data = test_features.copy()
    test_data[target_column] = test_target
    test_data.to_csv(TEST_DATA_PATH, index=False)

    metadata = {
        "dataset": DATA_PATH.name,
        "target_column": target_column,
        "dropped_columns": dropped_columns,
        "feature_columns": feature_columns,
        "rows": int(len(data)),
        "train_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": results,
    }
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved holdout data to {TEST_DATA_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
