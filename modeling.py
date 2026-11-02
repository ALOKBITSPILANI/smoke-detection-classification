from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        ),
        "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), LinearSVC(max_iter=10000, random_state=42)
        ),
    }


def get_positive_scores(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def calculate_metrics(actual, predicted, positive_scores):
    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "AUC": float(roc_auc_score(actual, positive_scores)),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(actual, predicted)),
    }
