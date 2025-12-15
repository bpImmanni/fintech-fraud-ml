import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Optional (Tier-3): MLflow tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False


RAW_PATH = "data/raw/creditcard.csv"
MODEL_PATH = "models/fraud_model.pkl"
METRICS_PATH = "reports/metrics.txt"

SHAP_BG_PATH = "data/processed/shap_background.csv"
DRIFT_BASELINE_PATH = "data/processed/train_baseline_sample.csv"
TRAINING_BASELINE_JSON = "reports/training_baseline.json"
SCHEMA_PATH = "reports/schema.json"


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. Put creditcard.csv there.")

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' not found.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # -----------------------------
    # Tier-3: Time-based split (no leakage)
    # If 'Time' exists, split by time: past -> future
    # Otherwise fallback to stratified random split
    # -----------------------------
    if "Time" in X.columns:
        cut = X["Time"].quantile(0.8)
        train_idx = X["Time"] <= cut
        test_idx = X["Time"] > cut

        X_train, y_train = X.loc[train_idx].copy(), y.loc[train_idx].copy()
        X_test, y_test = X.loc[test_idx].copy(), y.loc[test_idx].copy()
        split_note = "time_based_split(Time quantile=0.8)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        split_note = "random_stratified_split(test_size=0.2)"

    # -----------------------------
    # Tier-3: Save schema (for UI/API validation & column order)
    # -----------------------------
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump({"columns": list(X_train.columns)}, f, indent=2)

    # -----------------------------
    # Tier-2: Save SHAP background + drift baseline sample
    # -----------------------------
    # SHAP background: smaller sample (speed)
    X_train.sample(min(1000, len(X_train)), random_state=42).to_csv(SHAP_BG_PATH, index=False)
    # Drift baseline: larger sample for PSI stability
    X_train.sample(min(5000, len(X_train)), random_state=42).to_csv(DRIFT_BASELINE_PATH, index=False)

    # -----------------------------
    # Save baseline stats (optional, useful for monitoring)
    # -----------------------------
    baseline = {
        "split": split_note,
        "feature_means": X_train.mean(numeric_only=True).to_dict(),
        "feature_stds": X_train.std(numeric_only=True).replace(0, 1e-9).to_dict()
    }
    with open(TRAINING_BASELINE_JSON, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    # -----------------------------
    # Model
    # -----------------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=4,
        random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model)
    ])

    # -----------------------------
    # Tier-3: MLflow tracking (optional)
    # -----------------------------
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("fintech_fraud_detector")

        with mlflow.start_run():
            mlflow.log_param("split_strategy", split_note)
            mlflow.log_params({
                "model": "XGBClassifier",
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9
            })

            pipe.fit(X_train, y_train)

            proba = pipe.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)

            pr_auc = average_precision_score(y_test, proba)
            roc_auc = roc_auc_score(y_test, proba)
            report = classification_report(y_test, preds, digits=4)

            mlflow.log_metrics({"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)})
            mlflow.sklearn.log_model(pipe, "model")

    else:
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        pr_auc = average_precision_score(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        report = classification_report(y_test, preds, digits=4)

    # -----------------------------
    # Save model + metrics
    # -----------------------------
    joblib.dump(pipe, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Split: {split_note}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC:  {pr_auc:.4f}\n\n")
        f.write(report)

    print("Saved model to:", MODEL_PATH)
    print("Saved metrics to:", METRICS_PATH)
    print("Saved SHAP background to:", SHAP_BG_PATH)
    print("Saved drift baseline to:", DRIFT_BASELINE_PATH)
    print("Saved schema to:", SCHEMA_PATH)
    print("Saved training baseline to:", TRAINING_BASELINE_JSON)
    print("\nKey metrics:")
    print("Split:", split_note)
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC :", round(pr_auc, 4))

    if MLFLOW_AVAILABLE:
        print("\nMLflow is enabled. To view runs, execute: mlflow ui")


if __name__ == "__main__":
    main()
