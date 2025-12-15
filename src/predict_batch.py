import os
import joblib
import pandas as pd

MODEL_PATH = "models/fraud_model.pkl"

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run: python src/train.py")

    pipe = joblib.load(MODEL_PATH)

    # Example: predict on the same dataset (demo)
    df = pd.read_csv("data/raw/creditcard.csv")
    X = df.drop(columns=["Class"])

    proba = pipe.predict_proba(X)[:, 1]
    out = X.copy()
    out["fraud_probability"] = proba
    out["risk_band"] = pd.cut(
        out["fraud_probability"],
        bins=[-0.01, 0.2, 0.7, 1.01],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/predictions.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
