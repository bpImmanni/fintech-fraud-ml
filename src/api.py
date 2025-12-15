import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

MODEL_PATH = "models/fraud_model.pkl"

app = FastAPI(title="Fraud Scoring API")
pipe = joblib.load(MODEL_PATH)

class Transaction(BaseModel):
    features: Dict[str, Any]  # column_name -> value

@app.post("/score")
def score(tx: Transaction):
    df = pd.DataFrame([tx.features])
    proba = float(pipe.predict_proba(df)[:, 1][0])
    risk = "LOW" if proba < 0.2 else ("MEDIUM" if proba < 0.7 else "HIGH")
    return {"fraud_probability": proba, "risk_band": risk}
