from fastapi.testclient import TestClient
import pandas as pd
from src.api import app

client = TestClient(app)

def test_score_endpoint():
    df = pd.read_csv("data/raw/creditcard.csv").drop(columns=["Class"]).head(3)
    payload = df.to_dict(orient="records")

    resp = client.post("/score", json=payload)

    # If it fails, show FastAPI validation error (super useful)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3
