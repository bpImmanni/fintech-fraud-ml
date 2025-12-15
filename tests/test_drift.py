import os
import pandas as pd
from src.drift import drift_report

BASELINE = "data/processed/train_baseline_sample.csv"

def test_drift_baseline_exists():
    assert os.path.exists(BASELINE)

def test_drift_report_runs():
    train = pd.read_csv(BASELINE).head(1000)
    live = train.sample(200, random_state=42)

    rep = drift_report(train, live, top_n=5)
    assert "feature" in rep.columns
    assert "psi" in rep.columns
