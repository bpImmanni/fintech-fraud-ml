import json
import os
import pandas as pd

SCHEMA_PATH = "reports/schema.json"

def test_schema_file_exists():
    assert os.path.exists(SCHEMA_PATH), "reports/schema.json missing. Run: python src/train.py"

def test_schema_columns_match_data():
    schema = json.load(open(SCHEMA_PATH, "r", encoding="utf-8"))
    cols = schema["columns"]

    df = pd.read_csv("data/raw/creditcard.csv").drop(columns=["Class"])
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Schema columns missing in dataset: {missing}"
