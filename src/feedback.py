import sqlite3
from datetime import datetime

DB_PATH = "data/processed/feedback.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT,
      row_index INTEGER,
      fraud_probability REAL,
      risk_band TEXT,
      decision TEXT
    )
    """)
    con.commit()
    con.close()

def write_feedback(row_index: int, prob: float, risk: str, decision: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO feedback(ts,row_index,fraud_probability,risk_band,decision) VALUES (?,?,?,?,?)",
        (datetime.utcnow().isoformat(), row_index, float(prob), str(risk), str(decision))
    )
    con.commit()
    con.close()
