import os
import sqlite3
from datetime import datetime
from pathlib import Path

# Default location inside repo
DEFAULT_DB = Path("data") / "processed" / "feedback.db"
DB_PATH = Path(os.environ.get("FEEDBACK_DB_PATH", str(DEFAULT_DB)))

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            row_index INTEGER,
            fraud_probability REAL,
            risk_band TEXT,
            action TEXT
        )
    """)
    con.commit()
    con.close()

def write_feedback(row_index: int, fraud_probability: float, risk_band: str, action: str):
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO feedback (created_at, row_index, fraud_probability, risk_band, action)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), row_index, fraud_probability, risk_band, action))
    con.commit()
    con.close()
