import os
import sqlite3
from datetime import datetime

# Render-safe writable path (ephemeral but works)
DEFAULT_DB = "/tmp/feedback.db"
DB_PATH = os.environ.get("FEEDBACK_DB_PATH", DEFAULT_DB)

def init_db():
    # Ensure folder exists if someone sets FEEDBACK_DB_PATH to a nested path
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    con = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO feedback (created_at, row_index, fraud_probability, risk_band, action)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), row_index, fraud_probability, risk_band, action))
    con.commit()
    con.close()
