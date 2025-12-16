import os
import sqlite3
from datetime import datetime

# Render-safe writable location
DB_PATH = os.getenv("FEEDBACK_DB_PATH", "/tmp/feedback.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_index INTEGER,
            fraud_probability REAL,
            risk_band TEXT,
            label TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def write_feedback(row_index, fraud_probability, risk_band, label):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO feedback (row_index, fraud_probability, risk_band, label, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            row_index,
            fraud_probability,
            risk_band,
            label,
            datetime.utcnow().isoformat()
        )
    )

    conn.commit()
    conn.close()
