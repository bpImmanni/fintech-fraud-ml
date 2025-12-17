📊 FinTech Fraud Detector (End-to-End ML App)

A production-ready fraud detection system for credit-card transactions, built with machine learning + analytics tooling, and deployed on Streamlit Cloud with memory-safe design.

🔗 Live Demo:
👉 https://fintech-fraud-ml.streamlit.app/


🚀 What this project does

Upload a transactions CSV

Get fraud probability + risk band per transaction

Optionally evaluate performance if labels are present

Monitor data drift (PSI) vs training baseline

Capture analyst feedback (Confirm Fraud / False Positive)

Explain predictions using SHAP (resource-safe mode)

This project demonstrates real-world ML deployment constraints (memory limits, safe reruns, UI-driven scoring) — not just offline modeling.


🧠 Model & ML Details

Model: Gradient-boosted tree classifier (XGBoost-style pipeline)

Features: PCA-transformed transaction features (V1–V28, Amount, Time)

Output:

fraud_probability

risk_band: LOW / MEDIUM / HIGH

Evaluation (if Class column exists):

Precision

Recall

TP / FP / FN / TN


🖥️ User Interface (Streamlit)
Tabs

Score

CSV upload

Manual Run Scoring button (prevents crashes)

Preview results

On-demand CSV download (memory-safe)

Optional SHAP explanations (limited rows)

Model Health (Drift)

Population Stability Index (PSI)

Compares live data vs training baseline

Feedback

Analyst confirmation loop

Stores feedback in SQLite (local /tmp DB)


🧩 Explainability (SHAP)

Shows top-3 feature contributions

Disabled by default on cloud

Hard-capped rows to prevent memory restarts

Demonstrates practical explainability under infra constraints


📈 Drift Monitoring

Uses PSI (Population Stability Index)

Baseline sampled from training data

Flags:

🟢 OK

🟡 Moderate drift

🔴 High drift



🗂️ Project Structure
fintech-fraud-ml/
├── app/
│   └── app.py              # Streamlit application
├── src/
│   ├── train.py            # Model training
│   ├── drift.py            # PSI drift logic
│   ├── explain.py          # SHAP helpers
│   ├── feedback.py         # SQLite feedback loop
│   └── api.py              # (optional / local only)
├── models/
│   └── fraud_model.pkl     # Trained model
├── reports/
│   └── schema.json         # Expected feature schema
├── data/
│   └── processed/
│       ├── train_baseline_sample.csv
│       ├── shap_background.csv
│       └── .gitkeep
├── requirements.txt
├── README.md
└── .gitignore


