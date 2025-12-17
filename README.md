# 🚨 FinTech Fraud Detector — End-to-End ML System

A **production-grade fraud detection platform** for card transactions, combining  
**machine learning, explainability, monitoring, and analyst feedback** — delivered through a scalable Streamlit application.

🔗 **Live Demo**  
👉 https://fintech-fraud-ml.streamlit.app/

---

## ✨ Key Features

### 🔍 Fraud Scoring
- Upload transaction CSV files
- Predict **fraud probability** per transaction
- Assign **risk bands**: `LOW / MEDIUM / HIGH`
- Adjustable fraud threshold

### 📊 Model Evaluation (Optional)
- Automatically detects `Class` label if present
- Displays **Precision, Recall, TP / FP / FN / TN**

### 🧠 Explainability (SHAP)
- Top-3 feature contributions per transaction
- Optimized for memory-safe execution
- Disabled by default on cloud to prevent crashes

### 📈 Drift Monitoring
- Population Stability Index (PSI)
- Compares live data vs training baseline
- Highlights **distribution shift risks**

### 📝 Analyst Feedback Loop
- Confirm Fraud / Mark False Positive
- Stored safely using **SQLite**
- Designed for future model retraining pipelines

---

## 🏗️ Architecture Overview

CSV Upload  
↓  
Schema Validation  
↓  
XGBoost Model  
↓  
Fraud Probability + Risk Band  
↓  
┌─────────────┬─────────────┬─────────────┐  
│ Explainable │ Drift (PSI) │ Analyst │  
│ AI (SHAP) │ Monitoring │ Feedback DB │  
└─────────────┴─────────────┴─────────────┘  


---

## 🧪 Tech Stack

| Layer | Technology |
|-----|-----------|
| Model | XGBoost |
| UI | Streamlit |
| Explainability | SHAP |
| Drift Monitoring | PSI |
| Storage | SQLite |
| Serialization | Joblib |
| Deployment | Streamlit Cloud |
| Language | Python |

---

## 📁 Project Structure

fintech-fraud-ml/
│
├── app/
│ └── app.py # Streamlit UI
│
├── src/
│ ├── train.py # Model training
│ ├── drift.py # PSI drift detection
│ ├── explain.py # SHAP logic
│ ├── feedback.py # SQLite feedback store
│
├── models/
│ └── fraud_model.pkl # Trained model
│
├── data/
│ └── processed/
│ ├── train_baseline_sample.csv
│ ├── shap_background.csv
│
├── reports/
│ └── schema.json # Feature schema
│
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶️ Run Locally

### 1️⃣ Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Mac / Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2️⃣ Train Model
```bash
python src/train.py
```

### 3️⃣ Launch App
```bash
streamlit run app/app.py
```

## ☁️ Cloud Deployment (Streamlit)

This project is deployed on Streamlit Cloud with:

Memory-safe row limits

On-demand scoring & downloads

Disabled auto-reruns

SQLite stored in writable temp storage

🔗 Live App
👉 https://fintech-fraud-ml.streamlit.app/


## ⚠️ Design Decisions (Why This Works in Production)

Manual “Run Scoring” button avoids accidental recomputation

Row caps prevent out-of-memory crashes

Session-state minimization for cloud stability

On-demand CSV generation instead of auto downloads

SHAP guarded to prevent instance restarts

SQLite feedback designed for future retraining loops

## 🚀 Future Enhancements

1. 🔄 Automated retraining using analyst feedback

2. 📡 API service for real-time scoring

3. 🧪 Model versioning & rollback

4. 📊 Dashboard-level monitoring (Prometheus / Evidently)

5. 🔐 Auth + role-based access


👨‍💻 Author

Bhanu Prakash Immanni  
ML Engineer | FinTech & Analytics  
🔗 https://www.linkedin.com/in/bpimmanni/ / https://github.com/bpImmanni
