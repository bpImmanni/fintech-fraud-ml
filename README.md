# FinTech Fraud Detector (End-to-End ML + MLOps)

An end-to-end fraud detection system for card transactions:
- **Model**: XGBoost classifier
- **UI**: Streamlit (CSV upload → fraud probability + risk band)
- **API**: FastAPI `/score` endpoint for batch scoring
- **Explainability**: SHAP top reasons (optional)
- **Monitoring**: Drift detection (PSI) vs training baseline
- **Ops**: MLflow experiment tracking (optional), Docker + Compose

## Demo
- Streamlit UI: (add link after deploy)
- FastAPI API: (add link after deploy)

## Project structure
app/ # Streamlit UI
src/ # training, api, drift, explainability, feedback
models/ # trained model artifact (fraud_model.pkl)
reports/ # schema + metrics
tests/ # pytest tests
Dockerfile
docker-compose.yml
requirements.txt


## Run locally

### Create env + install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
# POST /score with JSON:
{"records":[{"Time":0,"V1":...,"V28":...,"Amount":149.62}]}
# Folder structure is preserved using .gitkeep.


Commit/push:
git add README.md
git commit -m "Add README"
git push
