FROM python:3.11-slim

WORKDIR /app

# System deps (helps shap/numba/xgboost wheels in some cases)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

# default command (compose will override per-service)
CMD ["bash", "-lc", "streamlit run app/app.py --server.address=0.0.0.0 --server.port=8501"]
