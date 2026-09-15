# PMLDL Assignment 1 — Adult Income MLOps Pipeline

## Structure
- `code/datasets` — data engineering
- `code/models` — model engineering (CatBoost / XGBoost / LightGBM + Optuna + MLflow)
- `code/deployment` — FastAPI + Streamlit + docker-compose
- `services/airflow/dags` — DAG, runs every 5 min

## Run locally
```bash
pip install -r requirements.txt
python code/datasets/data_engineering.py
python code/models/model_engineering.py
cd code/deployment && docker compose up -d --build