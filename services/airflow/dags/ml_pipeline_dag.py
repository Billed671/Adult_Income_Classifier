"""
Full MLOps pipeline:
1) data engineering
2) model engineering
3) deployment (docker compose up --build)
Runs every 5 minutes.
"""
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"
HOST_PROJECT_DIR = os.environ.get("HOST_PROJECT_DIR", PROJECT_DIR)

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="adult_income_pipeline",
    description="Data eng -> Model eng -> Deploy",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "assignment"],
) as dag:

    data_engineering = BashOperator(
        task_id="data_engineering",
        bash_command=f"cd {PROJECT_DIR} && python code/datasets/data_engineering.py",
    )

    model_engineering = BashOperator(
        task_id="model_engineering",
        bash_command=f"cd {PROJECT_DIR} && python code/models/model_engineering.py",
    )

    deployment = BashOperator(
        task_id="deployment",
        bash_command=(
            f'docker compose '
            f'--project-directory "{HOST_PROJECT_DIR}/code/deployment" '
            f'-f "{PROJECT_DIR}/code/deployment/docker-compose.yml" '
            f'up -d --build'
        ),
    )

    data_engineering >> model_engineering >> deployment