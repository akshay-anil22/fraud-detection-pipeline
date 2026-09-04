"""
Fraud Detection Pipeline - Ingestion DAG

Stage 1 (this DAG):
  fetch_data   -> download CSV from Kaggle and load into PostgreSQL
  validate_raw -> verify row/column counts, nulls, and class distribution

Later stages (future DAGs / tasks): transform, train, serve, monitor
"""
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DAG_FOLDER = "/opt/airflow"
SCRIPTS_DIR = os.path.join(DAG_FOLDER, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)  # allow operator imports from scripts

from fetch_data import download_dataset, load_to_postgres  # noqa: E402
from validate_raw import run_checks  # noqa: E402

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def fetch_data_task(**context):
    csv_path = download_dataset()
    result = load_to_postgres(csv_path)
    context["task_instance"].xcom_push(key="ingest_result", value=result)
    return result


def validate_raw_task(**context):
    checks, failures = run_checks()
    print("\n".join(checks))
    if failures:
        raise ValueError(f"Data quality checks failed: {failures}")
    return {"status": "passed", "checks": checks}


with DAG(
    dag_id="fraud_ingestion",
    default_args=default_args,
    description="Extract Credit Card Fraud dataset from Kaggle and load to PostgreSQL",
    schedule_interval="@daily",
    catchup=False,
    tags=["ingestion", "etl"],
) as dag:

    fetch_data = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data_task,
        provide_context=True,
    )

    validate_raw = PythonOperator(
        task_id="validate_raw",
        python_callable=validate_raw_task,
        provide_context=True,
    )

    fetch_data >> validate_raw
