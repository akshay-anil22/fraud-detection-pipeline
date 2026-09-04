"""
Fraud Detection Pipeline - Transform + Feature Quality DAG

Stage 2 (this DAG):
  transform         -> build feature_transactions from raw_transactions
  validate_features -> quality checks + class-imbalance report

Kept independent from fraud_ingestion for isolated testing; chained in a later stage.
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

from transform_features import build_features, verify  # noqa: E402
from validate_features import run_checks  # noqa: E402

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def transform_task(**context):
    build_features()
    result = verify()
    context["task_instance"].xcom_push(key="transform_result", value=result)
    return result


def validate_features_task(**context):
    checks, warnings, failures = run_checks()
    print("\n".join(checks))
    for w in warnings:
        print(f"WARNING: {w}")
    if failures:
        raise ValueError(f"Feature quality checks failed: {failures}")
    return {"status": "passed", "checks": checks}


with DAG(
    dag_id="fraud_transform",
    default_args=default_args,
    description="Engineer ML-ready features from raw_transactions and gate on quality",
    schedule_interval="15 0 * * *",  # daily at 00:15, after the ingestion DAG
    catchup=False,
    tags=["transform", "etl"],
) as dag:

    transform = PythonOperator(
        task_id="transform",
        python_callable=transform_task,
        provide_context=True,
    )

    validate_features = PythonOperator(
        task_id="validate_features",
        python_callable=validate_features_task,
        provide_context=True,
    )

    transform >> validate_features