"""
Fraud Detection Pipeline - Training DAG

Stage 3 (this DAG):
  train    -> stratified split, fit XGBoost with scale_pos_weight, save artifact
  evaluate -> reload artifact, score held-out test set, persist metrics

schedule=None: training only makes sense when the underlying data changes,
which never happens for this static Kaggle dataset. Re-running the same rows
every day is wasted compute, so this DAG is manual/on-demand only. A real
streaming-data pipeline would gate retraining on new-data volume or drift.
"""
import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

DAG_FOLDER = "/opt/airflow"
SCRIPTS_DIR = os.path.join(DAG_FOLDER, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)  # allow operator imports from scripts

from evaluate_model import evaluate  # noqa: E402
from train_model import train  # noqa: E402

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 0,  # don't auto-re-run a training job on failure
}


def train_task(**context):
    result = train()
    context["task_instance"].xcom_push(key="train_result", value=result)
    return result


def evaluate_task(**context):
    dag_run_id = context["dag_run"].run_id
    metrics = evaluate(dag_run_id=dag_run_id)
    context["task_instance"].xcom_push(key="metrics", value=metrics)
    return metrics


with DAG(
    dag_id="fraud_train",
    default_args=default_args,
    description="Train XGBoost on feature_transactions and persist evaluation metrics",
    schedule=None,  # on-demand only - static dataset
    catchup=False,
    tags=["train", "ml"],
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_task,
        provide_context=True,
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_task,
        provide_context=True,
    )

    train_model >> evaluate_model