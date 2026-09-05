# Fraud Detection ML Pipeline

End-to-end fraud detection system with automated ETL, model training, containerized serving, and monitoring.

## Architecture

```
Kaggle API  ->  Airflow DAGs  ->  PostgreSQL  ->  Model Training  ->  FastAPI  ->  Prometheus/Grafana
                    (ETL)          (warehouse)     (XGBoost)        (serving)      (monitoring)

DAG fraud_ingestion:   fetch_data   -> validate_raw        (Week 1)
DAG fraud_transform:   transform    -> validate_features   (Week 2)
DAG fraud_train:       train_model  -> evaluate_model      (Week 3)
```

## Stages

| Stage | DAG | Input | Output | Status |
|---|---|---|---|---|
| Week 1 - Ingest | `fraud_ingestion` | Kaggle CSV | `raw_transactions` | done |
| Week 2 - Transform | `fraud_transform` | `raw_transactions` | `feature_transactions` | done |
| Week 3 - Train | `fraud_train` | `feature_transactions` | XGBoost model + `model_metrics` | done |
| Week 4 - Serve | (planned) | model | FastAPI `/predict` | pending |
| Week 5 - Monitor | (planned) | API metrics | Prometheus/Grafana | pending |

## Tech Stack

- **Orchestration:** Apache Airflow 2.10 (LocalExecutor)
- **Database:** PostgreSQL 15
- **ML:** XGBoost, scikit-learn, pandas
- **Serving:** FastAPI + uvicorn
- **Monitoring:** Prometheus + Grafana
- **Containerization:** Docker Compose

## Quick Start

```bash
cp .env.example .env        # add your Kaggle API token
docker compose up -d        # start all services
# Airflow UI: http://localhost:8080 (admin / admin)
```

## Project Structure

```
fraud-detection-pipeline/
├── airflow/
│   ├── dags/          # Airflow DAG definitions
│   ├── scripts/       # Python scripts for DAG operators
│   └── requirements.txt
├── api/               # FastAPI serving layer
├── ml/                # Model training + evaluation
├── sql/               # PostgreSQL schema definitions
├── docker/            # Dockerfiles
├── monitoring/        # Prometheus + Grafana config
├── docker-compose.yml
└── .env.example
```

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 31 features, ~0.17% fraud rate.

## Feature Engineering (Week 2)

`feature_transactions` adds 7 engineered columns to the raw 31:

| Feature | Definition | Purpose |
|---|---|---|
| `tx_datetime` | readable timestamp (`time` anchored at 2013-09-01 00:00:00 UTC) | readable clock for downstream features |
| `hour_of_day` | 0-23 | fraud clusters at certain hours |
| `is_weekend` | 1 on Sat/Sun | weekend activity signal |
| `amount_log` | `ln(amount+1)` | spreads out tiny fraud amounts |
| `amount_bin` | small/medium/large/xl | coarse amount buckets |
| `v_magnitude` | `sqrt(sum(v1..v28^2))` | PCA vector magnitude |
| `duplicate_flag` | 1 if identical `(time, v1..v28, amount)` appears >1x | card-testing pattern |

Quality gate runs 6 checks + a class-imbalance report (baseline for Week 3 `scale_pos_weight`).

## Model Training (Week 3)

`fraud_train` is manual-only (`schedule=None`) — the dataset is static, so daily retraining is wasted compute. Trigger via Airflow UI or `airflow dags trigger fraud_train`.

- `scripts/train_model.py` — stratified 80/20 split (seeded), fits XGBoost with `scale_pos_weight` derived from the actual train split (~577:1), saves `xgb_fraud_model.json` + `feature_importances.json`.
- `scripts/evaluate_model.py` — reloads the artifact, scores the held-out set, writes `metrics_latest.json`, inserts a row into `model_metrics` (schema in `sql/create_model_metrics.sql`), and runs a predict-one sanity check.
- `scripts/modeling_common.py` — single source of truth for the feature list, paths, seed, and hyperparameters. `tx_datetime` is explicitly excluded from X (`hour_of_day`/`is_weekend` carry the signal).

Latest held-out performance: ROC-AUC 0.974, recall 0.837, precision 0.872, F1 0.854.

## License

MIT
