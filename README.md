# Fraud Detection ML Pipeline

End-to-end fraud detection system with automated ETL, model training, containerized serving, and monitoring.

## Architecture

```
Kaggle API  ->  Airflow DAG  ->  PostgreSQL  ->  Model Training  ->  FastAPI  ->  Prometheus/Grafana
                   (ETL)         (warehouse)     (XGBoost)        (serving)      (monitoring)
```

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

## License

MIT
