# Fraud Detection ML Pipeline

End-to-end fraud detection system with automated ETL, model training, containerized serving, and monitoring.

## Architecture

```
Kaggle API  ->  Airflow DAGs  ->  PostgreSQL  ->  Model Training  ->  FastAPI  ->  Prometheus/Grafana
                    (ETL)          (warehouse)     (XGBoost)        (serving)      (monitoring)

DAG fraud_ingestion:   fetch_data   -> validate_raw        (Week 1)
DAG fraud_transform:   transform    -> validate_features   (Week 2)
DAG fraud_train:       train_model  -> evaluate_model      (Week 3)
API fraud_api:         FastAPI /predict + /metrics          (Week 4)
Monitor:               Prometheus scrapes api, Grafana dashboards  (Week 5)
```

## Stages

| Stage | DAG | Input | Output | Status |
|---|---|---|---|---|
| Week 1 - Ingest | `fraud_ingestion` | Kaggle CSV | `raw_transactions` | done |
| Week 2 - Transform | `fraud_transform` | `raw_transactions` | `feature_transactions` | done |
| Week 3 - Train | `fraud_train` | `feature_transactions` | XGBoost model + `model_metrics` | done |
| Week 4 - Serve | `fraud_api` | model | FastAPI `/predict` + `/metrics` | done |
| Week 5 - Monitor | `prometheus` + `grafana` | API metrics | "Fraud Monitoring" dashboard | done |

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

[Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — 1,852,394 transactions (2019–2020), 9,651 fraudulent (~0.52%). `source_split` marks the chronological train/test boundary: the model is trained on the earlier half and evaluated on the later half it never saw.

## Feature Engineering (Week 2)

`feature_transactions` (gold) carries the engineering the model trains on:

| Feature | Definition | Purpose |
|---|---|---|
| `hour_of_day` | UTC hour of the transaction | fraud clusters at certain hours |
| `is_weekend` | 1 on Sat/Sun | weekend activity signal |
| `amount_log` | `ln(amt + 1)` | spreads out tiny fraud amounts |
| `age` | whole calendar years as of tx (from DOB); >110 nulled | demography signal |
| `distance_km` | great-circle distance card-pos → merchant-pos | impossible-travel fraud |
| `city_pop_bin` | 0..3 buckets of cardholder `city_pop` | small-city concentration |
| `is_new_merchant_for_card` | 1 if first time this card uses this merchant | card-testing pattern |

Quality gate (`validate_features.py`) runs structural + semantic checks plus a per-split class-imbalance report (the TRAIN split sets Week 3 `scale_pos_weight`).

## Model Training (Week 3)

`fraud_train` is manual-only (`schedule=None`) — the dataset is static, so daily retraining is wasted compute. Trigger via Airflow UI or `airflow dags trigger fraud_train`.

The model uses the dataset's own **chronological split**, not a random one: trained on `source_split='train'` (1,296,675 rows) and evaluated on the later `source_split='test'` (555,719 rows) — 7,506 + 2,145 = 9,651 total frauds. `scale_pos_weight` (171.75) comes from the train split only.

- `scripts/train_model.py` — trains on the temporal train split. Category target encoding is fit on **train only**, smoothed (m=30), and persisted to `category_target_map.json`; train-split imputation values persist to `imputation_values.json`. Artifacts are written *before* the matrices are built, so evaluate and serve consume the exact same numbers. Fixed `n_estimators=300` (no early stopping) keeps the saved/loaded tree count identical.
- `scripts/evaluate_model.py` — reloads the artifact + persisted maps, rebuilds the same matrices through the shared `build_matrices()`, scores the temporal test split, writes `metrics_latest.json`, inserts a row into `model_metrics` (schema in `sql/create_model_metrics.sql`), and runs a predict-one sanity check (fraud ≈0.9998 / legit ≈0.0003).
- `scripts/modeling_common.py` — single source of truth for the 8-feature list, artifact paths, transform constants, and hyperparameters.

Latest temporal-holdout performance (test = the future months the model never saw): ROC-AUC 0.998, recall 0.947 (2,031/2,145 frauds caught), F1 0.431.

## Serving (Week 4)

The trained artifact is served by a FastAPI container (`fraud_api`, port 8000). It mounts `models/` + `airflow/scripts/` read-only for the feature contract. The served encoder **reads the persisted Week-3 artifacts** (`category_target_map.json` + `imputation_values.json`) and mirrors the training transform 1:1 — no learned value is recomputed or hardcoded, so train == evaluate == serve.

- `POST /predict` — body is a kartik2112 raw transaction (`time`, `amount`, `category`, `dob`, `city_pop`, cardholder `lat`/`long`, `merch_lat`/`merch_long`, optional `is_new_merchant_for_card`); returns `{fraud_probability, prediction, latency_ms}`. Stale v1-v28 payloads now 422 loudly (`extra="forbid"`).
- `GET /metrics` — Prometheus text format: request count, latency histogram, outcome counter (scraped directly, no exporter).
- `GET /health` — liveness + model fingerprint (artifact mtime/size, loaded-at).
- Feature math mirrors the Week 2 SQL transform 1:1: `hour_of_day`/`is_weekend` derive from `time` via the fixed `2013-09-01 00:00:00Z` anchor; age/distance use the same helpers as training; population buckets + target encoding come from the persisted files. Feature order comes from `modeling_common.FEATURES` so train and serve can't drift.

Example: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @tx.json`

The demo console (`/`) is a form over the raw fields with a "Load random sample" that pulls 12 real gold-table transactions (6 fraud + 6 legit).

Parity: `api/parity_check.py` proves the served encoding reproduces training on 12 real rows (`encoder == gold features`, `served probability == artifact score`). Tests: `api/tests/test_predict.py` — 13 checks on real-row fixtures, no DB needed at test time (`pytest` via `api/requirements-dev.txt`).

## Monitoring (Week 5)

Prometheus (port 9090) pull-scrapes `http://api:8000/metrics` every 5s; Grafana (port 3000, `admin/admin`) is auto-provisioned with the **Fraud Monitoring** dashboard (request rate, latency p50/p95, outcome split, 5-minute flagged-fraud rate, API/Prometheus availability). All config is files under `monitoring/` — no manual setup.

Note: Prometheus runs **without a persistent volume**, so metrics reset on container restart — acceptable for a demo environment.

## License

MIT
