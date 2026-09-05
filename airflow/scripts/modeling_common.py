"""
Shared config for Week 3 training + evaluation.

Single source of truth for the feature list, artifact paths, and the
reproducibility seed so train and evaluate cannot drift apart.
"""
import os

# ---- Database -----------------------------------------------------------
TABLE = "feature_transactions"
TARGET = "class"

# ---- Artifact paths (shared volume mounted at /opt/airflow/models) ------
MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/airflow/models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_fraud_model.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importances.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_latest.json")

# ---- Reproducibility ----------------------------------------------------
# Fixed so every run produces identical splits + identical model numbers.
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---- Feature list -------------------------------------------------------
# Explicitly EXCLUDED from X (documented decision, not accidental omission):
#   id, ingested_at     - row/bookkeeping columns, no predictive signal
#   tx_datetime         - raw timestamp column. XGBoost cannot consume it
#                         (non-numeric), and its signal is already captured by
#                         the derived numeric features hour_of_day and
#                         is_weekend, which ARE included below.
#   amount_bin          - derived-from-amount bucket kept out of X to avoid
#                         redundancy with amount / amount_log.
#   class               - the target (y), kept out of X.
FEATURES = [
    "time",
    "amount",
    "hour_of_day",
    "is_weekend",
    "amount_log",
    "v_magnitude",
    "duplicate_flag",
] + [f"v{i}" for i in range(1, 29)]

# ---- XGBoost hyperparameters --------------------------------------------
# scale_pos_weight is filled in at train time from the actual train split.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
}


def get_engine():
    from sqlalchemy import create_engine

    POSTGRES_USER = os.environ["POSTGRES_USER"]
    POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
    POSTGRES_DB = os.environ["POSTGRES_DB"]
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )