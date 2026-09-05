"""
Shared config for Week 3 training + evaluation and the Week 2 transform
contract (single source of truth so SQL, train, and serve cannot drift).

Part 1 - feature list + transform constants (used by transform_features.py,
train/evaluate, and the serving encoder).
Part 2 - artifact paths (shared volume mounted at /opt/airflow/models).
"""
import os
from datetime import date, datetime
from math import asin, cos, radians, sin, sqrt

# ---- Database -----------------------------------------------------------
TABLE = "feature_transactions"
TARGET = "is_fraud"

# ---- Artifact paths (shared volume mounted at /opt/airflow/models) ------
MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/airflow/models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_fraud_model.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importances.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_latest.json")
CATEGORY_MAP_PATH = os.path.join(MODELS_DIR, "category_target_map.json")
IMPUTATION_PATH = os.path.join(MODELS_DIR, "imputation_values.json")

# ---- Reproducibility ----------------------------------------------------
# Fixed so every run produces identical splits + identical model numbers.
RANDOM_STATE = 42

# ---- Feature list -------------------------------------------------------
# The training matrix is 8 numeric features. category_target is learned at
# train time (fit on train split only) and injected from CATEGORY_MAP_PATH.
# Explicitly EXCLUDED from X (documented decisions):
#   id, trans_num, ingested_at        - bookkeeping / identifiers
#   source_split, unix_time           - the split marker + absolute clock
#   trans_date_trans_time             - raw timestamp (signal captured by
#                                        hour_of_day / is_weekend / age)
#   cc_num, merchant, category        - identities; category enters via
#                                        target encoding, merchant only via
#                                        is_new_merchant_for_card
#   amt, city_pop                     - raw forms; features are their
#                                        transforms (amount_log, city_pop_bin)
#   is_fraud                          - the target (y)
FEATURES = [
    "hour_of_day",
    "is_weekend",
    "amount_log",
    "age",
    "distance_km",
    "city_pop_bin",
    "is_new_merchant_for_card",
    "category_target",
]

# ---- Transform constants (fixed, documented - never fit-derived) -------
# city_pop buckets: <10k, 10k-100k, 100k-1M, >=1M
# Evidence (raw scan): p50~2408, p90~186140, p99~1577385, max~2906700.
POP_BIN_EDGES = [10_000, 100_000, 1_000_000]

# age guard: NULL only on parse failure or implausibly-old (>110).
# Under-18 rows are VALID parseable rows with higher-than-average fraud
# (train: 0.759% vs 0.579% overall) - they are kept, not nulled.
AGE_MAX_GUARD = 110

# distance_km guard: NULL when any coordinate is missing.
HAVERSINE_RADIUS_KM = 6371.0

# Deterministic fills for NULLs (train computes/stores the real ones in
# IMPUTATION_PATH at fit time; serve reads that file, never hardcodes):
DISTANCE_FILL = 0.0
AGE_FILL = "auto"  # replaced by train split median at fit time


def age_years(tx_ts: datetime, dob: str) -> float:
    """Whole calendar years, matching Postgres EXTRACT(year FROM age(tx, dob))."""
    bday = date.fromisoformat(str(dob)[:10])
    years = tx_ts.year - bday.year
    if (tx_ts.month, tx_ts.day) < (bday.month, bday.day):
        years -= 1
    return float(years)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km - byte-matches the SQL transform."""
    la1, la2, lo1, lo2 = radians(lat1), radians(lat2), radians(lon1), radians(lon2)
    h = (
        sin((la2 - la1) / 2) ** 2
        + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    )
    return 2 * HAVERSINE_RADIUS_KM * asin(sqrt(h))


def city_pop_bin(pop: int) -> int:
    """Fixed population bucket code 0..3, mirrors the SQL CASE."""
    for code, edge in enumerate(POP_BIN_EDGES):
        if pop < edge:
            return code
    return len(POP_BIN_EDGES)


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