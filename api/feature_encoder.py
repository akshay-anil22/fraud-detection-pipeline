"""Raw transaction -> the exact Week 3 (temporal holdout) 8-feature vector.

The encoder MUST produce the same numbers training produced for the same row.
Two mechanisms guarantee that:

  1. Derived math mirrors the Week 2 SQL transform / modeling_common 1:1:
       - hour_of_day / is_weekend from the fixed 2013-09-01T00:00:00Z anchor
       - amount_log = ln(amount + 1)
       - age via modeling_common.age_years (whole calendar years as of the
         transaction timestamp, matching Postgres EXTRACT(year FROM age()))
       - distance_km via modeling_common.haversine_km (matching the SQL)
       - city_pop_bin via the persisted population bucket edges
  2. Every learned/fit value is READ from the persisted Week 3 artifacts
     (category_target_map.json + imputation_values.json) - the same files
     train_model wrote and evaluate_model consumed. Nothing is recomputed or
     hardcoded, so serve == train == evaluate by construction.

is_new_merchant_for_card requires card-merchant history to compute (dataset
transform computes it via a window function); like the old duplicate_flag, the
caller supplies it and it defaults to 0 when unknown.

The feature ORDER is imported from modeling_common so serving can never drift
from training; the container bind-mounts airflow/scripts for that import.
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, "/opt/airflow/scripts")
from modeling_common import FEATURES, age_years, haversine_km  # noqa: E402

ANCHOR_UTC = datetime(2013, 9, 1, tzinfo=timezone.utc)

MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/airflow/models")


def _load_artifacts():
    with open(os.path.join(MODELS_DIR, "category_target_map.json")) as f:
        cat_map = json.load(f)
    with open(os.path.join(MODELS_DIR, "imputation_values.json")) as f:
        imp = json.load(f)
    return cat_map, imp


CAT_MAP, IMP = _load_artifacts()
_AGE_MAX = IMP.get("age_max_guard", 110)
_DIST_FILL = IMP.get("distance_fill", 0.0)
_POP_EDGES = IMP.get("pop_bin_edges", [10_000, 100_000, 1_000_000])
_AGE_MEDIAN = IMP.get("age_median")


def _age_from_dob(tx_ts: datetime, dob) -> float | None:
    if not dob:
        return None
    try:
        age = age_years(tx_ts, str(dob))
    except (ValueError, TypeError):
        return None
    if age <= 0 or (_AGE_MAX is not None and age > _AGE_MAX):
        return None
    return float(age)


def _distance_km(tx) -> float:
    if None in (tx.lat, tx.long, tx.merch_lat, tx.merch_long):
        return _DIST_FILL
    return haversine_km(
        float(tx.lat), float(tx.long),
        float(tx.merch_lat), float(tx.merch_long),
    )


def _city_pop_bin(pop: float) -> int:
    code = 0
    for edge in _POP_EDGES:
        if pop < edge:
            return code
        code += 1
    return code


def _category_target(category: str) -> float:
    mapping = {k: v for k, v in CAT_MAP.items() if k != "fallback"}
    return float(mapping.get(category, CAT_MAP["fallback"]))


def encode(tx) -> dict:
    """Turn a TransactionRequest into the named feature dict (Week 3 order)."""
    dt = ANCHOR_UTC + timedelta(seconds=float(tx.time))

    age = _age_from_dob(dt, getattr(tx, "dob", None))
    if age is None:
        age = float(_AGE_MEDIAN) if _AGE_MEDIAN is not None else 0.0

    features = {
        "hour_of_day": dt.hour,
        "is_weekend": 1 if dt.isoweekday() in (6, 7) else 0,
        "amount_log": math.log(float(tx.amount) + 1),
        "age": age,
        "distance_km": _distance_km(tx),
        "city_pop_bin": _city_pop_bin(float(getattr(tx, "city_pop", 0.0))),
        "is_new_merchant_for_card": int(getattr(tx, "is_new_merchant_for_card", 0) or 0),
        "category_target": _category_target(getattr(tx, "category", "") or ""),
    }
    return features


def encode_vector(tx) -> np.ndarray:
    """Feature dict -> float64 vector in exact FEATURES order (as in training)."""
    f = encode(tx)
    return np.asarray([f[name] for name in FEATURES], dtype=np.float64)