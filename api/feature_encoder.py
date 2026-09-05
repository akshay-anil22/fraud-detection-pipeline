"""Raw transaction -> the exact 36 features the model was trained on.

The encoder MUST reproduce the Week 2 SQL transform bit-for-bit, otherwise
serving silently disagrees with training. The two tricky derivations:

  hour_of_day / is_weekend come from the same fixed epoch anchor as the SQL:

      tx_datetime = TIMESTAMPTZ '2013-09-01 00:00:00+00' + time * INTERVAL '1 second'
      hour_of_day = EXTRACT(hour FROM tx_datetime)             (UTC hour)
      is_weekend  = 1 WHEN EXTRACT(isodow FROM tx_datetime) IN (6,7) ELSE 0

  Python mirrors that exactly: datetime(2013,9,1, tzinfo=utc) + timedelta(seconds=time),
  with .hour for the UTC hour and .isoweekday() in {6,7} for Sat/Sun.

The feature ORDER is imported from modeling_common so serving can never drift
from training; the container bind-mounts airflow/scripts for that import.
"""
import math
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, "/opt/airflow/scripts")  # single source of truth: modeling_common.FEATURES
from modeling_common import FEATURES  # noqa: E402

ANCHOR_UTC = datetime(2013, 9, 1, tzinfo=timezone.utc)  # time=0 maps here (community convention)


def encode(tx) -> dict:
    """Turn a TransactionRequest into the named feature dict."""
    dt = ANCHOR_UTC + timedelta(seconds=float(tx.time))

    features = {
        "time": float(tx.time),
        "amount": float(tx.amount),
        "hour_of_day": dt.hour,
        "is_weekend": 1 if dt.isoweekday() in (6, 7) else 0,
        "amount_log": math.log(float(tx.amount) + 1),
        "v_magnitude": math.sqrt(sum(float(getattr(tx, f"v{i}")) ** 2 for i in range(1, 29))),
        "duplicate_flag": int(getattr(tx, "duplicate_flag", 0)),
    }
    for i in range(1, 29):
        features[f"v{i}"] = float(getattr(tx, f"v{i}"))
    return features


def encode_vector(tx) -> np.ndarray:
    """Feature dict -> float64 vector in exact FEATURES order (as in training)."""
    f = encode(tx)
    return np.asarray([f[name] for name in FEATURES], dtype=np.float64)