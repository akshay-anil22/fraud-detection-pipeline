"""
Week 4 - serving-layer tests.

Fixtures are HARDCODED snapshots of real rows from feature_transactions
(fraud + legit), so the suite runs with zero database dependency.

FRAUD and LEGIT are the raw request fields only: time, v1..v28, amount.
PARITY tables hold (time -> hour_of_day, is_weekend) pairs taken from the
feature table to prove the Python encoder matches the Week 2 SQL transform.
"""
import os
import sys
from math import log1p

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from feature_encoder import FEATURES, encode, encode_vector  # noqa: E402
from main import MODEL_PATH, app  # noqa: E402

# ---- Hardcoded fixtures (snapshot from feature_transactions) --------------
FRAUD = {
    "time": 406.0, "amount": 0.0,
    "v1": -2.3122265423263, "v2": 1.95199201064158, "v3": -1.60985073229769,
    "v4": 3.9979055875468, "v5": -0.522187864667764, "v6": -1.42654531920595,
    "v7": -2.53738730624579, "v8": 1.39165724829804, "v9": -2.77008927719433,
    "v10": -2.77227214465915, "v11": 3.20203320709635, "v12": -2.89990738849473,
    "v13": -0.595221881324605, "v14": -4.28925378244217, "v15": 0.389724120274487,
    "v16": -1.14074717980657, "v17": -2.83005567450437, "v18": -0.0168224681808257,
    "v19": 0.416955705037907, "v20": 0.126910559061474, "v21": 0.517232370861764,
    "v22": -0.0350493686052974, "v23": -0.465211076182388, "v24": 0.320198198514526,
    "v25": 0.0445191674731724, "v26": 0.177839798284401, "v27": 0.261145002567677,
    "v28": -0.143275874698919,
}
LEGIT = {
    "time": 0.0, "amount": 149.62,
    "v1": -1.3598071336738, "v2": -0.0727811733098497, "v3": 2.53634673796914,
    "v4": 1.37815522427443, "v5": -0.338320769942518, "v6": 0.462387777762292,
    "v7": 0.239598554061257, "v8": 0.0986979012610507, "v9": 0.363786969611213,
    "v10": 0.0907941719789316, "v11": -0.551599533260813, "v12": -0.617800855762348,
    "v13": -0.991389847235408, "v14": -0.311169353699879, "v15": 1.46817697209427,
    "v16": -0.470400525259478, "v17": 0.207971241929242, "v18": 0.0257905801985591,
    "v19": 0.403992960255733, "v20": 0.251412098239705, "v21": -0.018306777944153,
    "v22": 0.277837575558899, "v23": -0.110473910188767, "v24": 0.0669280749146731,
    "v25": 0.128539358273528, "v26": -0.189114843888824, "v27": 0.133558376740387,
    "v28": -0.0210530534538215,
}

# (time, expected_hour_of_day, expected_is_weekend) from the SQL transform.
PARITY = [
    (406.0, 0, 1),    # 2013-09-01 00:06:46Z, Sunday
    (4462.0, 1, 1),   # 2013-09-01 01:14:22Z, Sunday
    (172800.0, 0, 0), # 2013-09-03 00:00:00Z, Tuesday
]


# ---- Feature encoder ------------------------------------------------------
class _Tx(dict):
    """dict with attribute access so the encoder can be called directly."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _tx(row: dict):
    return _Tx(row)


def test_encoder_parity_with_sql_transform():
    for t, hour, weekend in PARITY:
        feats = encode(_tx({"time": t, "amount": 100.0, "v1": 0.0, "v2": 0.0,
                            "v3": 0.0, "v4": 0.0, "v5": 0.0, "v6": 0.0,
                            "v7": 0.0, "v8": 0.0, "v9": 0.0, "v10": 0.0,
                            "v11": 0.0, "v12": 0.0, "v13": 0.0, "v14": 0.0,
                            "v15": 0.0, "v16": 0.0, "v17": 0.0, "v18": 0.0,
                            "v19": 0.0, "v20": 0.0, "v21": 0.0, "v22": 0.0,
                            "v23": 0.0, "v24": 0.0, "v25": 0.0, "v26": 0.0,
                            "v27": 0.0, "v28": 0.0, "duplicate_flag": 0}))
        assert feats["hour_of_day"] == hour, f"time={t} hour mismatch"
        assert feats["is_weekend"] == weekend, f"time={t} weekend mismatch"


def test_encoder_vector_order_matches_features():
    X = encode_vector(_tx(FRAUD))
    assert X.shape == (len(FEATURES),)
    assert FEATURES == [
        "time", "amount", "hour_of_day", "is_weekend", "amount_log",
        "v_magnitude", "duplicate_flag", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28",
    ]


def test_encoder_derived_math():
    feats = encode(_tx(LEGIT))
    assert feats["amount_log"] == pytest.approx(log1p(149.62))
    assert feats["v_magnitude"] == pytest.approx(
        (sum(v ** 2 for v in [LEGIT[f"v{i}"] for i in range(1, 29)])) ** 0.5)


# ---- API ------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs the lifespan -> model load
        yield c


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model artifact not mounted")
def test_predict_fraud(client):
    r = client.post("/predict", json=FRAUD)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fraud_probability", "prediction", "latency_ms"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] == 1


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model artifact not mounted")
def test_predict_legit(client):
    r = client.post("/predict", json=LEGIT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 0
    assert body["fraud_probability"] < 0.5


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model artifact not mounted")
def test_predict_accepts_duplicate_flag(client):
    payload = dict(FRAUD)
    payload["duplicate_flag"] = 0
    r = client.post("/predict", json=payload)
    assert r.status_code == 200


def test_predict_missing_amount_422(client):
    payload = dict(FRAUD)
    del payload["amount"]
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_unknown_field_422(client):
    payload = dict(FRAUD)
    payload["oops"] = 1
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_wrong_type_422(client):
    payload = dict(FRAUD)
    payload["amount"] = "lots"
    assert client.post("/predict", json=payload).status_code == 422


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()) >= {"status", "model_path", "loaded_at"}


def test_metrics_exposes_counters(client):
    if os.path.exists(MODEL_PATH):
        client.post("/predict", json=LEGIT)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predict_requests_total" in r.text
    assert "predict_latency_seconds_bucket" in r.text


def test_root(client):
    assert client.get("/").json()["service"] == "fraud-detection-api"