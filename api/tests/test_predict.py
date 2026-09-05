"""
Week 4 - serving-layer tests (kartik2112 contract, temporal-holdout model).

Fixtures are REAL rows snapshotted from feature_transactions (raw request
fields + the gold feature values the model trained on), carried in
fixtures/samples_dataset.json - so the suite runs with zero database.

The suite asserts the served path reproduces training on real rows:
  - encoder output == gold feature_transactions values (compute parity)
  - category_target == the PERSISTED category_target_map value
  - /predict probability == scoring the same artifact directly
Old v1-v28 PCA fixtures are gone: a stale "v1" body now 422s loudly
(extra="forbid"), which the negative test asserts.
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from feature_encoder import FEATURES, encode, encode_vector  # noqa: E402
from main import MODEL_PATH, app  # noqa: E402
from modeling_common import CATEGORY_MAP_PATH  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "fixtures", "samples_dataset.json")

with open(FIXTURES) as _f:
    SAMPLES = json.load(_f)
    FRAUD = next(s for s in SAMPLES if s["is_fraud"] == 1)
    LEGIT = next(s for s in SAMPLES if s["is_fraud"] == 0)


class _Tx(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def test_fixtures_are_real_rows():
    assert len(SAMPLES) == 12
    assert sum(1 for s in SAMPLES if s["is_fraud"] == 1) == 6
    assert sum(1 for s in SAMPLES if s["is_fraud"] == 0) == 6
    expected_keys = {"time", "amount", "category", "dob", "city_pop", "lat",
                     "long", "merch_lat", "merch_long", "is_new_merchant_for_card"}
    for s in SAMPLES:
        assert expected_keys <= set(s["request"])
        assert {f for f in s["expected_features"]}.issubset(set(FEATURES))


def test_encoder_matches_gold_features():
    with open(CATEGORY_MAP_PATH) as f:
        cat_map = json.load(f)
    for s in SAMPLES:
        got = encode(_Tx(s["request"]))
        exp = s["expected_features"]
        for feat in set(exp) | {"category_target"}:
            expected = (exp[feat] if feat in exp
                        else cat_map.get(s["request"]["category"], cat_map["fallback"]))
            assert got[feat] == pytest.approx(expected, rel=1e-6), \
                f"{s['trans_num']} {feat}: {got[feat]} != {expected}"


def test_encoder_vector_order_matches_features():
    X = encode_vector(_Tx(FRAUD["request"]))
    assert X.shape == (len(FEATURES),)
    assert FEATURES == ["hour_of_day", "is_weekend", "amount_log", "age",
                        "distance_km", "city_pop_bin",
                        "is_new_merchant_for_card", "category_target"]


def test_encoder_derived_math():
    feats = encode(_Tx(LEGIT["request"]))
    assert feats["amount_log"] == pytest.approx(
        (LEGIT["expected_features"]["amount_log"]), rel=1e-9)


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model artifact not mounted")
def test_predict_matches_persisted_artifact(client):
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    for s in SAMPLES:
        local = float(model.predict_proba(
            encode_vector(SimpleNamespace(**s["request"])).reshape(1, -1))[0, 1])
        r = client.post("/predict", json=s["request"])
        assert r.status_code == 200
        assert r.json()["fraud_probability"] == round(local, 4)


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model artifact not mounted")
def test_predict_fraud(client):
    r = client.post("/predict", json=FRAUD["request"])
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fraud_probability", "prediction", "latency_ms"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] == 1


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model artifact not mounted")
def test_predict_legit(client):
    r = client.post("/predict", json=LEGIT["request"])
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 0
    assert body["fraud_probability"] < 0.5


def test_predict_missing_amount_422(client):
    payload = dict(FRAUD["request"])
    del payload["amount"]
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_unknown_field_422(client):
    payload = dict(FRAUD["request"])
    payload["v1"] = -2.3122265423263
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_wrong_type_422(client):
    payload = dict(FRAUD["request"])
    payload["amount"] = "lots"
    assert client.post("/predict", json=payload).status_code == 422


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs the lifespan -> model load
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()) >= {"status", "model_path", "loaded_at"}


def test_metrics_exposes_counters(client):
    if os.path.exists(MODEL_PATH):
        client.post("/predict", json=LEGIT["request"])
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predict_requests_total" in r.text
    assert "predict_latency_seconds_bucket" in r.text


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="predict-btn"' in r.text
    assert "Fraud Detection API" in r.text
    assert "category" in r.text  # new-form select is present, no v1..v28 grid
    assert "v1" not in r.text


def test_zip_lookup_found(client):
    r = client.get("/zip/78702")
    assert r.status_code == 200
    body = r.json()
    assert body["zip"] == "78702"
    assert body["lat"] == pytest.approx(30.2638, abs=0.01)
    assert body["lng"] == pytest.approx(-97.7166, abs=0.01)
    assert body["state"] == "TX"


def test_zip_lookup_not_found(client):
    assert client.get("/zip/99999").status_code == 404


def test_zip_lookup_bad_code(client):
    assert client.get("/zip/12ab").status_code == 422
    assert client.get("/zip/123").status_code == 422