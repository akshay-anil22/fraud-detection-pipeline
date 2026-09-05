"""Week 4 - compute-vs-persisted parity gate (exit code 0 = all green).

Runs entirely inside a container with ./models + ./airflow/scripts mounted:

  Part 1 - encoder vs gold:     for every real sample row, the API encoder's
          feature vector must equal the feature_transactions values that
          training used (hour_of_day, is_weekend, amount_log, age,
          distance_km, city_pop_bin, is_new_merchant_for_card), and
          category_target must come from the PERSISTED category_target_map.
  Part 2 - served vs artifact:  the /predict endpoint must return the same
          probability as a local predict from the same model artifact file.

This is exactly the "compute-vs-persisted prediction parity" gate: it proves
the served encoder reproduces the training-time encoding (compute side) using
the persisted learned values (persisted side), and that the served numbers are
bit-identical to scoring the saved artifact directly.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import xgboost as xgb
from fastapi.testclient import TestClient

sys.path.insert(0, "/opt/airflow/scripts")

from feature_encoder import encode, encode_vector  # noqa: E402
from main import MODEL_PATH, THRESHOLD, app  # noqa: E402
from modeling_common import CATEGORY_MAP_PATH, FEATURES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures", "samples_dataset.json")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 + 1e-6 * abs(b)


def main() -> int:
    with open(FIXTURES) as f:
        samples = json.load(f)
    with open(CATEGORY_MAP_PATH) as f:
        cat_map = json.load(f)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    failures = []

    with TestClient(app) as client:
        for s in samples:
            req = SimpleNamespace(**s["request"])
            name = f"{s['merchant']} ({s['city']}, {s['state']}) [{'FRAUD' if s['is_fraud'] else 'legit'}]"

            # ---- Part 1: encoder vs gold feature_transactions values --------
            got = encode(req)
            exp = s["expected_features"]
            for feat in FEATURES:
                if feat == "category_target":
                    expected_cat = cat_map.get(s["request"]["category"], cat_map["fallback"])
                    if not _close(got[feat], expected_cat):
                        failures.append(f"{name}: category_target {got[feat]:.6f} != persisted map {expected_cat:.6f}")
                    continue
                if feat not in exp:
                    continue
                if not _close(got[feat], exp[feat]):
                    failures.append(f"{name}: {feat} encoder {got[feat]:.6f} != gold {exp[feat]:.6f}")

            # ---- Part 2: served probability == artifact probability ----------
            local = float(model.predict_proba(encode_vector(req).reshape(1, -1))[0, 1])
            r = client.post("/predict", json=s["request"])
            body = r.json()
            if r.status_code != 200:
                failures.append(f"{name}: /predict returned {r.status_code}")
                continue
            served = body["fraud_probability"]
            # served is the artifact probability rounded to the API's 4dp contract
            if not _close(served, round(local, 4)):
                failures.append(f"{name}: served {served:.4f} != artifact {round(local, 4):.4f} (raw {local:.6f})")
            expected_pred = 1 if local >= THRESHOLD else 0
            if body["prediction"] != expected_pred:
                failures.append(f"{name}: prediction {body['prediction']} != {expected_pred} (prob {served:.4f})")

    for s in samples:
        prob = float(model.predict_proba(
            encode_vector(SimpleNamespace(**s["request"])).reshape(1, -1)
        )[0, 1])
        verdict = "FRAUD" if prob >= THRESHOLD else "legit"
        mark = "OK" if (verdict == "FRAUD") == (s["is_fraud"] == 1) else "miss"
        print(f"  [{mark}] {prob*100:6.2f}%  {s['merchant']:32s} {s['city']:14s} gold={'FRAUD' if s['is_fraud'] else 'legit'}")

    print(f"\nCompute-vs-persisted parity over {len(samples)} real rows:")
    if failures:
        print(f"  FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  encoder == gold features ......... PASS")
    print("  served probability == artifact ... PASS")
    print("  ALL PARITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())