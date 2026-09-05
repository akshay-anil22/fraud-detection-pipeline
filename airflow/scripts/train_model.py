"""
Week 3 - train task.

Reads feature_transactions, does a stratified train/test split, fits XGBoost
with scale_pos_weight derived from the ACTUAL train split (not a hardcoded
number), and saves the model artifact + feature importances.

Split and model both use the fixed RANDOM_STATE so results reproduce exactly.
"""
import json
import os
import time

import pandas as pd
import xgboost as xgb
from sqlalchemy import text
from sklearn.model_selection import train_test_split

from modeling_common import (
    FEATURES,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TABLE,
    TARGET,
    TEST_SIZE,
    XGB_PARAMS,
    get_engine,
)


def load_feature_frame() -> pd.DataFrame:
    """Read the gold table and prepare a numeric model-ready frame."""
    engine = get_engine()
    cols = ", ".join(FEATURES + [TARGET])
    with engine.connect() as conn:
        res = conn.execute(text(f"SELECT {cols} FROM {TABLE}"))
        rows = res.fetchall()
        df = pd.DataFrame(rows, columns=res.keys() if res.keys() else FEATURES + [TARGET])
    engine.dispose()
    return df


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_feature_frame()
    X = df[FEATURES]
    y = df[TARGET]

    # Stratified split: at 0.17% fraud rate a plain shuffle can strand all
    # fraud on one side. stratification preserves the 0:1 ratio in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"Train split: {len(X_train)} rows (neg={n_neg}, pos={n_pos})")
    print(f"Test split : {len(X_test)} rows")
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    t0 = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    model.save_model(MODEL_PATH)
    print(f"Model artifact saved: {MODEL_PATH}")

    importances = dict(zip(FEATURES, model.feature_importances_.tolist()))
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(importances, f, indent=2)
    top5 = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("Top-5 features by importance:")
    for name, val in top5:
        print(f"  {name}: {val:.4f}")

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_neg": n_neg,
        "n_pos": n_pos,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "train_seconds": round(elapsed, 2),
        "model_path": MODEL_PATH,
    }


if __name__ == "__main__":
    print(train())