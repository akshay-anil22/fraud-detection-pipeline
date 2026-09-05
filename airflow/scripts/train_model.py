"""
Week 3 - train task, kartik2112 temporal holdout.

Trains on source_split='train' (2012-01-01 -> 2013-06-21), evaluates on
source_split='test' (2013-06-21 -> 2013-12-31) - the dataset's own
chronological split, so no random train_test_split is used.

Category target encoding is fit on the TRAIN split only and persisted to
category_target_map.json; imputation stats come from the TRAIN split only and
persist to imputation_values.json. evaluate_model and the serving encoder read
the SAME artifacts, so train/evaluate/serve cannot drift.
"""
import json
import os
import time

import pandas as pd
import xgboost as xgb
from sqlalchemy import text

from modeling_common import (
    AGE_MAX_GUARD,
    CATEGORY_MAP_PATH,
    DISTANCE_FILL,
    FEATURES,
    FEATURE_IMPORTANCE_PATH,
    IMPUTATION_PATH,
    MODEL_PATH,
    MODELS_DIR,
    POP_BIN_EDGES,
    RANDOM_STATE,
    TABLE,
    TARGET,
    XGB_PARAMS,
    get_engine,
)

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
SMOOTHING_M = 30.0  # m-estimate smoothing for the target encoding


def load_raw_feature_frame(split: str) -> pd.DataFrame:
    """Gold rows for one source_split, before category encoding/imputation."""
    raw_cols = [c for c in FEATURES if c != "category_target"]
    engine = get_engine()
    cols = ", ".join(raw_cols + [TARGET, "category", "source_split"])
    with engine.connect() as conn:
        res = conn.execute(text(
            f"SELECT {cols} FROM {TABLE} WHERE source_split = '{split}'"
        ))
        df = pd.DataFrame(res.fetchall(), columns=res.keys())
    engine.dispose()
    df["category_target"] = 0.0  # placeholder; replaced by apply_category_map
    return df


def fit_category_map(df_train: pd.DataFrame) -> dict:
    """Smoothed target mean per category, fit on TRAIN only."""
    group = df_train.groupby("category")[TARGET].agg(["mean", "count"])
    global_mean = float(df_train[TARGET].mean())
    m = SMOOTHING_M
    enc = {}
    for cat, row in group.iterrows():
        enc[cat] = float((row["count"] * row["mean"] + m * global_mean) / (row["count"] + m))
    enc["fallback"] = global_mean
    return enc


def apply_category_map(df: pd.DataFrame, cat_map: dict) -> pd.DataFrame:
    mapping = {k: v for k, v in cat_map.items() if k != "fallback"}
    df["category_target"] = df["category"].map(mapping).fillna(cat_map["fallback"])
    return df


def fit_imputation(df_train: pd.DataFrame) -> dict:
    """Statistics from the TRAIN split only (serve reads these, never hardcodes)."""
    return {
        "age_median": float(df_train["age"].median()) if df_train["age"].notna().any() else None,
        "distance_fill": DISTANCE_FILL,
        "pop_bin_edges": POP_BIN_EDGES,
        "age_max_guard": AGE_MAX_GUARD,
    }


def apply_imputation(df: pd.DataFrame, imp: dict) -> pd.DataFrame:
    if imp.get("age_median") is not None:
        df["age"] = df["age"].fillna(imp["age_median"])
    df["distance_km"] = df["distance_km"].fillna(imp["distance_fill"])
    return df


def build_matrices(cat_map: dict, imp: dict):
    """(X_train, y_train, X_test, y_test) built through the identical artifacts
    that evaluate_model and the serving encoder consume."""
    frames = []
    for split in (TRAIN_SPLIT, TEST_SPLIT):
        df = load_raw_feature_frame(split)
        df = apply_category_map(df, cat_map)
        df = apply_imputation(df, imp)
        assert list(df[FEATURES].columns) == FEATURES, "feature order drift"
        nans = int(df[FEATURES].isna().sum().sum())
        assert nans == 0, f"NaN survived imputation ({nans})"
        frames.append(df)

    train, test = frames
    return (
        train[FEATURES], train[TARGET].astype(int),
        test[FEATURES], test[TARGET].astype(int),
    )


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_raw = load_raw_feature_frame(TRAIN_SPLIT)
    cat_map = fit_category_map(train_raw)
    imp = fit_imputation(train_raw)

    # Persist BEFORE building matrices: evaluate/serve consume these files.
    with open(CATEGORY_MAP_PATH, "w") as f:
        json.dump(cat_map, f, indent=2)
    with open(IMPUTATION_PATH, "w") as f:
        json.dump(imp, f, indent=2)
    print(f"Saved {CATEGORY_MAP_PATH} ({len(cat_map) - 1} categories + fallback)")
    print(f"Saved {IMPUTATION_PATH} ({imp})")

    X_train, y_train, X_test, y_test = build_matrices(cat_map, imp)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"Train split ({TRAIN_SPLIT}): {len(X_train):,} rows (neg={n_neg:,}, pos={n_pos:,})")
    print(f"Test split  ({TEST_SPLIT}): {len(X_test):,} rows")
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    t0 = time.time()
    model = xgb.XGBClassifier(**params)
    # Fixed n_estimators (no early stopping): keeps the saved artifact's tree
    # count identical to what evaluation/serve predict - no iteration-range
    # ambiguity between fit-time best_iteration and load-time predictions.
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
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
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_neg": n_neg,
        "n_pos": n_pos,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "train_seconds": round(elapsed, 2),
        "model_path": MODEL_PATH,
    }


if __name__ == "__main__":
    print(train())