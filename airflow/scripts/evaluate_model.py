"""
Week 3 - evaluate task.

Reloads the saved artifact, rebuilds the SAME stratified split, scores the
held-out test set, writes a metrics JSON, inserts a metrics row into Postgres
(for Week 5 dashboards), and runs a predict_one sanity check.
"""
import json

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import text
import xgboost as xgb

from modeling_common import (
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TABLE,
    TARGET,
    TEST_SIZE,
    get_engine,
)
from train_model import load_feature_frame


def evaluate(dag_run_id: str | None = None):
    df = load_feature_frame()
    X = df[FEATURES]
    y = df[TARGET]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "test_frauds": int((y_test == 1).sum()),
        "test_legit": int((y_test == 0).sum()),
        "random_state": RANDOM_STATE,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    _insert_metrics_row(metrics, y_train, dag_run_id)
    _print_report(metrics)
    _predict_one_sanity(model, X_test, y_test)

    return metrics


def _insert_metrics_row(metrics, y_train, dag_run_id=None):
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO model_metrics "
                "(dag_run_id, model_path, scale_pos_weight, n_train, n_test, metrics) "
                "VALUES "
                "(:dag_run_id, :path, :spw, :ntrain, :ntest, CAST(:metrics AS jsonb))"
            ),
            {
                "dag_run_id": dag_run_id,
                "path": MODEL_PATH,
                "spw": n_neg / n_pos,
                "ntrain": int(len(y_train)),
                "ntest": metrics["test_frauds"] + metrics["test_legit"],
                "metrics": json.dumps(metrics),
            },
        )
    print("Metrics row inserted into postgres.model_metrics")


def _print_report(metrics):
    cm = metrics["confusion_matrix"]
    print("=" * 50)
    print("MODEL EVALUATION (held-out test set)")
    print("=" * 50)
    print(f"  Recall    : {metrics['recall']:.4f}  (caught {cm['tp']}/{metrics['test_frauds']} frauds)")
    print(f"  Precision : {metrics['precision']:.4f}  (of {cm['tp'] + cm['fp']} alarms, {cm['tp']} real)")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Missed    : {cm['fn']} frauds  |  False alarms: {cm['fp']}")
    print("=" * 50)


def _predict_one_sanity(model, X_test, y_test):
    """Feed one known-fraud and one known-legit row; the model should agree."""
    import numpy as np

    fraud_row = np.where(y_test.to_numpy() == 1)[0][0]
    legit_row = np.where(y_test.to_numpy() == 0)[0][0]

    def score(idx):
        return float(model.predict_proba(X_test.iloc[[idx]])[0][1])

    legit_prob, fraud_prob = score(legit_row), score(fraud_row)
    legit_ok = legit_prob < 0.5
    fraud_ok = fraud_prob >= 0.5
    print(f"Sanity check: legit prob={legit_prob:.4f} (expect <0.5) -> {'OK' if legit_ok else 'BAD'}")
    print(f"Sanity check: fraud prob={fraud_prob:.4f} (expect >=0.5) -> {'OK' if fraud_ok else 'BAD'}")
    if not (legit_ok and fraud_ok):
        raise ValueError("predict_one sanity check failed - the artifact misclassifies known labels")


if __name__ == "__main__":
    print(evaluate())