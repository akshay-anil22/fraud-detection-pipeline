"""
Quality gate for the feature_transactions (gold) table.

Runs sanity checks plus a formal class-imbalance report used as the
baseline for the Week 3 XGBoost `scale_pos_weight` decision.
"""
import os
import sys

from sqlalchemy import create_engine, text

SOURCE_TABLE = "raw_transactions"
TARGET_TABLE = "feature_transactions"
EXPECTED_TOTAL_COLUMNS = 40  # id + 31 data + ingested_at + 7 features
BASELINE_FRAUD_RATE = 0.0017  # ~0.17% fraud (284315 legit / 492 fraud)
RATE_WARN_FACTOR = 10.0  # warn if fraud rate drifts >10x from baseline


def get_engine():
    POSTGRES_USER = os.environ["POSTGRES_USER"]
    POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
    POSTGRES_DB = os.environ["POSTGRES_DB"]
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


def run_checks():
    engine = get_engine()
    checks = []
    warnings = []
    failures = []

    with engine.connect() as conn:
        # 1. Row count matches source
        rows = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE}")).scalar()
        src_rows = conn.execute(text(f"SELECT COUNT(*) FROM {SOURCE_TABLE}")).scalar()
        ok = rows == src_rows
        checks.append(f"Row count: {rows} (source {src_rows}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"row_count: {rows} != {src_rows}")

        # 2. Column count (schema drift guard)
        cols = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.columns "
            f"WHERE table_name = '{TARGET_TABLE}'"
        )).scalar()
        ok = cols == EXPECTED_TOTAL_COLUMNS
        checks.append(f"Column count: {cols} (expected {EXPECTED_TOTAL_COLUMNS}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"column_count: {cols} != {EXPECTED_TOTAL_COLUMNS}")

        # 3. No nulls in critical feature columns
        nulls = conn.execute(text(
            f"SELECT COUNT(*) FROM {TARGET_TABLE} "
            "WHERE hour_of_day IS NULL OR amount_log IS NULL "
            "OR v_magnitude IS NULL OR duplicate_flag IS NULL"
        )).scalar()
        ok = nulls == 0
        checks.append(f"Nulls in critical features: {nulls} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"null_features: {nulls}")

        # 4. amount_log finite (no NaN/Inf from ln(0) edge cases)
        nonfinite = conn.execute(text(
            f"SELECT COUNT(*) FROM {TARGET_TABLE} "
            "WHERE amount_log = 'Infinity' OR amount_log = '-Infinity' "
            "OR amount_log != amount_log"
        )).scalar()
        ok = nonfinite == 0
        checks.append(f"Non-finite amount_log: {nonfinite} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"nonfinite_amount_log: {nonfinite}")

        # 5. hour_of_day within 0-23
        bad_hours = conn.execute(text(
            f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE hour_of_day < 0 OR hour_of_day > 23"
        )).scalar()
        ok = bad_hours == 0
        checks.append(f"hour_of_day out of range: {bad_hours} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"hour_of_day_range: {bad_hours}")

        # 6. Duplicate/identical-row report (card-testing signal)
        dups = conn.execute(text(
            f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE duplicate_flag = 1"
        )).scalar()
        checks.append(f"Duplicate (identical) rows flagged: {dups}")

        # 7. Class-imbalance report (formal)
        dist = conn.execute(text(
            f"SELECT class, COUNT(*) FROM {TARGET_TABLE} GROUP BY class ORDER BY class"
        )).fetchall()
        dist_map = {int(c): int(n) for c, n in dist}
        legit = dist_map.get(0, 0)
        fraud = dist_map.get(1, 0)
        total = legit + fraud
        fraud_rate = fraud / total if total else 0.0
        checks.append("=" * 46)
        checks.append("CLASS-IMBALANCE REPORT")
        checks.append(f"  legit (class=0): {legit}")
        checks.append(f"  fraud (class=1): {fraud}")
        checks.append(f"  fraud rate     : {fraud_rate:.4%}  ({1/fraud_rate:.1f} per 1 fraud)" if fraud else "  fraud rate: 0.00%")
        checks.append(f"  -> use scale_pos_weight ~ {legit/fraud:.1f} in Week 3 XGBoost" if fraud else "  -> no fraud rows present!")
        if not fraud:
            failures.append("no_fraud_rows")
        # early-warning: ratio drift
        if fraud and fraud_rate > BASELINE_FRAUD_RATE * RATE_WARN_FACTOR:
            warnings.append(
                f"fraud rate {fraud_rate:.4%} is >{RATE_WARN_FACTOR}x baseline "
                f"{BASELINE_FRAUD_RATE:.4%} - source data may have changed"
            )
        checks.append("=" * 46)

    return checks, warnings, failures


if __name__ == "__main__":
    checks, warnings, failures = run_checks()
    print("\n".join(checks))
    for w in warnings:
        print(f"WARNING: {w}")
    if failures:
        print(f"\nVALIDATION FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll quality checks PASSED.")
    sys.exit(0)