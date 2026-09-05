"""
Quality gate for the feature_transactions (gold) table.

Runs structural + semantic checks, plus the per-split class-imbalance
report that backs the Week 3 scale_pos_weight decision. Per-split only:
scale_pos_weight comes from the TRAIN split, never a blended number.
"""
import os
import sys

from sqlalchemy import create_engine, text

SOURCE_TABLE = "raw_transactions"
TARGET_TABLE = "feature_transactions"
EXPECTED_TOTAL_COLUMNS = 19

# Precommitted NULL policy (decided before the data scan):
#   age / distance_km NULL share per split:
#     > 1.0% -> hard failure (parse regression or source drift)
#     0.5-1.0% -> warning only
NULL_FAIL_FRACTION = 0.010
NULL_WARN_FRACTION = 0.005


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
        # 1. Row count matches source (1:1, idempotent rebuild)
        rows = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE}")).scalar()
        src_rows = conn.execute(text(f"SELECT COUNT(*) FROM {SOURCE_TABLE}")).scalar()
        ok = rows == src_rows
        checks.append(f"Row count: {rows:,} (source {src_rows:,}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"row_count: {rows} != {src_rows}")

        # 2. Per-split row counts
        by_split = conn.execute(text(
            f"SELECT source_split, COUNT(*) FROM {TARGET_TABLE} GROUP BY source_split ORDER BY source_split"
        )).fetchall()
        split_rows = {s: int(n) for s, n in by_split}
        checks.append(f"Per-split rows: {split_rows}")

        # 3. Column count (schema drift guard)
        cols = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.columns "
            f"WHERE table_name = '{TARGET_TABLE}'"
        )).scalar()
        ok = cols == EXPECTED_TOTAL_COLUMNS
        checks.append(f"Column count: {cols} (expected {EXPECTED_TOTAL_COLUMNS}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"column_count: {cols} != {EXPECTED_TOTAL_COLUMNS}")

        # 4. Hard-null features must stay null-free
        hard_nulls = conn.execute(text(
            f"SELECT COUNT(*) FROM {TARGET_TABLE} "
            "WHERE hour_of_day IS NULL OR is_weekend IS NULL OR amount_log IS NULL "
            "OR city_pop_bin IS NULL OR is_new_merchant_for_card IS NULL"
        )).scalar()
        ok = hard_nulls == 0
        checks.append(f"Nulls in hard features: {hard_nulls} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"hard_null_features: {hard_nulls}")

        # 5. NULL policy for age / distance_km (per split, precommitted tiers)
        null_stats = conn.execute(text(
            f"SELECT source_split, "
            "COUNT(*) FILTER (WHERE age IS NULL), "
            "COUNT(*) FILTER (WHERE distance_km IS NULL), "
            "COUNT(*) "
            f"FROM {TARGET_TABLE} GROUP BY source_split"
        )).fetchall()
        for split, age_n, dist_n, total in null_stats:
            for name, n in (("age", int(age_n)), ("distance_km", int(dist_n))):
                frac = int(n) / int(total) if total else 0.0
                tier = "PASS" if frac <= NULL_WARN_FRACTION else ("WARN" if frac <= NULL_FAIL_FRACTION else "FAIL")
                checks.append(f"NULL[{split}] {name}: {int(n)} ({frac:.3%}) -> {tier}")
                if frac > NULL_FAIL_FRACTION:
                    failures.append(f"null_{name}_{split}: {n}/{total}")
                elif frac > NULL_WARN_FRACTION:
                    warnings.append(f"{name} null share {frac:.3%} in split={split} exceeds WARN band")

        # 6. Range checks
        bad_hour = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE hour_of_day < 0 OR hour_of_day > 23")).scalar()
        ok = bad_hour == 0
        checks.append(f"hour_of_day out of [0,23]: {bad_hour} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"hour_range: {bad_hour}")

        bad_flag = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE is_weekend NOT IN (0,1) OR city_pop_bin NOT IN (0,1,2,3) OR is_new_merchant_for_card NOT IN (0,1)")).scalar()
        ok = bad_flag == 0
        checks.append(f"flag/bucket out of domain: {bad_flag} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"flag_range: {bad_flag}")

        bad_age = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE age IS NOT NULL AND age <= 0")).scalar()
        ok = bad_age == 0
        checks.append(f"Non-positive ages: {bad_age} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"age_range: {bad_age}")

        bad_dist = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE distance_km IS NOT NULL AND (distance_km < 0 OR distance_km > 20000)")).scalar()
        ok = bad_dist == 0
        checks.append(f"distance_km outside [0, 20000]: {bad_dist} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"distance_range: {bad_dist}")

        # 7. amount_log == ln(amt + 1) (SQL/python parity spot check)
        spot = conn.execute(text(
            f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE NOT (amount_log = ln(amt + 1))"
        )).scalar()
        ok = spot == 0
        checks.append(f"amount_log == ln(amt+1) mismatches: {spot} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"amount_log_parity: {spot}")

        # 8. Per-split class-imbalance report (formal; scale_pos_weight from TRAIN only)
        checks.append("=" * 66)
        checks.append("PER-SPLIT CLASS-IMBALANCE REPORT (gold layer)")
        dist = conn.execute(text(
            f"SELECT source_split, is_fraud, COUNT(*) FROM {TARGET_TABLE} "
            "GROUP BY source_split, is_fraud ORDER BY source_split, is_fraud"
        )).fetchall()
        split_rates = {}
        for split in ("train", "test"):
            s_rows = split_rows.get(split, 0)
            if not s_rows:
                continue
            s_fraud = sum(int(n) for sp, lb, n in dist if sp == split and lb == 1)
            rate = s_fraud / s_rows
            split_rates[split] = rate
            checks.append(f"  split={split:<5} legit={s_rows - s_fraud:,} fraud={s_fraud:,} rate={rate:.4%}")
        if "train" not in split_rates:
            failures.append("no_train_split_present")
        if split_rates:
            train_rate = split_rates["train"]
            s_fraud = sum(int(n) for sp, lb, n in dist if sp == "train" and lb == 1)
            s_rows = split_rows["train"]
            spw = (s_rows - s_fraud) / s_fraud if s_fraud else 0.0
            checks.append(f"  -> train scale_pos_weight ~ {spw:.1f} (legit/fraud = TRAIN split only)")
            if not (0.002 < train_rate < 0.02):
                warnings.append(f"train fraud rate {train_rate:.4%} outside expected 0.2-2% band")
        checks.append("=" * 66)

        # 9. Cross-check vs raw level (gold rates must equal raw rates)
        raw_rates = conn.execute(text(
            f"SELECT source_split, COUNT(*) FILTER (WHERE is_fraud=1)::float / COUNT(*) "
            f"FROM {SOURCE_TABLE} GROUP BY source_split"
        )).fetchall()
        for split, raw_rate in raw_rates:
            gold_rate = split_rates.get(split)
            if gold_rate is not None and abs(gold_rate - raw_rate) > 1e-9:
                failures.append(f"rate_drift_{split}: gold {gold_rate:.6f} vs raw {raw_rate:.6f}")
        checks.append("Gold vs raw fraud rates cross-checked -> PASS" if not failures or all(
            "rate_drift" not in f for f in failures
        ) else "Gold vs raw fraud rates cross-checked -> FAIL")

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