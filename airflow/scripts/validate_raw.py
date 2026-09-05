"""
Quality gate: verify the raw data landed in PostgreSQL correctly.
Fails loudly (returns non-zero) if the data does not match expectations.

Dataset: Kartik2112 fraud-detection (chronological train/test split).
"""
import os
import sys

from sqlalchemy import create_engine, text

# Pinned from the live dataset (verified on branch swap-kartik2112).
EXPECTED_ROWS = {"train": 1_296_675, "test": 555_719}
EXPECTED_TOTAL_ROWS = 1_852_394
EXPECTED_DATA_COLUMNS = 22
EXPECTED_TOTAL_COLUMNS = 25  # 22 data + id + source_split + ingested_at


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
    failures = []

    with engine.connect() as conn:
        # 1. Total row count
        total = conn.execute(text("SELECT COUNT(*) FROM raw_transactions")).scalar()
        total_ok = total == EXPECTED_TOTAL_ROWS
        checks.append(f"Total rows: {total:,} (expected {EXPECTED_TOTAL_ROWS:,}) -> {'PASS' if total_ok else 'FAIL'}")
        if not total_ok:
            failures.append(f"total_rows: {total} != {EXPECTED_TOTAL_ROWS}")

        # 2. Per-split row counts
        by_split = conn.execute(text(
            "SELECT source_split, COUNT(*) FROM raw_transactions GROUP BY source_split"
        )).fetchall()
        split_map = {s: int(n) for s, n in by_split}
        for split in ("train", "test"):
            expected = EXPECTED_ROWS[split]
            actual = split_map.get(split, 0)
            ok = actual == expected
            checks.append(f"  split={split}: {actual:,} rows (expected {expected:,}) -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{split}_rows: {actual} != {expected}")

        # 3. Column count (22 data + id + source_split + ingested_at)
        cols = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = 'raw_transactions'"
        )).scalar()
        col_ok = cols == EXPECTED_TOTAL_COLUMNS
        checks.append(f"Column count: {cols} (expected {EXPECTED_TOTAL_COLUMNS}) -> {'PASS' if col_ok else 'FAIL'}")
        if not col_ok:
            failures.append(f"column_count: {cols} != {EXPECTED_TOTAL_COLUMNS}")

        # 4. No nulls in critical columns
        nulls = conn.execute(text(
            "SELECT COUNT(*) FROM raw_transactions "
            "WHERE trans_date_trans_time IS NULL OR cc_num IS NULL OR merchant IS NULL "
            "OR category IS NULL OR amt IS NULL OR unix_time IS NULL OR is_fraud IS NULL"
        )).scalar()
        null_ok = nulls == 0
        checks.append(f"Nulls in critical columns: {nulls} -> {'PASS' if null_ok else 'FAIL'}")
        if not null_ok:
            failures.append(f"null_count: {nulls}")

        # 5. Amounts positive
        neg = conn.execute(text("SELECT COUNT(*) FROM raw_transactions WHERE amt <= 0")).scalar()
        amt_ok = neg == 0
        checks.append(f"Non-positive amounts: {neg} -> {'PASS' if amt_ok else 'FAIL'}")
        if not amt_ok:
            failures.append(f"non_positive_amounts: {neg}")

        # 6. Temporal split invariant: train's latest <= test's earliest
        ranges = conn.execute(text(
            "SELECT source_split, MIN(unix_time), MAX(unix_time) "
            "FROM raw_transactions GROUP BY source_split"
        )).fetchall()
        range_map = {s: (int(a), int(b)) for s, a, b in ranges}
        train_hi = range_map.get("train", (0, 0))[1]
        test_lo = range_map.get("test", (0, 0))[0]
        chronological = train_hi <= test_lo
        checks.append(f"Temporal split invariant: train.max={train_hi:,} <= test.min={test_lo:,} -> {'PASS' if chronological else 'FAIL'}")
        if not chronological:
            failures.append(f"temporal_split: train.max {train_hi} > test.min {test_lo}")

        # 7. Per-split class distribution (formal imbalance picture at raw level)
        checks.append("=" * 64)
        checks.append("PER-SPLIT CLASS DISTRIBUTION (raw level)")
        dist = conn.execute(text(
            "SELECT source_split, is_fraud, COUNT(*) "
            "FROM raw_transactions GROUP BY source_split, is_fraud ORDER BY source_split, is_fraud"
        )).fetchall()
        for split, label, n in dist:
            checks.append(f"  split={split:<5} class={label}: {int(n):,}")
        if dist:
            for split in ("train", "test"):
                s_rows = split_map.get(split, 0)
                if not s_rows:
                    continue
                s_fraud = sum(int(n) for sp, lb, n in dist if sp == split and lb == 1)
                checks.append(f"  split={split:<5} fraud rate: {s_fraud/s_rows:.4%}  ({s_fraud:,} fraud / {s_rows:,})")
        checks.append("=" * 64)

    return checks, failures


if __name__ == "__main__":
    checks, failures = run_checks()
    print("\n".join(checks))
    if failures:
        print(f"\nVALIDATION FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll quality checks PASSED.")
    sys.exit(0)