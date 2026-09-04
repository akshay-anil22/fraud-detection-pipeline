"""
Quality gate: verify the raw data landed in PostgreSQL correctly.
Fails loudly (returns non-zero) if the data does not match expectations.
"""
import os
import sys

from sqlalchemy import create_engine, text

EXPECTED_ROWS = 284_807
EXPECTED_DATA_COLUMNS = 31
EXPECTED_TOTAL_COLUMNS = 33  # 31 data columns + id (serial) + ingested_at (timestamptz)


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
        # 1. Row count
        rows = conn.execute(text("SELECT COUNT(*) FROM raw_transactions")).scalar()
        row_ok = rows == EXPECTED_ROWS
        checks.append(f"Row count: {rows} (expected {EXPECTED_ROWS}) -> {'PASS' if row_ok else 'FAIL'}")
        if not row_ok:
            failures.append(f"row_count: {rows} != {EXPECTED_ROWS}")

        # 2. Column count (31 data + id + ingested_at)
        cols = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = 'raw_transactions'"
        )).scalar()
        col_ok = cols == EXPECTED_TOTAL_COLUMNS
        checks.append(f"Column count: {cols} (expected {EXPECTED_TOTAL_COLUMNS}) -> {'PASS' if col_ok else 'FAIL'}")
        if not col_ok:
            failures.append(f"column_count: {cols} != {EXPECTED_TOTAL_COLUMNS}")

        # 3. No nulls in critical columns
        nulls = conn.execute(text(
            "SELECT COUNT(*) FROM raw_transactions WHERE class IS NULL OR time IS NULL OR amount IS NULL"
        )).scalar()
        null_ok = nulls == 0
        checks.append(f"Nulls in critical columns: {nulls} -> {'PASS' if null_ok else 'FAIL'}")
        if not null_ok:
            failures.append(f"null_count: {nulls}")

        # 4. Class distribution
        dist = conn.execute(text(
            "SELECT class, COUNT(*) FROM raw_transactions GROUP BY class ORDER BY class"
        )).fetchall()
        dist_str = ", ".join(f"class={c}: {n}" for c, n in dist)
        checks.append(f"Class distribution: {dist_str}")

        # 5. Amount sanity: no negatives
        neg_amounts = conn.execute(text(
            "SELECT COUNT(*) FROM raw_transactions WHERE amount < 0"
        )).scalar()
        amount_ok = neg_amounts == 0
        checks.append(f"Negative amounts: {neg_amounts} -> {'PASS' if amount_ok else 'FAIL'}")
        if not amount_ok:
            failures.append(f"negative_amounts: {neg_amounts}")

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
