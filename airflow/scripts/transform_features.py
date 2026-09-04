"""
Transform task: build the ML-ready feature_transactions table from raw_transactions.

All feature math lives in one auditable INSERT ... SELECT. The source is re-read
each run and the target is truncated first, keeping the task idempotent.
"""
import os

from sqlalchemy import create_engine, text

TARGET_TABLE = "feature_transactions"
SOURCE_TABLE = "raw_transactions"

ANCHOR_TS = "2013-09-01 00:00:00+00"  # time=0 maps here (community convention)


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


FEATURE_SQL = f"""
TRUNCATE {TARGET_TABLE} RESTART IDENTITY;

INSERT INTO {TARGET_TABLE}
    (time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
     v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28,
     amount, class,
     tx_datetime, hour_of_day, is_weekend, amount_log, amount_bin,
     v_magnitude, duplicate_flag)
SELECT
    r.time, r.v1, r.v2, r.v3, r.v4, r.v5, r.v6, r.v7, r.v8, r.v9, r.v10,
    r.v11, r.v12, r.v13, r.v14, r.v15, r.v16, r.v17, r.v18, r.v19, r.v20,
    r.v21, r.v22, r.v23, r.v24, r.v25, r.v26, r.v27, r.v28, r.amount, r.class,
    TIMESTAMPTZ '{ANCHOR_TS}' + r.time * INTERVAL '1 second' AS tx_datetime,
    EXTRACT(hour FROM TIMESTAMPTZ '{ANCHOR_TS}' + r.time * INTERVAL '1 second')::int AS hour_of_day,
    CASE WHEN EXTRACT(isodow FROM TIMESTAMPTZ '{ANCHOR_TS}' + r.time * INTERVAL '1 second') IN (6, 7)
         THEN 1 ELSE 0 END AS is_weekend,
    ln(r.amount + 1) AS amount_log,
    CASE WHEN r.amount < 10 THEN 'small'
         WHEN r.amount < 100 THEN 'medium'
         WHEN r.amount < 1000 THEN 'large'
         ELSE 'xl' END AS amount_bin,
    sqrt(r.v1*r.v1 + r.v2*r.v2 + r.v3*r.v3 + r.v4*r.v4 + r.v5*r.v5
       + r.v6*r.v6 + r.v7*r.v7 + r.v8*r.v8 + r.v9*r.v9 + r.v10*r.v10
       + r.v11*r.v11 + r.v12*r.v12 + r.v13*r.v13 + r.v14*r.v14 + r.v15*r.v15
       + r.v16*r.v16 + r.v17*r.v17 + r.v18*r.v18 + r.v19*r.v19 + r.v20*r.v20
       + r.v21*r.v21 + r.v22*r.v22 + r.v23*r.v23 + r.v24*r.v24 + r.v25*r.v25
       + r.v26*r.v26 + r.v27*r.v27 + r.v28*r.v28) AS v_magnitude,
    CASE WHEN COUNT(*) OVER (PARTITION BY r.time, r.v1, r.v2, r.v3, r.v4, r.v5,
           r.v6, r.v7, r.v8, r.v9, r.v10, r.v11, r.v12, r.v13, r.v14, r.v15,
           r.v16, r.v17, r.v18, r.v19, r.v20, r.v21, r.v22, r.v23, r.v24,
           r.v25, r.v26, r.v27, r.v28, r.amount) > 1 THEN 1 ELSE 0 END
         AS duplicate_flag
FROM {SOURCE_TABLE} r;
"""


def build_features():
    engine = get_engine()
    with engine.begin() as conn:  # begin() auto-commits on success
        conn.execute(text(FEATURE_SQL))
    return True


def verify():
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE}")).scalar()
        src_rows = conn.execute(text(f"SELECT COUNT(*) FROM {SOURCE_TABLE}")).scalar()
        cols = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.columns "
            f"WHERE table_name = '{TARGET_TABLE}'"
        )).scalar()
        if rows != src_rows:
            raise ValueError(f"Row mismatch: {TARGET_TABLE}={rows} vs {SOURCE_TABLE}={src_rows}")
    print(f"Transform verified: {rows} rows, {cols} columns in {TARGET_TABLE} (source {src_rows})")
    return {"rows": rows, "columns": cols}


if __name__ == "__main__":
    build_features()
    verify()
    print("RESULT: feature_transactions ready")