"""
Transform task: build the ML-ready feature_transactions table from raw_transactions.

All feature math lives in one auditable INSERT ... SELECT. The source is
re-read each run and the target is truncated first, keeping the task
idempotent. The numeric feature contract lives in modeling_common.py.

Target encoding of `category` is deliberately NOT done here - the transform
stays pure SQL. It is fit on the train split only, at train time, and
serialized to category_target_map.json (see modeling_common.CATEGORY_MAP_PATH).
"""
import os

from sqlalchemy import create_engine, text

from modeling_common import AGE_MAX_GUARD, HAVERSINE_RADIUS_KM, POP_BIN_EDGES

TARGET_TABLE = "feature_transactions"
SOURCE_TABLE = "raw_transactions"


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


# Mirrors the Python guards in modeling_common (fixed, documented constants).
_POP_CASE = " ".join(
    f"WHEN city_pop < {edge} THEN {code} " for code, edge in enumerate(POP_BIN_EDGES)
)

FEATURE_SQL = f"""
TRUNCATE {TARGET_TABLE} RESTART IDENTITY;

INSERT INTO {TARGET_TABLE}
    (trans_num, source_split, unix_time, trans_date_trans_time, is_fraud,
     cc_num, merchant, category, amt, city_pop,
     hour_of_day, is_weekend, amount_log, age, distance_km,
     city_pop_bin, is_new_merchant_for_card)
SELECT
    r.trans_num,
    r.source_split,
    r.unix_time,
    r.trans_date_trans_time,
    r.is_fraud,
    r.cc_num,
    r.merchant,
    r.category,
    r.amt,
    r.city_pop,
    EXTRACT(hour FROM r.trans_date_trans_time)::smallint AS hour_of_day,
    CASE WHEN EXTRACT(isodow FROM r.trans_date_trans_time) IN (6, 7)
         THEN 1 ELSE 0 END::smallint AS is_weekend,
    ln(r.amt + 1) AS amount_log,
    -- age guard: NULL only for implausibly-old rows (parse-safe: DOB is
    -- validated non-null, well-formed in this dataset). Under-18 kept.
    CASE WHEN EXTRACT(year FROM age(r.trans_date_trans_time, r.dob::date)) > {AGE_MAX_GUARD}
         THEN NULL
         ELSE EXTRACT(year FROM age(r.trans_date_trans_time, r.dob::date))::double precision
    END AS age,
    -- haversine in km, NULL when any coordinate is missing.
    CASE WHEN r.lat IS NULL OR r.long IS NULL
              OR r.merch_lat IS NULL OR r.merch_long IS NULL
         THEN NULL
         ELSE 2 * {HAVERSINE_RADIUS_KM} * asin(sqrt(
                sin(radians(r.merch_lat - r.lat)/2)^2
                + cos(radians(r.lat)) * cos(radians(r.merch_lat))
                  * sin(radians(r.merch_long - r.long)/2)^2
              ))::double precision
    END AS distance_km,
    CASE {_POP_CASE} ELSE {len(POP_BIN_EDGES)} END::smallint AS city_pop_bin,
    CASE WHEN row_number() OVER (PARTITION BY r.cc_num, r.merchant ORDER BY r.unix_time) = 1
         THEN 1 ELSE 0 END::smallint AS is_new_merchant_for_card
FROM {SOURCE_TABLE} r;
"""


def build_features():
    engine = get_engine()
    with engine.begin() as conn:
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