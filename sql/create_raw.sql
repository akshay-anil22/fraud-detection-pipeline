-- Kartik2112 Fraud Detection raw data schema
-- Source: Kaggle kartik2112/fraud-detection (fraudTrain.csv + fraudTest.csv)
--
-- 25 columns = id + 22 data columns + source_split + ingested_at.
-- source_split records which CSV the row came from ('train' | 'test') so the
-- chronological train/test holdout survives into the gold layer.

CREATE TABLE IF NOT EXISTS raw_transactions (
    id                    BIGSERIAL PRIMARY KEY,
    trans_date_trans_time TIMESTAMPTZ NOT NULL,
    cc_num                TEXT NOT NULL,
    merchant              TEXT NOT NULL,
    category              TEXT NOT NULL,
    amt                   DOUBLE PRECISION NOT NULL,
    first                 TEXT,
    last                  TEXT,
    gender                TEXT,
    street                TEXT,
    city                  TEXT,
    state                 TEXT,
    zip                   TEXT,
    lat                   DOUBLE PRECISION,
    long                  DOUBLE PRECISION,
    city_pop              BIGINT,
    job                   TEXT,
    dob                   TEXT,
    trans_num             TEXT NOT NULL,
    unix_time             BIGINT NOT NULL,
    merch_lat             DOUBLE PRECISION,
    merch_long            DOUBLE PRECISION,
    is_fraud              SMALLINT NOT NULL,
    source_split          TEXT NOT NULL,
    ingested_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_transactions_split_is_fraud ON raw_transactions (source_split, is_fraud);
CREATE INDEX IF NOT EXISTS idx_raw_transactions_cc_num ON raw_transactions (cc_num);
CREATE INDEX IF NOT EXISTS idx_raw_transactions_unix_time ON raw_transactions (unix_time);