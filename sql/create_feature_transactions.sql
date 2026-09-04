-- Credit Card Fraud Detection - engineered feature table (ML-ready "gold" layer)
-- Source: raw_transactions (Week 1 ingestion)
--
-- 40 columns = id + 31 data columns + ingested_at + 7 engineered features
--   tx_datetime   : readable timestamp derived from relative `time` (anchored at 2013-09-01 00:00:00 UTC)
--   hour_of_day   : 0-23
--   is_weekend    : 1 if Saturday/Sunday
--   amount_log    : ln(amount + 1) - spreads out small fraud amounts
--   amount_bin    : coarse magnitude bucket (small/medium/large/xl)
--   v_magnitude   : sqrt(sum(v1^2..v28^2)) - PCA vector magnitude signal
--   duplicate_flag: 1 if an identical (time, v1..v28, amount) row appears more than once

CREATE TABLE IF NOT EXISTS feature_transactions (
    id             BIGSERIAL PRIMARY KEY,
    time           DOUBLE PRECISION NOT NULL,
    v1             DOUBLE PRECISION,
    v2             DOUBLE PRECISION,
    v3             DOUBLE PRECISION,
    v4             DOUBLE PRECISION,
    v5             DOUBLE PRECISION,
    v6             DOUBLE PRECISION,
    v7             DOUBLE PRECISION,
    v8             DOUBLE PRECISION,
    v9             DOUBLE PRECISION,
    v10            DOUBLE PRECISION,
    v11            DOUBLE PRECISION,
    v12            DOUBLE PRECISION,
    v13            DOUBLE PRECISION,
    v14            DOUBLE PRECISION,
    v15            DOUBLE PRECISION,
    v16            DOUBLE PRECISION,
    v17            DOUBLE PRECISION,
    v18            DOUBLE PRECISION,
    v19            DOUBLE PRECISION,
    v20            DOUBLE PRECISION,
    v21            DOUBLE PRECISION,
    v22            DOUBLE PRECISION,
    v23            DOUBLE PRECISION,
    v24            DOUBLE PRECISION,
    v25            DOUBLE PRECISION,
    v26            DOUBLE PRECISION,
    v27            DOUBLE PRECISION,
    v28            DOUBLE PRECISION,
    amount         DOUBLE PRECISION NOT NULL,
    class          SMALLINT NOT NULL,
    ingested_at    TIMESTAMPTZ DEFAULT NOW(),
    tx_datetime    TIMESTAMPTZ NOT NULL,
    hour_of_day    SMALLINT NOT NULL,
    is_weekend     SMALLINT NOT NULL,
    amount_log     DOUBLE PRECISION NOT NULL,
    amount_bin     TEXT NOT NULL,
    v_magnitude    DOUBLE PRECISION NOT NULL,
    duplicate_flag SMALLINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feature_transactions_class ON feature_transactions (class);
CREATE INDEX IF NOT EXISTS idx_feature_transactions_hour ON feature_transactions (hour_of_day);
CREATE INDEX IF NOT EXISTS idx_feature_transactions_dup ON feature_transactions (duplicate_flag);