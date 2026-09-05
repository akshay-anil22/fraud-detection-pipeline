-- Kartik2112 Fraud Detection - engineered feature table (ML-ready "gold" layer)
-- Source: raw_transactions (chronological train/test split)
--
-- 19 columns: provenance + label + raw audit + 7 engineered numeric features.
-- (category is kept RAW in gold; target-encoding is fit at train time and
--  applied through the stored category_target_map.json artifact.)
--
-- Engineered features:
--   hour_of_day            : 0-23 (EXTRACT from trans_date_trans_time)
--   is_weekend             : 1 if Saturday/Sunday
--   amount_log             : ln(amt + 1)
--   age                    : whole calendar years vs dob (NULL only >110)
--   distance_km            : haversine(card home, merchant) in km (NULL if coord missing)
--   city_pop_bin           : fixed buckets 0/1/2/3 (edges 10k/100k/1M)
--   is_new_merchant_for_card: 1 if first occurrence of (cc_num, merchant)

CREATE TABLE IF NOT EXISTS feature_transactions (
    id                       BIGSERIAL PRIMARY KEY,
    trans_num                TEXT NOT NULL,
    source_split             TEXT NOT NULL,
    unix_time                BIGINT NOT NULL,
    trans_date_trans_time    TIMESTAMPTZ NOT NULL,
    is_fraud                 SMALLINT NOT NULL,
    cc_num                   TEXT NOT NULL,
    merchant                 TEXT NOT NULL,
    category                 TEXT NOT NULL,
    amt                      DOUBLE PRECISION NOT NULL,
    city_pop                 BIGINT NOT NULL,
    hour_of_day              SMALLINT NOT NULL,
    is_weekend               SMALLINT NOT NULL,
    amount_log               DOUBLE PRECISION NOT NULL,
    age                      DOUBLE PRECISION,
    distance_km              DOUBLE PRECISION,
    city_pop_bin             SMALLINT NOT NULL,
    is_new_merchant_for_card SMALLINT NOT NULL,
    ingested_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feature_transactions_split ON feature_transactions (source_split);
CREATE INDEX IF NOT EXISTS idx_feature_transactions_split_fraud ON feature_transactions (source_split, is_fraud);
CREATE INDEX IF NOT EXISTS idx_feature_transactions_cc_merchant ON feature_transactions (cc_num, merchant, unix_time);