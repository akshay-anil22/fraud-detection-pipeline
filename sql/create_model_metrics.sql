-- Week 3: one row per training run, so Week 5 dashboards can chart regressions.
CREATE TABLE IF NOT EXISTS model_metrics (
    id               BIGSERIAL PRIMARY KEY,
    dag_run_id       TEXT,
    model_path       TEXT NOT NULL,
    scale_pos_weight DOUBLE PRECISION NOT NULL,
    n_train          INTEGER NOT NULL,
    n_test           INTEGER NOT NULL,
    metrics          JSONB NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_created ON model_metrics (created_at);