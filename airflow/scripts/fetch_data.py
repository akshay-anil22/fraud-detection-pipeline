"""
Extract task: download the Kartik2112 Fraud Detection dataset from Kaggle
and bulk-load both CSVs into PostgreSQL with a source_split marker.

The dataset ships as a genuinely chronological train/test split:
  fraudTrain.csv  2012-01-01 -> 2013-06-21 12:13:37Z  (source_split='train')
  fraudTest.csv 2013-06-21 12:14:25Z -> 2013-12-31  (source_split='test')
"""
import csv
import io
import os

import psycopg2
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine, text

DATASET = "kartik2112/fraud-detection"
TARGET_TABLE = "raw_transactions"

# (filename, source_split, expected_rows) - pinned from the live dataset.
FILES = [
    ("fraudTrain.csv", "train", 1_296_675),
    ("fraudTest.csv", "test", 555_719),
]

# 22 data columns (the CSV has a leading unnamed index column that is skipped).
COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "first", "last", "gender", "street", "city", "state", "zip", "lat",
    "long", "city_pop", "job", "dob", "trans_num", "unix_time",
    "merch_lat", "merch_long", "is_fraud",
]

EXPECTED_TOTAL_ROWS = 1_852_394  # train 1,296,675 + test 555,719

DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data")
os.makedirs(DATA_DIR, exist_ok=True)


def configure_kaggle_auth():
    """Write the KAGGLE_API_TOKEN env var to ~/.kaggle/access_token."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        return
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    token_file = os.path.join(kaggle_dir, "access_token")
    existing = ""
    if os.path.exists(token_file):
        with open(token_file) as f:
            existing = f.read().strip()
    if existing != token:
        with open(token_file, "w") as f:
            f.write(token)
        print(f"Wrote Kaggle access token to {token_file}")


def row_count(path: str) -> int:
    """Count data rows in a CSV (header excluded)."""
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def download_dataset() -> dict:
    """Download (if needed) both CSVs and return {filename: path}.

    Each file is cached independently; FORCE_REFRESH=1 always pulls a fresh
    copy of the whole dataset.
    """
    force = os.environ.get("FORCE_REFRESH", "").lower() in {"1", "true", "yes"}

    paths = {}
    missing = []
    for filename, _split, expected in FILES:
        path = os.path.join(DATA_DIR, filename)
        paths[filename] = path
        if force or not os.path.exists(path) or row_count(path) < expected:
            missing.append(filename)
        else:
            print(f"Dataset already present: {path} ({row_count(path)} rows)")

    if not missing:
        return paths

    print("Downloading dataset from Kaggle..." + (" (forced refresh)" if force else ""))
    configure_kaggle_auth()
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        DATASET,
        path=DATA_DIR,
        unzip=True,
        quiet=False,
    )
    print(f"Download complete:")
    for filename, path in paths.items():
        print(f"  {path} ({row_count(path)} rows)")
    return paths


def load_to_postgres(csv_path: str, source_split: str, truncate_first: bool = False) -> dict:
    """Load one CSV into PostgreSQL using psycopg2 COPY for speed."""
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    try:
        cur = conn.cursor()

        if truncate_first:
            print(f"Truncating {TARGET_TABLE}...")
            cur.execute(f"TRUNCATE {TARGET_TABLE} RESTART IDENTITY")

        print(f"Loading {csv_path} (split={source_split}) into {TARGET_TABLE}...")

        copy_cols = ", ".join(COLUMNS + ["source_split"])

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header (starts with the unnamed index column)

            buffer = io.StringIO()
            row_count = 0
            for row in reader:
                # Drop the leading unnamed index column, then append split tag.
                data_row = row[1:] + [source_split]
                buffer.write("\t".join(data_row) + "\n")
                row_count += 1

                if row_count % 50_000 == 0:
                    buffer.seek(0)
                    cur.copy_expert(
                        f"COPY {TARGET_TABLE} ({copy_cols}) FROM STDIN WITH (FORMAT text, NULL '')",
                        buffer,
                    )
                    buffer = io.StringIO()
                    print(f"  ... loaded {row_count} rows")

            if buffer.tell() > 0:
                buffer.seek(0)
                cur.copy_expert(
                    f"COPY {TARGET_TABLE} ({copy_cols}) FROM STDIN WITH (FORMAT text, NULL '')",
                    buffer,
                )

        conn.commit()
        print(f"COPY complete: {row_count} rows sent (split={source_split})")
        return {"rows": row_count, "split": source_split}
    finally:
        conn.close()


def verify_load() -> dict:
    """Confirm the total and per-split row counts stored in Postgres."""
    engine = create_engine(
        f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'postgres')}:{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )
    with engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM {TARGET_TABLE}")).scalar()
        by_split = conn.execute(text(
            f"SELECT source_split, COUNT(*) FROM {TARGET_TABLE} GROUP BY source_split ORDER BY source_split"
        )).fetchall()
    return {"total": int(total), "by_split": {s: int(n) for s, n in by_split}}


if __name__ == "__main__":
    paths = download_dataset()
    for filename, split, _expected in FILES:
        load_to_postgres(paths[filename], split, truncate_first=(filename == FILES[0][0]))
    stats = verify_load()
    assert stats["total"] == EXPECTED_TOTAL_ROWS, f"row mismatch: {stats}"
    print(f"RESULT: total rows={stats['total']} by_split={stats['by_split']}")