"""
Extract task: download the Credit Card Fraud Detection dataset from Kaggle
and bulk-load it into PostgreSQL.
"""
import csv
import io
import os

import psycopg2
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine, text

DATASET = "mlg-ulb/creditcardfraud"
TARGET_TABLE = "raw_transactions"

EXPECTED_ROWS = 284_807
EXPECTED_COLUMNS = 31

DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data")
os.makedirs(DATA_DIR, exist_ok=True)
RAW_CSV = os.path.join(DATA_DIR, "creditcard.csv")


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


def download_dataset() -> str:
    """Download (if needed) and return path to the raw CSV.

    By default the cached CSV is reused if present. Set FORCE_REFRESH=1
    to always pull the latest version from the Kaggle API (picks up any
    changes made to the remote dataset).
    """
    force = os.environ.get("FORCE_REFRESH", "").lower() in {"1", "true", "yes"}

    if not force and os.path.exists(RAW_CSV):
        row_count = sum(1 for _ in open(RAW_CSV, encoding="utf-8")) - 1
        if row_count >= EXPECTED_ROWS:
            print(f"Dataset already present: {RAW_CSV} ({row_count} rows)")
            return RAW_CSV

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
    print(f"Download complete: {RAW_CSV}")
    return RAW_CSV


COLUMNS = [
    "time", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
    "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
    "v28", "amount", "class",
]


def load_to_postgres(csv_path: str) -> dict:
    """Load the CSV into PostgreSQL using psycopg2 COPY for speed."""
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    try:
        cur = conn.cursor()

        print(f"Truncating {TARGET_TABLE}...")
        cur.execute(f"TRUNCATE {TARGET_TABLE} RESTART IDENTITY")

        print(f"Loading {csv_path} into {TARGET_TABLE}...")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header

            buffer = io.StringIO()
            row_count = 0
            for row in reader:
                # Strip quotes from Class column and write tab-separated
                row[-1] = row[-1].strip('"')
                buffer.write("\t".join(row) + "\n")
                row_count += 1

                if row_count % 50_000 == 0:
                    buffer.seek(0)
                    cur.copy_expert(
                        f"COPY {TARGET_TABLE} ({', '.join(COLUMNS)}) FROM STDIN WITH (FORMAT text, NULL '')",
                        buffer,
                    )
                    buffer = io.StringIO()
                    print(f"  ... loaded {row_count} rows")

            # Flush remaining rows
            if buffer.tell() > 0:
                buffer.seek(0)
                cur.copy_expert(
                    f"COPY {TARGET_TABLE} ({', '.join(COLUMNS)}) FROM STDIN WITH (FORMAT text, NULL '')",
                    buffer,
                )

        conn.commit()
        print(f"COPY complete: {row_count} rows sent")

        cur.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}")
        stored_rows = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{TARGET_TABLE}'")
        stored_cols = cur.fetchone()[0]

        print(f"Load verified: {stored_rows} rows, {stored_cols} columns in {TARGET_TABLE}")
        return {"rows": stored_rows, "columns": stored_cols}
    finally:
        conn.close()


if __name__ == "__main__":
    path = download_dataset()
    result = load_to_postgres(path)
    print(f"RESULT: rows={result['rows']}, columns={result['columns']}")
