"""
Extract task: download the Credit Card Fraud Detection dataset from Kaggle
and bulk-load it into PostgreSQL.
"""
import os

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine, text

DATASET = "mlg-ulb/creditcardfraud"
TARGET_TABLE = "raw_transactions"

# Expected dataset dimensions for validation
EXPECTED_ROWS = 284_807
EXPECTED_COLUMNS = 31

DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data")
os.makedirs(DATA_DIR, exist_ok=True)
RAW_CSV = os.path.join(DATA_DIR, "creditcard.csv")


def configure_kaggle_auth():
    """Make the KAGGLE_API_TOKEN env var usable by the kaggle CLI.

    Newer token format (KGAT_...) is read by the client from
    ~/.kaggle/access_token. If only the env var is present, write it there.
    """
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
    """Download (if needed) and return path to the raw CSV."""
    if os.path.exists(RAW_CSV):
        row_count = sum(1 for _ in open(RAW_CSV, encoding="utf-8")) - 1
        if row_count >= EXPECTED_ROWS:
            print(f"Dataset already present: {RAW_CSV} ({row_count} rows)")
            return RAW_CSV

    print("Downloading dataset from Kaggle...")
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


def load_to_postgres(csv_path: str) -> dict:
    """Load the CSV into PostgreSQL and return row/column counts."""
    from sqlalchemy import inspect

    POSTGRES_USER = os.environ["POSTGRES_USER"]
    POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
    POSTGRES_DB = os.environ["POSTGRES_DB"]
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")

    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Class column arrives as quoted string in some dumps; coerce to int
    df["Class"] = df["Class"].astype(int)
    df["Time"] = df["Time"].astype(float)
    df["Amount"] = df["Amount"].astype(float)

    # Use COPY-style bulk insert via to_sql (fastest path with pandas)
    print(f"Loading {len(df)} rows into {TARGET_TABLE}...")
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {TARGET_TABLE} RESTART IDENTITY"))
        df.to_sql(
            TARGET_TABLE,
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=10_000,
        )

    with engine.connect() as conn:
        stored_rows = conn.execute(
            text(f"SELECT COUNT(*) FROM {TARGET_TABLE}")
        ).scalar()

    print(f"Load complete: {stored_rows} rows in {TARGET_TABLE}")
    return {"rows": stored_rows, "columns": len(df.columns)}


if __name__ == "__main__":
    path = download_dataset()
    result = load_to_postgres(path)
    print(f"RESULT: rows={result['rows']}, columns={result['columns']}")
