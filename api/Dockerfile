FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api/schemas.py api/feature_encoder.py api/main.py api/parity_check.py /app/
COPY api/static /app/static
COPY api/fixtures/samples_dataset.json /app/fixtures/samples_dataset.json

# Bind-mounted at runtime: ./models -> /opt/airflow/models (model artifact)
#                              ./airflow/scripts -> /opt/airflow/scripts (feature contract)
ENV MODEL_PATH=/opt/airflow/models/xgb_fraud_model.json
ENV PYTHONPATH=/app:/opt/airflow/scripts

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]