"""Week 4 - Pydantic contracts for the /predict endpoint (kartik2112 raw form).

The request carries the RAW kartik2112 fields the encoder needs to derive the
Week 3 feature set (hour_of_day, is_weekend, amount_log, age, distance_km,
city_pop_bin, is_new_merchant_for_card, category_target):

  time    - seconds since 2013-09-01T00:00:00Z (same time anchor the Week 2 SQL
            transform used: tx_datetime = TIMESTAMPTZ '2013-09-01 00:00:00+00'
            + time * INTERVAL '1 second').
  dob     - cardholder birth date (YYYY-MM-DD). Absent/unparseable -> the age
            is imputed from the persisted training median (mirrors the gold
            table NULL policy).
  lat/long + merch_lat/merch_long - cardholder + merchant coordinates.
            Any missing coordinate -> distance_km is filled with the persisted
            training fill value.
  is_new_merchant_for_card - needs card-merchant *history* to compute (like the
            old duplicate_flag), so a caller that knows it supplies it, else 0.
            The demo "Load random sample" always carries the gold value.

Unknown fields are rejected (extra="forbid") so a stale v1-v28 payload gets a
loud 422 instead of a silent wrong prediction.
"""
from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: float = Field(..., ge=0, description="Seconds since 2013-09-01T00:00:00Z (Week 2 anchor)")
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    category: str = Field(..., description="Card category (one of the 14, e.g. grocery_pos)")
    dob: str | None = Field(None, description="Cardholder birth date YYYY-MM-DD (imputed if absent)")
    city_pop: float = Field(..., ge=0, description="Cardholder city population (binned at encode time)")
    lat: float | None = Field(None, description="Cardholder latitude")
    long: float | None = Field(None, description="Cardholder longitude")
    merch_lat: float | None = Field(None, description="Merchant latitude")
    merch_long: float | None = Field(None, description="Merchant longitude")
    is_new_merchant_for_card: int = Field(0, ge=0, le=1, description="1 if first time this card uses this merchant (optional)")


class PredictionResponse(BaseModel):
    """Prediction for a single transaction."""
    fraud_probability: float = Field(..., ge=0, le=1, description="P(fraud) from the model")
    prediction: int = Field(..., ge=0, le=1, description="1 if fraud_probability >= threshold, else 0")
    latency_ms: float = Field(..., ge=0, description="Wall-clock time for encode + predict")