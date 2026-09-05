"""Week 4 - Pydantic contracts for the /predict endpoint."""
from pydantic import BaseModel, Field, ConfigDict


class TransactionRequest(BaseModel):
    """Raw transaction as provided to the API.

    time is seconds elapsed since 2013-09-01T00:00:00Z (same anchor the Week 2
    SQL transform used). All v-features are the PCA-transformed card features.
    duplicate_flag is optional: it needs transaction *history* to compute, so a
    caller that knows the value can supply it; otherwise it defaults to 0.
    """
    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422, not silent

    time: float = Field(..., ge=0, description="Seconds since 2013-09-01T00:00:00Z")
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    duplicate_flag: int = Field(0, ge=0, le=1, description="1 if an identical tx was seen before (optional)")
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float


class PredictionResponse(BaseModel):
    """Prediction for a single transaction."""
    fraud_probability: float = Field(..., ge=0, le=1, description="P(fraud) from the model")
    prediction: int = Field(..., ge=0, le=1, description="1 if fraud_probability >= threshold, else 0")
    latency_ms: float = Field(..., ge=0, description="Wall-clock time for encode + predict")