"""
API Request Schemas & Response Models
=====================================
Pydantic models for request validation and response serialisation.
Provides auto-generated OpenAPI docs with examples.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ───────────────────────────── Request Models ──────────────────────────── #


class TransactionInput(BaseModel):
    """Single credit-card transaction for fraud scoring."""

    Time: float = Field(..., description="Seconds elapsed from first transaction in dataset")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., ge=0, description="Transaction amount in currency units")
    transaction_memo: Optional[str] = Field(
        default="Standard purchase",
        description="Optional text memo associated with the transaction",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Time": 10000.0,
                    "V1": -1.2, "V2": 0.3, "V3": 1.1, "V4": 0.5, "V5": -0.2,
                    "V6": 0.1, "V7": 0.2, "V8": -0.1, "V9": 0.4, "V10": -0.3,
                    "V11": 0.2, "V12": -0.5, "V13": 0.1, "V14": -0.2, "V15": 0.3,
                    "V16": -0.1, "V17": 0.2, "V18": 0.1, "V19": -0.3, "V20": 0.05,
                    "V21": -0.02, "V22": 0.1, "V23": -0.03, "V24": 0.2, "V25": -0.1,
                    "V26": 0.05, "V27": 0.02, "V28": -0.01,
                    "Amount": 150.50,
                    "transaction_memo": "Online electronics purchase",
                }
            ]
        }
    }


# ───────────────────────────── Response Models ─────────────────────────── #


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    fraud_probability: float = Field(..., description="Probability of fraud (0–1)")
    prediction: int = Field(..., description="Binary prediction (0 = legit, 1 = fraud)")
    risk_level: str = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    rule_triggered: Optional[str] = Field(
        default=None,
        description="Name of the business rule that overrode the ML prediction, or null",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="End-to-end inference latency in milliseconds",
    )



class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(..., description="healthy | unhealthy")
    model_loaded: bool
    error: Optional[str] = None


class FeatureContribution(BaseModel):
    """Single feature SHAP contribution."""

    feature: str
    shap_value: float


class ExplanationResponse(BaseModel):
    """Response from the /explain endpoint."""

    top_features: list[FeatureContribution]


class AsyncTaskResponse(BaseModel):
    """Response from the /predict_async endpoint."""

    task_id: str
    status: str = "Processing"


class TaskStatusResponse(BaseModel):
    """Response from the /task_status endpoint."""

    state: str
    status: Optional[str] = None
    result: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    details: Optional[str] = None