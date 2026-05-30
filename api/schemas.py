"""
api/schemas.py
==============
Pydantic request/response models for the Fraud Detection Platform.

Backward compatible: all original schemas (TransactionInput, PredictionResponse,
HealthResponse, ExplanationResponse, AsyncTaskResponse, TaskStatusResponse,
ErrorResponse) are preserved with their original fields. New fields are added
as Optional so existing callers are not broken.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════ #
#  REQUEST — Transaction Input
# ══════════════════════════════════════════════════════════════════════════ #

class TransactionInput(BaseModel):
    """
    Single credit-card transaction for fraud scoring.

    Core PCA fields (V1-V28, Time, Amount) are required for the ML model.
    All metadata fields are optional — they enrich the response, enable
    blacklist/whitelist checks, velocity tracking, and case creation.
    """

    # ── Core ML fields (unchanged) ─────────────────────────────────────
    Time: float = Field(..., description="Seconds elapsed from first transaction in dataset")
    V1:  float = Field(..., description="PCA component 1")
    V2:  float = Field(..., description="PCA component 2")
    V3:  float = Field(..., description="PCA component 3")
    V4:  float = Field(..., description="PCA component 4")
    V5:  float = Field(..., description="PCA component 5")
    V6:  float = Field(..., description="PCA component 6")
    V7:  float = Field(..., description="PCA component 7")
    V8:  float = Field(..., description="PCA component 8")
    V9:  float = Field(..., description="PCA component 9")
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

    # ── Optional real-world metadata (new) ────────────────────────────
    transaction_id:   Optional[str] = Field(default=None, description="Unique transaction reference")
    customer_id:      Optional[str] = Field(default=None, description="Customer identifier")
    merchant_id:      Optional[str] = Field(default=None, description="Merchant identifier")
    device_id:        Optional[str] = Field(default=None, description="Device fingerprint or ID")
    ip_address:       Optional[str] = Field(default=None, description="Client IP address")
    country:          Optional[str] = Field(default=None, description="ISO 3166-1 alpha-2 country code")
    channel:          Optional[str] = Field(default=None, description="Transaction channel: ONLINE / POS / ATM / MOBILE")
    transaction_type: Optional[str] = Field(default=None, description="Type: PURCHASE / WITHDRAWAL / TRANSFER / REFUND")
    currency:         Optional[str] = Field(default="GBP", description="ISO 4217 currency code")
    timestamp:        Optional[datetime] = Field(default=None, description="Transaction timestamp (UTC)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Time": 10000.0,
                    "V1": -1.2, "V2": 0.3,  "V3": 1.1,  "V4": 0.5,  "V5": -0.2,
                    "V6":  0.1, "V7": 0.2,  "V8": -0.1, "V9": 0.4,  "V10": -0.3,
                    "V11": 0.2, "V12": -0.5,"V13": 0.1, "V14": -0.2,"V15": 0.3,
                    "V16":-0.1, "V17": 0.2, "V18": 0.1, "V19": -0.3,"V20": 0.05,
                    "V21":-0.02,"V22": 0.1, "V23":-0.03,"V24": 0.2, "V25": -0.1,
                    "V26": 0.05,"V27": 0.02,"V28":-0.01,
                    "Amount": 150.50,
                    "transaction_memo": "Online electronics purchase",
                    "transaction_id": "TXN-20240501-001",
                    "customer_id":    "CUST-4821",
                    "merchant_id":    "MERCH-992",
                    "device_id":      "DEV-AB12CD",
                    "ip_address":     "192.168.1.10",
                    "country":        "GB",
                    "channel":        "ONLINE",
                    "transaction_type": "PURCHASE",
                    "currency":       "GBP",
                }
            ]
        }
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  RESPONSE — Prediction
# ══════════════════════════════════════════════════════════════════════════ #

class PredictionResponse(BaseModel):
    """
    Response from /predict.

    Original fields (fraud_probability, prediction, risk_level,
    rule_triggered, latency_ms) are preserved unchanged.
    New fields are Optional with safe defaults so old clients still work.
    """

    # ── Original fields (unchanged) ───────────────────────────────────
    fraud_probability: float = Field(..., description="Probability of fraud (0–1)")
    prediction:        int   = Field(..., description="Binary prediction: 0=legit, 1=fraud")
    risk_level:        str   = Field(..., description="LOW | MEDIUM | HIGH")
    rule_triggered:    Optional[str]   = Field(default=None, description="Business rule that overrode ML, or null")
    latency_ms:        Optional[float] = Field(default=None, description="End-to-end inference latency (ms)")

    # ── New enriched fields ───────────────────────────────────────────
    transaction_id:      Optional[str]       = Field(default=None, description="Transaction reference passed in request")
    customer_id:         Optional[str]       = Field(default=None)
    merchant_id:         Optional[str]       = Field(default=None)
    decision:            Optional[str]       = Field(default=None, description="APPROVE | REVIEW | BLOCK")
    reason_codes:        Optional[List[str]] = Field(default=None, description="List of reason codes explaining the decision")
    model_version:       Optional[str]       = Field(default=None, description="Version of the model that scored this transaction")
    case_id:             Optional[str]       = Field(default=None, description="Created case ID for REVIEW/BLOCK decisions")
    potential_loss:      Optional[float]     = Field(default=None, description="Estimated financial exposure (same as Amount for fraud)")
    blocked_loss:        Optional[float]     = Field(default=None, description="Amount blocked if decision=BLOCK")
    false_positive_cost: Optional[float]     = Field(default=None, description="Estimated operational cost if this is a false positive")


# ══════════════════════════════════════════════════════════════════════════ #
#  RESPONSE — Health (original, unchanged)
# ══════════════════════════════════════════════════════════════════════════ #

class HealthResponse(BaseModel):
    """Response from /health — enhanced but backward compatible."""

    status:               str            = Field(..., description="healthy | unhealthy | degraded")
    model_loaded:         bool
    error:                Optional[str]  = None

    # New fields — optional so old clients ignore them
    api_status:           Optional[str]  = Field(default="ok")
    model_error:          Optional[str]  = None
    redis_status:         Optional[str]  = Field(default="unavailable")
    database_status:      Optional[str]  = Field(default="ok")
    uptime_seconds:       Optional[float]= None
    model_version:        Optional[str]  = None
    environment:          Optional[str]  = Field(default="development")
    last_prediction_time: Optional[str]  = None


# ══════════════════════════════════════════════════════════════════════════ #
#  RESPONSE — Explainability (original, unchanged)
# ══════════════════════════════════════════════════════════════════════════ #

class FeatureContribution(BaseModel):
    """Single feature SHAP contribution."""
    feature:    str
    shap_value: float


class ExplanationResponse(BaseModel):
    """Response from /explain."""
    top_features: List[FeatureContribution]


# ══════════════════════════════════════════════════════════════════════════ #
#  RESPONSE — Async / Task (original, unchanged)
# ══════════════════════════════════════════════════════════════════════════ #

class AsyncTaskResponse(BaseModel):
    """Response from /predict_async."""
    task_id: str
    status:  str = "Processing"


class TaskStatusResponse(BaseModel):
    """Response from /task_status/{task_id}."""
    state:  str
    status: Optional[str]  = None
    result: Optional[dict] = None


# ══════════════════════════════════════════════════════════════════════════ #
#  RESPONSE — Standard Error (original + request_id)
# ══════════════════════════════════════════════════════════════════════════ #

class ErrorResponse(BaseModel):
    """Standard error response for all new endpoints."""
    error:      str
    details:    Optional[str] = None
    request_id: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════ #
#  CASES
# ══════════════════════════════════════════════════════════════════════════ #

class CaseCreateRequest(BaseModel):
    """Body for POST /cases (manual case creation)."""
    transaction_id:    Optional[str]   = None
    customer_id:       Optional[str]   = None
    merchant_id:       Optional[str]   = None
    fraud_probability: Optional[float] = None
    risk_level:        Optional[str]   = Field(default="MEDIUM", description="LOW | MEDIUM | HIGH")
    decision:          Optional[str]   = Field(default="REVIEW",  description="APPROVE | REVIEW | BLOCK")
    assigned_to:       Optional[str]   = None


class CaseResponse(BaseModel):
    """Full case record returned from case endpoints."""
    id:                str
    transaction_id:    Optional[str]   = None
    customer_id:       Optional[str]   = None
    merchant_id:       Optional[str]   = None
    fraud_probability: Optional[float] = None
    risk_level:        Optional[str]   = None
    decision:          Optional[str]   = None
    status:            str
    assigned_to:       Optional[str]   = None
    created_at:        str
    updated_at:        str
    notes:             Optional[List[dict]] = None   # included when fetching single case


class CaseStatusUpdate(BaseModel):
    """Body for PATCH /cases/{case_id}/status."""
    status: str = Field(
        ...,
        description="OPEN | UNDER_REVIEW | CONFIRMED_FRAUD | FALSE_POSITIVE | CLOSED",
    )


class CaseAssignRequest(BaseModel):
    """Body for PATCH /cases/{case_id}/assign."""
    assigned_to: str = Field(..., description="Analyst username or ID to assign the case to")


class CaseNoteRequest(BaseModel):
    """Body for POST /cases/{case_id}/notes."""
    note:       str            = Field(..., description="Note text")
    analyst_id: Optional[str] = Field(default=None, description="Analyst submitting the note")


# ══════════════════════════════════════════════════════════════════════════ #
#  REVIEW QUEUE
# ══════════════════════════════════════════════════════════════════════════ #

class ReviewDecisionRequest(BaseModel):
    """Body for PATCH /review/{case_id}/decision."""
    decision:   str            = Field(..., description="CONFIRMED_FRAUD | FALSE_POSITIVE | CLOSED")
    analyst_id: Optional[str] = None
    notes:      Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════ #
#  FEEDBACK
# ══════════════════════════════════════════════════════════════════════════ #

class FeedbackRequest(BaseModel):
    """Body for POST /feedback."""
    transaction_id: Optional[str] = None
    case_id:        Optional[str] = None
    actual_label:   str           = Field(..., description="FRAUD | LEGIT")
    analyst_id:     Optional[str] = None
    notes:          Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "TXN-20240501-001",
                    "case_id":        "CASE-AB12CD34EF56",
                    "actual_label":   "FRAUD",
                    "analyst_id":     "analyst_001",
                    "notes":          "Confirmed by cardholder — account takeover.",
                }
            ]
        }
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  ALERTS
# ══════════════════════════════════════════════════════════════════════════ #

class AlertResponse(BaseModel):
    """Alert record returned from /alerts/recent."""
    id:             str
    transaction_id: Optional[str]  = None
    case_id:        Optional[str]  = None
    severity:       str
    message:        str
    acknowledged:   bool
    created_at:     str


class AlertAcknowledgeResponse(BaseModel):
    """Response from PATCH /alerts/{alert_id}/acknowledge."""
    alert_id:     str
    acknowledged: bool
    message:      str


# ══════════════════════════════════════════════════════════════════════════ #
#  BLACKLIST / WHITELIST
# ══════════════════════════════════════════════════════════════════════════ #

class ListEntityRequest(BaseModel):
    """
    Body for POST /blacklist and POST /whitelist.
    entity_type must be one of: customer, merchant, device, ip, country.
    """
    entity_type:  str            = Field(..., description="customer | merchant | device | ip | country")
    entity_value: str            = Field(..., description="The value to block/trust")
    reason:       Optional[str] = None
    added_by:     Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "entity_type":  "ip",
                    "entity_value": "185.220.101.5",
                    "reason":       "Known Tor exit node used in multiple fraud attempts",
                    "added_by":     "fraud_analyst_01",
                }
            ]
        }
    }


class ListEntityResponse(BaseModel):
    """Single blacklist/whitelist record."""
    id:           str
    entity_type:  str
    entity_value: str
    reason:       Optional[str] = None
    added_by:     Optional[str] = None
    created_at:   str


# ══════════════════════════════════════════════════════════════════════════ #
#  DASHBOARD SUMMARY
# ══════════════════════════════════════════════════════════════════════════ #

class DashboardSummaryResponse(BaseModel):
    """Response from GET /dashboard/summary."""
    total_predictions_today: int
    high_risk_today:         int
    medium_risk_today:       int
    low_risk_today:          int
    review_queue_size:       int
    confirmed_fraud_count:   int
    false_positive_count:    int
    average_latency_ms:      float
    model_loaded:            bool
    model_version:           Optional[str] = None
    drift_status:            Optional[str] = Field(default="unknown", description="ok | drifted | unknown")
    api_status:              str           = "ok"


# ══════════════════════════════════════════════════════════════════════════ #
#  BUSINESS IMPACT
# ══════════════════════════════════════════════════════════════════════════ #

class BusinessImpactResponse(BaseModel):
    """Response from GET /business/impact."""
    total_potential_loss:          float
    blocked_loss_today:            float
    estimated_false_positive_cost: float
    confirmed_fraud_loss:          float


# ══════════════════════════════════════════════════════════════════════════ #
#  CUSTOMER / MERCHANT RISK PROFILES
# ══════════════════════════════════════════════════════════════════════════ #

class CustomerRiskProfileResponse(BaseModel):
    """Response from GET /customers/{customer_id}/risk-profile."""
    customer_id:           str
    total_transactions:    int
    high_risk_count:       int
    confirmed_fraud_count: int
    average_amount:        float
    last_transaction_time: Optional[str] = None
    risk_tier:             str           = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    linked_devices:        int
    linked_merchants:      int


class MerchantRiskProfileResponse(BaseModel):
    """Response from GET /merchants/{merchant_id}/risk-profile."""
    merchant_id:        str
    total_transactions: int
    high_risk_count:    int
    fraud_rate:         float  = Field(..., description="Fraud percentage (0–100)")
    average_amount:     float
    risk_tier:          str    = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    linked_customers:   int


# ══════════════════════════════════════════════════════════════════════════ #
#  VELOCITY
# ══════════════════════════════════════════════════════════════════════════ #

class VelocityResponse(BaseModel):
    """Response from GET /velocity/{entity_type}/{entity_value}."""
    entity_type:        str
    entity_value:       str
    count_last_1h:      int
    count_last_24h:     int
    max_amount_24h:     float
    avg_fraud_prob_24h: float
    unique_customers:   int
    unique_devices:     int
    unique_ips:         int
    velocity_risk:      str           = Field(..., description="LOW | MEDIUM | HIGH")
    signals:            List[str]     = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════ #
#  ATTACK DETECTION
# ══════════════════════════════════════════════════════════════════════════ #

class AttackDetail(BaseModel):
    """A single detected attack pattern."""
    attack_type:       str
    severity:          str
    affected_entities: List[Any]  = Field(default_factory=list)
    detail:            str
    recommendation:    str


class AttackDetectionResponse(BaseModel):
    """Response from GET /attack/detection."""
    attack_detected:  bool
    overall_severity: str           = Field(..., description="NONE | HIGH | CRITICAL")
    attacks:          List[AttackDetail] = Field(default_factory=list)
    checked_at:       str


# ══════════════════════════════════════════════════════════════════════════ #
#  MODEL OPS
# ══════════════════════════════════════════════════════════════════════════ #

class ModelVersionResponse(BaseModel):
    """Response from GET /model/version."""
    model_version: str
    algorithm:     str
    trained_on:    Optional[str] = None
    threshold:     Optional[float] = None
    feature_count: Optional[int]  = None


class ModelPerformanceResponse(BaseModel):
    """Response from GET /model/performance."""
    model_version: str
    algorithm:     str
    trained_on:    Optional[str]   = None
    threshold:     Optional[float] = None
    auc:           Optional[float] = None
    precision:     Optional[float] = None
    recall:        Optional[float] = None
    f1_score:      Optional[float] = None
    feature_count: Optional[int]   = None


# ══════════════════════════════════════════════════════════════════════════ #
#  CHAMPION-CHALLENGER
# ══════════════════════════════════════════════════════════════════════════ #

class PredictCompareResponse(BaseModel):
    """Response from POST /predict_compare."""
    champion_result:   dict
    challenger_result: dict
    disagreement:      bool
    recommendation:    str
    challenger_mode:   str = Field(..., description="real | simulated")


# ══════════════════════════════════════════════════════════════════════════ #
#  GLOBAL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════ #

class GlobalFeatureImportance(BaseModel):
    """Single feature with global importance score."""
    feature:    str
    importance: float
    rank:       int


class GlobalExplanationResponse(BaseModel):
    """Response from GET /explain/global."""
    top_global_features: List[GlobalFeatureImportance]
    explanation_method:  str = Field(..., description="shap | feature_importance | unavailable")
    model_version:       Optional[str] = None
    data_source:         str           = Field(..., description="real | fallback | unavailable")