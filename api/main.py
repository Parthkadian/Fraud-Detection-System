"""
Fraud Detection API
===================
Production-grade FastAPI application for real-time and batch fraud scoring.

Features
--------
- Real-time single-transaction prediction   ``POST /predict``
- Batch CSV-style prediction                ``POST /predict_batch``
- SHAP explainability                       ``POST /explain``
- Evidently data-drift reporting            ``POST /drift_report``
- Celery-backed async prediction            ``POST /predict_async``
- Health check with model-load status       ``GET  /health``
- Prometheus metrics endpoint               ``GET  /metrics``
- Prediction audit trail                    ``GET  /audit/history``
- Business rules listing                    ``GET  /rules``
- Model card                                ``GET  /model_card``
- Request-ID tracking and timing middleware
- CORS support for cross-origin dashboard access
"""

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from api.schemas import (
    TransactionInput,
    PredictionResponse,
    HealthResponse,
    ExplanationResponse,
    AsyncTaskResponse,
    TaskStatusResponse,
)
from api.middleware import RequestIDMiddleware, RequestTimingMiddleware
from src.inference.predict import FraudPredictor
from src.inference.batch_predict import BatchPredictor
from src.explainability.shap_explainer import ShapExplainer

# ═══════════════════════════════════════════════════════════════════════════ #
#  Logging
# ═══════════════════════════════════════════════════════════════════════════ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud_detection_logger")

# ═══════════════════════════════════════════════════════════════════════════ #
#  Model state (module-level, loaded once)
# ═══════════════════════════════════════════════════════════════════════════ #
predictor: FraudPredictor | None = None
batch_predictor: BatchPredictor | None = None
shap_explainer: ShapExplainer | None = None
MODEL_LOADED: bool = False
MODEL_ERROR: str | None = None


# ═══════════════════════════════════════════════════════════════════════════ #
#  Lifespan (replaces deprecated @app.on_event)
# ═══════════════════════════════════════════════════════════════════════════ #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, clean up on shutdown."""
    global predictor, batch_predictor, shap_explainer, MODEL_LOADED, MODEL_ERROR

    try:
        predictor = FraudPredictor()
        batch_predictor = BatchPredictor()
        shap_explainer = ShapExplainer()
        MODEL_LOADED = True
        logger.info("✅ All models loaded successfully")
    except Exception as exc:
        MODEL_LOADED = False
        MODEL_ERROR = str(exc)
        logger.error(f"❌ Model loading failed: {exc}")

    logger.info("🚀 Fraud Detection API v2.0 started")
    yield
    logger.info("🛑 Fraud Detection API shutting down")


# ═══════════════════════════════════════════════════════════════════════════ #
#  App
# ═══════════════════════════════════════════════════════════════════════════ #
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Production-grade fraud risk scoring API with real-time prediction, "
        "batch analysis, SHAP explainability, data-drift monitoring, and "
        "async Celery-backed processing."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Helper
# ═══════════════════════════════════════════════════════════════════════════ #
def _require_model():
    """Raise 503 if the model is not loaded."""
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {MODEL_ERROR}",
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Routes
# ═══════════════════════════════════════════════════════════════════════════ #
@app.get("/", tags=["System"])
def home():
    """Root endpoint — confirms the API is running."""
    return {"message": "Fraud Detection API is running", "version": "2.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Returns model loading status and overall health."""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "unhealthy",
        model_loaded=MODEL_LOADED,
        error=MODEL_ERROR,
    )


# ── Prediction ───────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Score a single transaction",
)
def predict(transaction: TransactionInput):
    """
    Score a single credit-card transaction and return the fraud probability,
    binary prediction, and risk level.
    """
    _require_model()
    try:
        result = predictor.predict(transaction.model_dump())
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict_batch", tags=["Prediction"], summary="Score a batch of transactions")
def predict_batch(transactions: list[dict]):
    """Score multiple transactions at once.  Returns a list of scored records."""
    _require_model()
    try:
        df = pd.DataFrame(transactions)
        result_df = batch_predictor.predict_dataframe(df)
        return result_df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ── Explainability ───────────────────────────────────────────────────────
@app.post(
    "/explain",
    response_model=ExplanationResponse,
    tags=["Explainability"],
    summary="SHAP explanation for a transaction",
)
def explain(transaction: TransactionInput):
    """
    Generate a SHAP-based explanation showing the top contributing features
    for a single transaction's fraud score.
    """
    _require_model()
    try:
        explanation = shap_explainer.explain_single(transaction.model_dump(), top_n=10)
        return explanation
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ── Drift Monitoring ─────────────────────────────────────────────────────
@app.post("/drift_report", tags=["Monitoring"], summary="Generate data-drift report")
def generate_drift_report(transactions: list[dict]):
    """
    Compare recent production transactions against the training reference data
    and generate an Evidently data-drift HTML report.
    """
    from src.monitoring.drift_detection import DriftMonitor
    from fastapi.responses import FileResponse

    try:
        monitor = DriftMonitor()
        report_path = monitor.generate_drift_report(transactions)

        if "Error" in report_path:
            raise HTTPException(status_code=400, detail=report_path)

        if os.path.exists(report_path):
            return FileResponse(
                report_path, media_type="text/html", filename="data_drift.html"
            )
        raise HTTPException(status_code=500, detail="Report generation failed.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Async (Celery) ───────────────────────────────────────────────────────
@app.post(
    "/predict_async",
    response_model=AsyncTaskResponse,
    tags=["Async"],
    summary="Queue a transaction for async scoring",
)
def predict_async(transaction: TransactionInput):
    """Enqueue a transaction onto the Celery worker queue for async processing."""
    from api.celery_worker import predict_transaction

    _require_model()
    try:
        task = predict_transaction.delay(transaction.model_dump())
        return AsyncTaskResponse(task_id=task.id, status="Processing")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/task_status/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Async"],
    summary="Poll async task status",
)
def get_task_status(task_id: str):
    """Check the status and result of a previously queued async prediction."""
    from api.celery_worker import celery_app
    from celery.result import AsyncResult

    task = AsyncResult(task_id, app=celery_app)

    if task.state == "PENDING":
        return TaskStatusResponse(state=task.state, status="Pending...")
    elif task.state != "FAILURE":
        return TaskStatusResponse(state=task.state, result=task.result)
    else:
        return TaskStatusResponse(state=task.state, status=str(task.info))


# ── Prometheus Metrics ───────────────────────────────────────────────────
@app.get(
    "/metrics",
    tags=["Observability"],
    summary="Prometheus metrics endpoint",
    response_class=PlainTextResponse,
)
def prometheus_metrics():
    """
    Exposes Prometheus-format operational metrics:
    - ``fraud_predictions_total`` — labelled by risk_level
    - ``fraud_flagged_total`` — fraud-positive count
    - ``fraud_rule_triggered_total`` — business rule override count
    - ``fraud_prediction_latency_seconds`` — latency histogram
    - ``fraud_last_probability`` — latest fraud probability gauge
    - ``fraud_model_info`` — static model metadata
    """
    from src.monitoring.prometheus_metrics import get_metrics
    metrics = get_metrics()
    text, content_type = metrics.get_metrics_output()
    return PlainTextResponse(content=text, media_type=content_type)


# ── Audit Trail ──────────────────────────────────────────────────────────
@app.get(
    "/audit/history",
    tags=["Observability"],
    summary="Retrieve prediction audit trail",
)
def audit_history(
    limit: int = Query(default=100, ge=1, le=1000, description="Number of records to return"),
):
    """
    Returns the most recent predictions from the SQLite audit log,
    including input hash, probability, prediction, risk level,
    any triggered business rule, and inference latency.
    """
    from src.monitoring.audit_log import PredictionAuditLogger
    audit = PredictionAuditLogger()
    return {
        "records": audit.get_recent(limit=limit),
        "stats": audit.get_stats(),
        "limit": limit,
    }


# ── Business Rules ───────────────────────────────────────────────────────
@app.get(
    "/rules",
    tags=["Observability"],
    summary="List active business rules",
)
def list_rules():
    """
    Returns all active business rules loaded from
    ``configs/business_rules.yaml``.  Rules can be inspected without
    restarting the API.
    """
    from src.rules.rule_engine import BusinessRuleEngine
    engine = BusinessRuleEngine()
    return {
        "rules": engine.get_rules(),
        "total": len(engine.get_rules()),
    }


# ── Model Card ───────────────────────────────────────────────────────────
@app.get(
    "/model_card",
    tags=["Observability"],
    summary="Retrieve the model card (Google format)",
)
def model_card():
    """
    Returns the structured model card documenting:
    - Model details (version, algorithm, date trained)
    - Intended use and out-of-scope uses
    - Training data statistics
    - Performance metrics
    - Ethical considerations and limitations
    - Deployment configuration
    """
    card_path = Path("docs/model_card.json")
    if not card_path.exists():
        raise HTTPException(status_code=404, detail="Model card not found.")
    with open(card_path, "r", encoding="utf-8") as f:
        return json.load(f)