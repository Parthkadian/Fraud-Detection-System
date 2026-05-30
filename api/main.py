"""
Fraud Detection API
===================
Production-grade FastAPI application for real-time and batch fraud scoring.

Features
--------
- Real-time single-transaction prediction   ``POST /predict``
- Batch CSV-style prediction                ``POST /predict_batch``
- Champion-challenger comparison            ``POST /predict_compare``
- SHAP explainability                       ``POST /explain``
- Global feature importance                 ``GET  /explain/global``
- Evidently data-drift reporting            ``POST /drift_report``
- Celery-backed async prediction            ``POST /predict_async``
- Health check with model-load status       ``GET  /health``
- Prometheus metrics endpoint               ``GET  /metrics``
- Prediction audit trail                    ``GET  /audit/history``
- Business rules listing                    ``GET  /rules``
- Model card                                ``GET  /model_card``
- Human review queue                        ``GET  /review/queue``
- Analyst decision submission               ``PATCH /review/{case_id}/decision``
- Blacklist / whitelist management          ``POST|GET|DELETE /blacklist`` etc.
- Customer / merchant risk profiles         ``GET  /customers/{id}/risk-profile`` etc.
- Velocity signals                          ``GET  /velocity/{entity_type}/{value}``
- Alert management                          ``GET  /alerts/recent`` etc.
- Dashboard & business intelligence         ``GET  /dashboard/summary`` etc.
- Model version & performance               ``GET  /model/version`` etc.
- Case management                           ``POST|GET|PATCH /cases`` etc.
- Request-ID tracking and timing middleware
- CORS support for cross-origin dashboard access
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import (
    RequestIDMiddleware,
    RequestTimingMiddleware,
    RateLimitMiddleware,
)
from api.schemas import HealthResponse
from src.inference.predict import FraudPredictor
from src.inference.batch_predict import BatchPredictor
from src.explainability.shap_explainer import ShapExplainer
import api.database as db

# ── Routers ───────────────────────────────────────────────────────────────
from api.routes import prediction   as prediction_router
from api.routes import cases        as cases_router
from api.routes import review       as review_router
from api.routes import lists        as lists_router
from api.routes import profiles     as profiles_router
from api.routes import alerts       as alerts_router
from api.routes import business     as business_router
from api.routes import model_ops    as model_ops_router
from api.routes import explainability as explainability_router
from api.routes import monitoring   as monitoring_router

# ═══════════════════════════════════════════════════════════════════════════ #
#  Logging
# ═══════════════════════════════════════════════════════════════════════════ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud_detection_logger")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Model state  (module-level, loaded once at startup)
# ═══════════════════════════════════════════════════════════════════════════ #
predictor:      FraudPredictor | None = None
batch_predictor: BatchPredictor | None = None
shap_explainer:  ShapExplainer  | None = None
MODEL_LOADED: bool      = False
MODEL_ERROR:  str | None = None


# ═══════════════════════════════════════════════════════════════════════════ #
#  Lifespan  (startup / shutdown)
# ═══════════════════════════════════════════════════════════════════════════ #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB tables and load ML models on startup; clean up on shutdown."""
    global predictor, batch_predictor, shap_explainer, MODEL_LOADED, MODEL_ERROR

    # ── 1. Database ───────────────────────────────────────────────────────
    try:
        db.init_db()
    except Exception as exc:
        logger.error(f"❌ Database initialisation failed: {exc}")
        # Non-fatal: API can still serve predictions without platform DB

    # ── 2. ML models ──────────────────────────────────────────────────────
    try:
        predictor       = FraudPredictor()
        batch_predictor = BatchPredictor()
        shap_explainer  = ShapExplainer()
        MODEL_LOADED    = True
        logger.info("✅ All models loaded successfully")
    except Exception as exc:
        MODEL_LOADED = False
        MODEL_ERROR  = str(exc)
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
        "batch analysis, SHAP explainability, data-drift monitoring, async "
        "Celery-backed processing, human review queue, blacklist/whitelist "
        "management, customer/merchant risk profiles, and business intelligence."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Middleware  (order matters: outermost added last) ─────────────────────
app.add_middleware(RateLimitMiddleware)
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
#  Routers
# ═══════════════════════════════════════════════════════════════════════════ #

# Prediction  — /predict, /predict_batch, /predict_compare
app.include_router(prediction_router.router)

# Case management  — /cases/*
app.include_router(cases_router.router)

# Human review queue  — /review/*
app.include_router(review_router.router)

# Blacklist / whitelist  — /blacklist, /whitelist
app.include_router(lists_router.router)

# Risk profiles & velocity  — /customers/*, /merchants/*, /velocity/*
app.include_router(profiles_router.router)

# Alerts  — /alerts/*
app.include_router(alerts_router.router)

# Dashboard & business intelligence  — /dashboard/*, /business/*, /attack/*
app.include_router(business_router.router)

# Model operations  — /model/*
app.include_router(model_ops_router.router)

# Explainability  — /explain, /explain/global
app.include_router(explainability_router.router)

# Monitoring & observability  — /health, /metrics, /audit/history,
#                               /rules, /model_card, /drift_report
app.include_router(monitoring_router.router)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Root endpoint
# ═══════════════════════════════════════════════════════════════════════════ #
@app.get("/", tags=["System"], summary="API root")
def home():
    """Root endpoint — confirms the API is running and returns version info."""
    return {
        "message": "Fraud Detection API is running",
        "version": "2.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }