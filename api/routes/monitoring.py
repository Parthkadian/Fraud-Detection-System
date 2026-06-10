"""
api/routes/monitoring.py
========================
System observability and monitoring endpoints.

Routes:
  GET  /health          — liveness + model/database/uptime status
  GET  /metrics         — Prometheus-format metrics (plain text)
  GET  /audit/history   — recent prediction audit trail
  GET  /rules           — active business rules
  GET  /model_card      — structured model card from docs/model_card.json
  POST /drift_report    — Evidently data-drift HTML report

All existing logic from api/main.py is preserved verbatim; the routes are
simply reorganised here so api/main.py can remain the single source of truth
for app state while the router handles HTTP concerns.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

import api.database as db
from api.middleware import verify_api_key
from api.schemas import HealthResponse

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(tags=["Monitoring & Observability"])

# Module-load timestamp used to compute uptime_seconds.
# Close enough to server startup for observability purposes.
_MODULE_START: float = time.monotonic()

# Path constants
_MODEL_CARD_PATH = Path("docs/model_card.json")


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_main_state() -> dict:
    """
    Read all relevant state from api.main without crashing.
    Returns a dict with keys: model_loaded, model_error, predictor, model_version.
    """
    state = {
        "model_loaded":  False,
        "model_error":   None,
        "predictor":     None,
        "model_version": None,
    }
    try:
        from api.main import MODEL_ERROR, MODEL_LOADED, predictor  # type: ignore[attr-defined]
        state["model_loaded"] = bool(MODEL_LOADED)
        state["model_error"]  = MODEL_ERROR
        state["predictor"]    = predictor
        if MODEL_LOADED and predictor is not None:
            state["model_version"] = getattr(predictor, "version", None)
    except Exception as exc:
        logger.debug(f"Could not read api.main state: {exc}")
    return state


def _get_database_status() -> str:
    """
    Probe the SQLite platform DB with a lightweight query.
    Returns "ok" or "error".
    """
    try:
        db.get_dashboard_summary()
        return "ok"
    except Exception as exc:
        logger.warning(f"Database health probe failed: {exc}")
        return "error"


def _get_last_prediction_time() -> str | None:
    """
    Return the created_at timestamp of the most recent prediction, or None.
    """
    try:
        recent = db.get_recent_predictions(limit=1)
        if recent:
            return recent[0].get("created_at")
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /health
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description=(
        "Returns the overall health of the API including model load status, "
        "database connectivity, uptime, and environment. "
        "Always responds — never raises an error — so it is safe to use as a "
        "liveness probe in Docker / Kubernetes / Railway."
    ),
)
def health_check():
    """
    Liveness and readiness probe.

    Returns:
        status              – "healthy" | "unhealthy" | "degraded"
        model_loaded        – whether the ML model initialised successfully
        error               – model load error message, or null
        api_status          – always "ok" if this endpoint responds
        model_error         – alias of error field for clarity
        redis_status        – "unavailable" (Redis is optional in this deployment)
        database_status     – "ok" | "error" based on a lightweight DB probe
        uptime_seconds      – seconds since this module was first imported
        model_version       – version string of the loaded model, or null
        environment         – value of the ENVIRONMENT env var (default "development")
        last_prediction_time– ISO-8601 UTC timestamp of the most recent prediction
    """
    state           = _get_main_state()
    database_status = _get_database_status()
    uptime_seconds  = round(time.monotonic() - _MODULE_START, 1)

    model_loaded  = state["model_loaded"]
    model_error   = state["model_error"]
    model_version = state["model_version"]

    # Degraded = model fine but DB unhealthy; Unhealthy = model not loaded
    if not model_loaded:
        status = "unhealthy"
    elif database_status != "ok":
        status = "degraded"
    else:
        status = "healthy"

    last_prediction_time = _get_last_prediction_time()

    return HealthResponse(
        status               = status,
        model_loaded         = model_loaded,
        error                = model_error,
        api_status           = "ok",
        model_error          = model_error,
        redis_status         = "unavailable",
        database_status      = database_status,
        uptime_seconds       = uptime_seconds,
        model_version        = model_version,
        environment          = os.getenv("ENVIRONMENT", "development"),
        last_prediction_time = last_prediction_time,
    )


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /metrics
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    description=(
        "Exposes Prometheus-format operational metrics for scraping by a Prometheus "
        "server or Grafana agent. Returns plain text in the Prometheus exposition format."
    ),
    response_class=PlainTextResponse,
)
def prometheus_metrics():
    """
    Prometheus metrics output.

    Metrics exposed:
        fraud_predictions_total          — labelled by risk_level
        fraud_flagged_total              — fraud-positive prediction count
        fraud_rule_triggered_total       — business rule override count
        fraud_prediction_latency_seconds — latency histogram
        fraud_last_probability           — latest fraud probability (gauge)
        fraud_model_info                 — static model metadata (info metric)
    """
    try:
        from src.monitoring.prometheus_metrics import get_metrics
        metrics = get_metrics()
        text, content_type = metrics.get_metrics_output()
        return PlainTextResponse(content=text, media_type=content_type)
    except Exception as exc:
        logger.error(f"Prometheus metrics generation failed: {exc}")
        # Return an empty-but-valid Prometheus response rather than crashing
        return PlainTextResponse(
            content=f"# ERROR: metrics unavailable — {exc}\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /audit/history
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/audit/history",
    summary="Retrieve prediction audit trail",
    description=(
        "Returns the most recent prediction records from the audit log, including "
        "input hash, fraud probability, binary prediction, risk level, any triggered "
        "business rule, and inference latency."
    ),
)
def audit_history(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Number of records to return (1–1000).",
    ),
    _: None = Depends(verify_api_key),
):
    """
    Prediction audit trail from the SQLite audit log.

    Returns:
        records – list of audit log entries
        stats   – aggregate statistics (total, fraud rate, avg latency)
        limit   – the limit applied
    """
    try:
        from src.monitoring.audit_log import PredictionAuditLogger
        audit = PredictionAuditLogger()
        return {
            "records": audit.get_recent(limit=limit),
            "stats":   audit.get_stats(),
            "limit":   limit,
        }
    except Exception as exc:
        logger.error(f"Audit history fetch failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "Audit log unavailable",
                "details": str(exc),
            },
        )


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /rules
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/rules",
    summary="List active business rules",
    description=(
        "Returns all active business rules loaded from configs/business_rules.yaml. "
        "Rules can be inspected at runtime without restarting the API."
    ),
)
def list_rules(
    _: None = Depends(verify_api_key),
):
    """
    Active business rule definitions.

    Returns:
        rules – list of rule objects (name, condition, action, threshold, etc.)
        total – count of active rules
    """
    try:
        from src.rules.rule_engine import BusinessRuleEngine
        engine = BusinessRuleEngine()
        rules  = engine.get_rules()
        return {
            "rules": rules,
            "total": len(rules),
        }
    except Exception as exc:
        logger.error(f"Business rules fetch failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "Rules unavailable",
                "details": str(exc),
            },
        )


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /model_card
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/model_card",
    summary="Retrieve the model card (Google format)",
    description=(
        "Returns the structured model card documenting model details, intended use, "
        "training data statistics, performance metrics, ethical considerations, and "
        "deployment configuration. Reads from docs/model_card.json."
    ),
)
def get_model_card():
    """
    Structured model card.

    Returns the full contents of docs/model_card.json.
    Raises HTTP 404 if the model card file does not exist.
    """
    if not _MODEL_CARD_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error":   "Model card not found",
                "details": f"Expected at '{_MODEL_CARD_PATH}'. "
                           "Ensure the file exists in your deployment.",
            },
        )
    try:
        with open(_MODEL_CARD_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error(f"Model card read failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "Model card unreadable",
                "details": str(exc),
            },
        )


# ══════════════════════════════════════════════════════════════════════════ #
#  POST /drift_report
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "/drift_report",
    summary="Generate Evidently data-drift report",
    description=(
        "Compares recent production transaction data against the training reference "
        "dataset and generates an Evidently HTML data-drift report. "
        "Returns the report as a downloadable HTML file."
    ),
)
def generate_drift_report(
    transactions: list[dict],
    _: None = Depends(verify_api_key),
):
    """
    Evidently data-drift report generation.

    Body:
        transactions – list of recent transaction dicts (same schema as /predict input)

    Returns:
        HTML file download (Content-Type: text/html)

    Raises:
        HTTP 400 — if the drift monitor returns an error string
        HTTP 500 — on unexpected failure
    """
    try:
        from src.monitoring.drift_detection import DriftMonitor
        monitor     = DriftMonitor()
        report_path = monitor.generate_drift_report(transactions)

        if "Error" in str(report_path):
            raise HTTPException(
                status_code=400,
                detail={
                    "error":   "Drift report generation failed",
                    "details": report_path,
                },
            )

        if os.path.exists(report_path):
            return FileResponse(
                report_path,
                media_type="text/html",
                filename="data_drift.html",
            )

        raise HTTPException(
            status_code=500,
            detail={
                "error":   "Report file not found after generation",
                "details": f"Expected output at: {report_path}",
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Drift report failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "Drift report error",
                "details": str(exc),
            },
        )
