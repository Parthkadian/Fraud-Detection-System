"""
api/routes/business.py
======================
Dashboard, business impact, and attack detection endpoints.

Routes:
  GET /dashboard/summary   — today's prediction counts, risk breakdown, queue size,
                             average latency, model/api status
  GET /business/impact     — financial exposure: potential loss, blocked loss,
                             false positive cost, confirmed fraud loss
  GET /attack/detection    — card testing, fraud spike, IP abuse, device abuse signals
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from api.middleware import verify_api_key
import api.database as db

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(tags=["Dashboard & Business Intelligence"])


# ── Helper: safely read model state from main app ─────────────────────────

def _get_model_state() -> tuple[bool, str | None]:
    """
    Read MODEL_LOADED and model version from api.main without crashing.
    Returns (model_loaded: bool, model_version: str | None).
    Falls back gracefully if import or attribute access fails.
    """
    try:
        from api.main import MODEL_LOADED, predictor  # type: ignore[attr-defined]
        version = getattr(predictor, "version", None) if MODEL_LOADED else None
        return bool(MODEL_LOADED), version
    except Exception:
        return False, None


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /dashboard/summary
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/dashboard/summary",
    summary="Get dashboard summary",
    description=(
        "Returns today's prediction counts broken down by risk level, the current "
        "human review queue size, average inference latency, and model/API status. "
        "All numeric fields fall back to safe zero values if no data exists yet."
    ),
)
def get_dashboard_summary(
    _: None = Depends(verify_api_key),
):
    """
    Aggregate KPIs for the analyst dashboard.

    Returns:
        total_predictions_today – scored transactions since UTC midnight
        high_risk_today         – HIGH-risk transaction count today
        medium_risk_today       – MEDIUM-risk transaction count today
        low_risk_today          – LOW-risk transaction count today
        review_queue_size       – open OPEN + UNDER_REVIEW cases awaiting analyst
        confirmed_fraud_count   – all-time cases confirmed as fraud
        false_positive_count    – all-time cases resolved as false positive
        average_latency_ms      – mean model inference latency today (ms)
        model_loaded            – whether the ML model is currently loaded
        model_version           – version string of the loaded model, or null
        drift_status            – placeholder: "unknown" (extend with drift monitor)
        api_status              – always "ok" if this endpoint responds
    """
    try:
        summary = db.get_dashboard_summary()
    except Exception as exc:
        logger.error(f"Dashboard summary DB error: {exc}")
        summary = {}

    model_loaded, model_version = _get_model_state()

    return {
        "total_predictions_today": summary.get("total_predictions_today", 0) or 0,
        "high_risk_today":         summary.get("high_risk_today", 0) or 0,
        "medium_risk_today":       summary.get("medium_risk_today", 0) or 0,
        "low_risk_today":          summary.get("low_risk_today", 0) or 0,
        "review_queue_size":       summary.get("review_queue_size", 0) or 0,
        "confirmed_fraud_count":   summary.get("confirmed_fraud_count", 0) or 0,
        "false_positive_count":    summary.get("false_positive_count", 0) or 0,
        "average_latency_ms":      summary.get("average_latency_ms", 0.0) or 0.0,
        "model_loaded":            model_loaded,
        "model_version":           model_version,
        "drift_status":            "unknown",
        "api_status":              "ok",
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /business/impact
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/business/impact",
    summary="Get business impact / financial exposure",
    description=(
        "Returns today's financial exposure summary: total potential loss from "
        "all scored transactions, amount blocked by BLOCK decisions, estimated "
        "false-positive operational cost, and all-time confirmed fraud loss. "
        "All values are in the platform's base currency (GBP)."
    ),
)
def get_business_impact(
    _: None = Depends(verify_api_key),
):
    """
    Financial impact and fraud cost breakdown.

    Returns:
        total_potential_loss           – sum of potential_loss for all today's predictions
        blocked_loss_today             – sum of amount blocked (decision=BLOCK) today
        estimated_false_positive_cost  – estimated operational cost from flagged-but-legit txns today
        confirmed_fraud_loss           – sum of transaction amounts for CONFIRMED_FRAUD cases (all time)
        currency                       – base currency (GBP)
    """
    try:
        impact = db.get_business_impact()
    except Exception as exc:
        logger.error(f"Business impact DB error: {exc}")
        impact = {}

    return {
        "total_potential_loss":          impact.get("total_potential_loss", 0.0) or 0.0,
        "blocked_loss_today":            impact.get("blocked_loss_today", 0.0) or 0.0,
        "estimated_false_positive_cost": impact.get("estimated_false_positive_cost", 0.0) or 0.0,
        "confirmed_fraud_loss":          impact.get("confirmed_fraud_loss", 0.0) or 0.0,
        "currency":                      "GBP",
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /attack/detection
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/attack/detection",
    summary="Get coordinated attack / fraud pattern signals",
    description=(
        "Detects coordinated fraud patterns in the last 24 hours. "
        "Checks for: FRAUD_SPIKE (sudden high-risk surge), CARD_TESTING "
        "(many micro-transactions from same device), IP_ABUSE (single IP "
        "linked to many customers), and DEVICE_ABUSE (shared device across "
        "multiple customers). Returns NONE severity when no patterns are detected."
    ),
)
def get_attack_detection(
    _: None = Depends(verify_api_key),
):
    """
    Coordinated fraud attack detection signals.

    Returns:
        attack_detected  – true if any pattern was triggered
        overall_severity – NONE | HIGH | CRITICAL
        attacks          – list of attack detail objects, each with:
                             attack_type, severity, affected_entities,
                             detail, recommendation
        checked_at       – UTC ISO-8601 timestamp of when the check ran
    """
    try:
        signals = db.get_attack_signals()
    except Exception as exc:
        logger.error(f"Attack detection DB error: {exc}")
        # Return a safe fallback — never let analytics crash the API
        signals = {
            "attack_detected":  False,
            "overall_severity": "NONE",
            "attacks":          [],
            "checked_at":       None,
        }

    if signals.get("attack_detected"):
        logger.warning(
            f"Attack detection: {signals.get('overall_severity')} severity — "
            f"{len(signals.get('attacks', []))} pattern(s) detected."
        )

    return {
        "attack_detected":  signals.get("attack_detected", False),
        "overall_severity": signals.get("overall_severity", "NONE"),
        "attacks":          signals.get("attacks", []),
        "checked_at":       signals.get("checked_at"),
    }
