"""
api/routes/alerts.py
====================
Alert management endpoints.

Routes:
  GET   /alerts/recent                   — fetch recent alerts, unacknowledged first
  PATCH /alerts/{alert_id}/acknowledge   — mark a single alert as acknowledged
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

import api.database as db
from api.middleware import verify_api_key

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(prefix="/alerts", tags=["Alerts"])


# ── GET /alerts/recent ─────────────────────────────────────────────────────

@router.get(
    "/recent",
    summary="Get recent alerts",
    description=(
        "Returns the most recent alerts, with unacknowledged alerts surfaced first. "
        "Use the `limit` query parameter to control how many records are returned "
        "(default: 100, max: 500)."
    ),
)
def get_recent_alerts(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of alerts to return (1–500).",
    ),
    _: None = Depends(verify_api_key),
):
    """
    Retrieve recent platform alerts.

    Alerts are ordered: unacknowledged first, then newest first within each group.

    Returns:
        alerts               – list of alert records
        total                – count of records returned
        unacknowledged_count – count of alerts not yet acknowledged
    """
    alerts = db.get_recent_alerts(limit=limit)

    unacknowledged_count = sum(
        1 for a in alerts if not a.get("acknowledged")
    )

    logger.info(
        f"Recent alerts fetched: {len(alerts)} records "
        f"({unacknowledged_count} unacknowledged)."
    )

    return {
        "alerts":               alerts,
        "total":                len(alerts),
        "unacknowledged_count": unacknowledged_count,
    }


# ── PATCH /alerts/{alert_id}/acknowledge ──────────────────────────────────

@router.patch(
    "/{alert_id}/acknowledge",
    summary="Acknowledge an alert",
    description=(
        "Mark a specific alert as acknowledged. "
        "Returns HTTP 404 if no alert with the given ID exists."
    ),
)
def acknowledge_alert(
    alert_id: str,
    _: None = Depends(verify_api_key),
):
    """
    Acknowledge a platform alert by ID.

    Once acknowledged, the alert will no longer appear at the top of
    GET /alerts/recent (acknowledged alerts are sorted to the bottom).

    Raises:
        HTTP 404 — if the alert_id does not exist
    """
    updated = db.acknowledge_alert(alert_id)

    if not updated:
        raise HTTPException(
            status_code=404,
            detail={
                "error":   "Alert not found",
                "details": f"No alert with id '{alert_id}' exists.",
            },
        )

    logger.info(f"Alert '{alert_id}' acknowledged.")

    return {
        "alert_id":     alert_id,
        "acknowledged": True,
        "message":      f"Alert '{alert_id}' has been acknowledged.",
    }
