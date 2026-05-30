"""
api/routes/profiles.py
======================
Risk profile and velocity analytics endpoints.

Routes:
  GET /customers/{customer_id}/risk-profile      — customer risk profile
  GET /merchants/{merchant_id}/risk-profile      — merchant risk profile
  GET /velocity/{entity_type}/{entity_value}     — real-time velocity signals

Valid velocity entity_type values: customer, merchant, device, ip
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.middleware import verify_api_key
import api.database as db

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(tags=["Risk Profiles & Velocity"])

# entity_type values accepted by the velocity endpoint
_VELOCITY_ENTITY_TYPES = {"customer", "merchant", "device", "ip"}


# ── Helper ─────────────────────────────────────────────────────────────────

def _validate_velocity_entity_type(entity_type: str) -> None:
    """
    Raise HTTP 422 if entity_type is not one of the supported velocity types.
    """
    if entity_type not in _VELOCITY_ENTITY_TYPES:
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "Invalid entity_type",
                "details": (
                    f"'{entity_type}' is not a valid velocity entity type. "
                    f"Must be one of: {sorted(_VELOCITY_ENTITY_TYPES)}"
                ),
            },
        )


# ══════════════════════════════════════════════════════════════════════════ #
#  CUSTOMER RISK PROFILE
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/customers/{customer_id}/risk-profile",
    summary="Get customer risk profile",
    description=(
        "Returns aggregated risk intelligence for a specific customer. "
        "If no transactions exist for the customer, a safe zero-value profile "
        "is returned (no 404/crash)."
    ),
)
def get_customer_risk_profile(
    customer_id: str,
    _: None = Depends(verify_api_key),
):
    """
    Aggregate prediction and case data for a single customer.

    Returns:
        customer_id           – the queried customer
        total_transactions    – all-time scored transaction count
        high_risk_count       – transactions scored HIGH risk
        confirmed_fraud_count – cases confirmed as fraud
        average_amount        – mean transaction amount
        last_transaction_time – ISO-8601 timestamp of latest transaction, or null
        risk_tier             – LOW | MEDIUM | HIGH | CRITICAL
        linked_devices        – distinct devices seen for this customer
        linked_merchants      – distinct merchants seen for this customer
        data_available        – false when no transactions exist yet
    """
    profile = db.get_customer_risk_profile(customer_id)

    # Determine whether any data actually exists for this customer
    data_available = (profile.get("total_transactions") or 0) > 0

    if not data_available:
        logger.info(f"Customer risk profile requested for unknown customer '{customer_id}' — returning empty profile.")

    return {
        "customer_id":           profile.get("customer_id", customer_id),
        "total_transactions":    profile.get("total_transactions", 0) or 0,
        "high_risk_count":       profile.get("high_risk_count", 0) or 0,
        "confirmed_fraud_count": profile.get("confirmed_fraud_count", 0) or 0,
        "average_amount":        profile.get("average_amount", 0.0) or 0.0,
        "last_transaction_time": profile.get("last_transaction_time"),
        "risk_tier":             profile.get("risk_tier", "LOW"),
        "linked_devices":        profile.get("linked_devices", 0) or 0,
        "linked_merchants":      profile.get("linked_merchants", 0) or 0,
        "data_available":        data_available,
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  MERCHANT RISK PROFILE
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/merchants/{merchant_id}/risk-profile",
    summary="Get merchant risk profile",
    description=(
        "Returns aggregated risk intelligence for a specific merchant. "
        "If no transactions exist for the merchant, a safe zero-value profile "
        "is returned (no 404/crash)."
    ),
)
def get_merchant_risk_profile(
    merchant_id: str,
    _: None = Depends(verify_api_key),
):
    """
    Aggregate prediction data for a single merchant.

    Returns:
        merchant_id        – the queried merchant
        total_transactions – all-time scored transaction count
        high_risk_count    – transactions scored HIGH risk
        fraud_rate         – percentage of transactions predicted as fraud (0–100)
        average_amount     – mean transaction amount
        risk_tier          – LOW | MEDIUM | HIGH | CRITICAL
        linked_customers   – distinct customers seen at this merchant
        data_available     – false when no transactions exist yet
    """
    profile = db.get_merchant_risk_profile(merchant_id)

    data_available = (profile.get("total_transactions") or 0) > 0

    if not data_available:
        logger.info(f"Merchant risk profile requested for unknown merchant '{merchant_id}' — returning empty profile.")

    return {
        "merchant_id":        profile.get("merchant_id", merchant_id),
        "total_transactions": profile.get("total_transactions", 0) or 0,
        "high_risk_count":    profile.get("high_risk_count", 0) or 0,
        "fraud_rate":         profile.get("fraud_rate", 0.0) or 0.0,
        "average_amount":     profile.get("average_amount", 0.0) or 0.0,
        "risk_tier":          profile.get("risk_tier", "LOW"),
        "linked_customers":   profile.get("linked_customers", 0) or 0,
        "data_available":     data_available,
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  VELOCITY
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/velocity/{entity_type}/{entity_value}",
    summary="Get real-time velocity signals for an entity",
    description=(
        "Returns transaction frequency, volume, fraud probability patterns, and "
        "risk signals observed in the last 1 hour and 24 hours for the given entity. "
        "Valid entity_type values: customer, merchant, device, ip."
    ),
)
def get_velocity(
    entity_type: str,
    entity_value: str,
    _: None = Depends(verify_api_key),
):
    """
    Real-time velocity intelligence for a specific entity.

    Returns:
        entity_type        – the queried entity type
        entity_value       – the queried entity value
        count_last_1h      – transaction count in the last 1 hour
        count_last_24h     – transaction count in the last 24 hours
        max_amount_24h     – maximum transaction amount in last 24 hours
        avg_fraud_prob_24h – average fraud probability score in last 24 hours
        unique_customers   – distinct customers linked to this entity (24h)
        unique_devices     – distinct devices linked to this entity (24h)
        unique_ips         – distinct IPs linked to this entity (24h)
        velocity_risk      – LOW | MEDIUM | HIGH
        signals            – list of triggered velocity signal codes
        data_available     – false when no recent transactions exist

    Raises:
        HTTP 422 — if entity_type is not one of: customer, merchant, device, ip
    """
    _validate_velocity_entity_type(entity_type)

    try:
        stats = db.get_velocity_stats(entity_type, entity_value)
    except ValueError as exc:
        # Defensive: db raises ValueError for invalid entity_type,
        # but we already validated above — re-raise as 422 just in case.
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "Invalid entity_type",
                "details": str(exc),
            },
        ) from exc

    data_available = (stats.get("count_last_24h") or 0) > 0

    if not data_available:
        logger.info(
            f"Velocity query for {entity_type}='{entity_value}' — "
            "no recent transactions found, returning empty velocity profile."
        )

    return {
        "entity_type":        stats.get("entity_type", entity_type),
        "entity_value":       stats.get("entity_value", entity_value),
        "count_last_1h":      stats.get("count_last_1h", 0) or 0,
        "count_last_24h":     stats.get("count_last_24h", 0) or 0,
        "max_amount_24h":     stats.get("max_amount_24h", 0.0) or 0.0,
        "avg_fraud_prob_24h": stats.get("avg_fraud_prob_24h", 0.0) or 0.0,
        "unique_customers":   stats.get("unique_customers", 0) or 0,
        "unique_devices":     stats.get("unique_devices", 0) or 0,
        "unique_ips":         stats.get("unique_ips", 0) or 0,
        "velocity_risk":      stats.get("velocity_risk", "LOW"),
        "signals":            stats.get("signals", []),
        "data_available":     data_available,
    }
