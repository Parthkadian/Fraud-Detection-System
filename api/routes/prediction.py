"""
api/routes/prediction.py
========================
Prediction router — /predict, /predict_batch, /predict_compare

Features added over the original main.py endpoints:
  - API key authentication on all routes
  - Blacklist/whitelist entity checks before ML inference
  - Velocity signals appended to reason codes
  - Enhanced PredictionResponse (decision, reason_codes, case_id, model_version,
    potential_loss, blocked_loss, false_positive_cost)
  - Auto case creation for REVIEW / BLOCK decisions
  - Auto alert creation for HIGH risk / BLOCK decisions
  - Champion-challenger /predict_compare with simulated challenger fallback
  - All results persisted to predictions table (api/database.py)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
import pandas as pd

import api.database as db
from api.middleware import verify_api_key
from api.schemas import (
    PredictCompareResponse,
    PredictionResponse,
    TransactionInput,
)

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(tags=["Prediction"])

# ── Cost constants (can be moved to env vars later) ───────────────────────
_FALSE_POSITIVE_COST = 15.0   # £15 per false positive (operational cost)
_FRAUD_MULTIPLIER    = 1.0    # potential_loss = amount * multiplier


# ══════════════════════════════════════════════════════════════════════════ #
#  Helpers
# ══════════════════════════════════════════════════════════════════════════ #

def _get_predictor():
    """Import predictor from main app state. Avoids circular imports."""
    from api.main import MODEL_ERROR, MODEL_LOADED, predictor
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_ERROR}")
    return predictor


def _resolve_decision(risk_level: str, rule_triggered: Optional[str]) -> str:
    """
    Map risk level + rule to a final decision string.
      LOW    → APPROVE
      MEDIUM → REVIEW
      HIGH   → BLOCK
    A triggered rule always forces at least REVIEW (BLOCK if risk already HIGH).
    """
    if risk_level == "HIGH":
        return "BLOCK"
    if risk_level == "MEDIUM" or rule_triggered:
        return "REVIEW"
    return "APPROVE"


def _build_reason_codes(
    risk_level: str,
    rule_triggered: Optional[str],
    blacklist_hit: Optional[str],
    whitelist_hit: bool,
    velocity_signals: list[str],
    fraud_probability: float,
) -> list[str]:
    """Compile a human-readable list of reason codes for the decision."""
    codes: list[str] = []

    if blacklist_hit:
        codes.append(f"BLACKLIST_MATCH:{blacklist_hit}")
    if whitelist_hit:
        codes.append("WHITELIST_MATCH")
    if rule_triggered:
        codes.append(f"RULE:{rule_triggered}")
    if risk_level == "HIGH":
        codes.append("HIGH_FRAUD_PROBABILITY")
    elif risk_level == "MEDIUM":
        codes.append("ELEVATED_FRAUD_PROBABILITY")
    if fraud_probability >= 0.9:
        codes.append("VERY_HIGH_MODEL_CONFIDENCE")
    codes.extend(velocity_signals)

    return codes if codes else ["MODEL_SCORE_ONLY"]


def _check_entities(txn: TransactionInput) -> tuple[Optional[str], bool]:
    """
    Check transaction metadata against blacklist and whitelist.

    Returns:
        (blacklist_hit: str | None, whitelist_hit: bool)
    blacklist_hit is the entity_type:value string that matched, or None.
    """
    checks = [
        ("customer", txn.customer_id),
        ("merchant", txn.merchant_id),
        ("device",   txn.device_id),
        ("ip",       txn.ip_address),
        ("country",  txn.country),
    ]

    blacklist_hit: Optional[str] = None
    whitelist_hit: bool = False

    for entity_type, entity_value in checks:
        if not entity_value:
            continue
        if db.is_blacklisted(entity_type, entity_value):
            blacklist_hit = f"{entity_type}:{entity_value}"
            break   # first blacklist hit wins
        if db.is_whitelisted(entity_type, entity_value):
            whitelist_hit = True

    return blacklist_hit, whitelist_hit


def _get_velocity_signals(txn: TransactionInput) -> list[str]:
    """
    Run quick velocity checks on available entity identifiers.
    Returns a list of signal strings for reason_codes.
    """
    signals: list[str] = []
    checks = [
        ("customer", txn.customer_id),
        ("device",   txn.device_id),
        ("ip",       txn.ip_address),
    ]
    for entity_type, entity_value in checks:
        if not entity_value:
            continue
        try:
            stats = db.get_velocity_stats(entity_type, entity_value)
            signals.extend(stats.get("signals", []))
        except Exception:
            pass    # velocity check failure must never block prediction
    return list(set(signals))   # deduplicate


def _compute_costs(
    amount: float,
    decision: str,
    prediction: int,
) -> tuple[float, float, float]:
    """
    Estimate potential_loss, blocked_loss, false_positive_cost.

    potential_loss      = amount (what could be lost if this is real fraud)
    blocked_loss        = amount if decision==BLOCK (we stopped it)
    false_positive_cost = flat operational cost if decision is REVIEW or BLOCK
                          but this could be a false positive
    """
    potential_loss      = round(amount * _FRAUD_MULTIPLIER, 2)
    blocked_loss        = round(amount, 2) if decision == "BLOCK" else 0.0
    false_positive_cost = round(_FALSE_POSITIVE_COST, 2) if decision in ("REVIEW", "BLOCK") else 0.0
    return potential_loss, blocked_loss, false_positive_cost


def _maybe_create_case(
    txn: TransactionInput,
    pred_result: dict,
    decision: str,
) -> Optional[str]:
    """Create a case record for REVIEW or BLOCK decisions. Returns case_id or None."""
    if decision not in ("REVIEW", "BLOCK"):
        return None
    try:
        case_id = db.create_case({
            "transaction_id":    txn.transaction_id,
            "customer_id":       txn.customer_id,
            "merchant_id":       txn.merchant_id,
            "fraud_probability": pred_result.get("fraud_probability"),
            "risk_level":        pred_result.get("risk_level"),
            "decision":          decision,
            "status":            "OPEN",
        })
        return case_id
    except Exception as exc:
        logger.error(f"Case creation failed: {exc}")
        return None


def _maybe_create_alert(
    txn: TransactionInput,
    risk_level: str,
    decision: str,
    case_id: Optional[str],
) -> None:
    """Auto-create an alert for HIGH risk or BLOCK decisions."""
    if risk_level != "HIGH" and decision != "BLOCK":
        return
    try:
        severity = "CRITICAL" if decision == "BLOCK" else "HIGH"
        msg = (
            f"{severity} fraud alert: transaction {txn.transaction_id or 'N/A'} "
            f"scored {risk_level} risk with decision {decision}."
        )
        db.create_alert({
            "transaction_id": txn.transaction_id,
            "case_id":        case_id,
            "severity":       severity,
            "message":        msg,
        })
    except Exception as exc:
        logger.error(f"Alert creation failed: {exc}")


def _build_full_response(
    txn: TransactionInput,
    raw: dict,
    blacklist_hit: Optional[str],
    whitelist_hit: bool,
    velocity_signals: list[str],
    predictor_version: str,
) -> dict:
    """
    Assemble the full enriched prediction dict from raw model output + metadata.
    """
    risk_level     = raw["risk_level"]
    rule_triggered = raw.get("rule_triggered")

    # Blacklist override: force HIGH / BLOCK regardless of model score
    if blacklist_hit:
        risk_level     = "HIGH"
        rule_triggered = rule_triggered or "BLACKLIST_MATCH"

    # Whitelist downgrade: reduce from BLOCK → REVIEW unless very high confidence
    if whitelist_hit and not blacklist_hit and raw["fraud_probability"] < 0.9:
        if risk_level == "HIGH":
            risk_level = "MEDIUM"

    decision = _resolve_decision(risk_level, rule_triggered)

    reason_codes = _build_reason_codes(
        risk_level, rule_triggered, blacklist_hit,
        whitelist_hit, velocity_signals, raw["fraud_probability"],
    )

    potential_loss, blocked_loss, fp_cost = _compute_costs(
        txn.Amount, decision, raw["prediction"]
    )

    case_id = _maybe_create_case(txn, {**raw, "risk_level": risk_level}, decision)
    _maybe_create_alert(txn, risk_level, decision, case_id)

    result = {
        # Original fields
        "fraud_probability": raw["fraud_probability"],
        "prediction":        raw["prediction"],
        "risk_level":        risk_level,
        "rule_triggered":    rule_triggered,
        "latency_ms":        raw.get("latency_ms"),
        # New fields
        "transaction_id":      txn.transaction_id,
        "customer_id":         txn.customer_id,
        "merchant_id":         txn.merchant_id,
        "decision":            decision,
        "reason_codes":        reason_codes,
        "model_version":       predictor_version,
        "case_id":             case_id,
        "potential_loss":      potential_loss,
        "blocked_loss":        blocked_loss,
        "false_positive_cost": fp_cost,
    }

    # Persist to platform DB (non-blocking — errors logged, not raised)
    try:
        db.insert_prediction({
            **result,
            "amount":           txn.Amount,
            "device_id":        txn.device_id,
            "ip_address":       txn.ip_address,
            "country":          txn.country,
            "channel":          txn.channel,
            "transaction_type": txn.transaction_type,
            "currency":         txn.currency,
        })
    except Exception as exc:
        logger.error(f"Prediction persistence failed: {exc}")

    return result


# ══════════════════════════════════════════════════════════════════════════ #
#  POST /predict
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a single transaction",
    description=(
        "Score a single credit-card transaction. Returns fraud probability, "
        "decision (APPROVE/REVIEW/BLOCK), reason codes, and auto-creates a "
        "case + alert for high-risk transactions."
    ),
)
def predict(
    transaction: TransactionInput,
    _: None = Depends(verify_api_key),
):
    predictor = _get_predictor()

    try:
        blacklist_hit, whitelist_hit = _check_entities(transaction)
        velocity_signals             = _get_velocity_signals(transaction)
        raw                          = predictor.predict(transaction.model_dump())

        result = _build_full_response(
            transaction, raw,
            blacklist_hit, whitelist_hit, velocity_signals,
            predictor.version,
        )
        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════ #
#  POST /predict_batch
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "/predict_batch",
    summary="Score a batch of transactions",
    description="Score multiple transactions at once. Returns a list of scored records.",
)
def predict_batch(
    transactions: list[dict],
    _: None = Depends(verify_api_key),
):
    from api.main import MODEL_ERROR, MODEL_LOADED, batch_predictor
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_ERROR}")

    try:
        df         = pd.DataFrame(transactions)
        result_df  = batch_predictor.predict_dataframe(df)
        return result_df.to_dict(orient="records")
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════ #
#  POST /predict_compare  (champion-challenger)
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "/predict_compare",
    response_model=PredictCompareResponse,
    summary="Champion-challenger prediction comparison",
    description=(
        "Run champion (production) and challenger (simulated) models side-by-side. "
        "If no real challenger exists, threshold is shifted ±0.05 to simulate one."
    ),
)
def predict_compare(
    transaction: TransactionInput,
    _: None = Depends(verify_api_key),
):
    predictor = _get_predictor()

    try:
        # ── Champion: normal prediction ───────────────────────────────
        blacklist_hit, whitelist_hit = _check_entities(transaction)
        velocity_signals             = _get_velocity_signals(transaction)
        raw_champion                 = predictor.predict(transaction.model_dump())

        champion = _build_full_response(
            transaction, raw_champion,
            blacklist_hit, whitelist_hit, velocity_signals,
            predictor.version,
        )

        # ── Challenger: simulate by shifting threshold ────────────────
        original_threshold      = predictor.threshold
        predictor.threshold     = max(0.01, original_threshold - 0.05)   # lower → stricter
        raw_challenger          = predictor.predict(transaction.model_dump())
        predictor.threshold     = original_threshold                      # restore

        challenger = _build_full_response(
            transaction, raw_challenger,
            blacklist_hit, whitelist_hit, velocity_signals,
            f"{predictor.version}-challenger",
        )

        disagreement = champion["decision"] != challenger["decision"]

        if disagreement:
            recommendation = (
                f"Champion says {champion['decision']} but challenger says "
                f"{challenger['decision']}. Recommend manual review."
            )
        else:
            recommendation = f"Both models agree: {champion['decision']}. High confidence."

        return PredictCompareResponse(
            champion_result=champion,
            challenger_result=challenger,
            disagreement=disagreement,
            recommendation=recommendation,
            challenger_mode="simulated",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"predict_compare error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))
