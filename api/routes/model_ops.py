"""
api/routes/model_ops.py
=======================
Model metadata and performance endpoints.

Routes:
  GET /model/version      — model identity: version, algorithm, training date, threshold
  GET /model/performance  — full performance metrics: AUC, precision, recall, F1, etc.

Data source (in priority order):
  1. docs/model_card.json   — structured model card (primary source)
  2. api.main.predictor     — live model object attributes (version, threshold)
  3. Hard-coded fallbacks   — safe defaults when neither source is available
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from api.middleware import verify_api_key

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(prefix="/model", tags=["Model Operations"])

# Path to the model card JSON (relative to project root / CWD)
_MODEL_CARD_PATH = Path("docs/model_card.json")


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_model_card() -> dict[str, Any]:
    """
    Load and parse docs/model_card.json.
    Returns an empty dict if the file is missing or malformed.
    """
    try:
        if _MODEL_CARD_PATH.exists():
            with open(_MODEL_CARD_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as exc:
        logger.warning(f"Could not read model card at '{_MODEL_CARD_PATH}': {exc}")
    return {}


def _get_live_model_state() -> tuple[str | None, float | None, int | None]:
    """
    Attempt to read version, threshold, and feature_count from the live
    predictor loaded in api.main.

    Returns (version, threshold, feature_count) — any may be None on failure.
    """
    try:
        from api.main import MODEL_LOADED, predictor  # type: ignore[attr-defined]
        if not MODEL_LOADED or predictor is None:
            return None, None, None
        version       = getattr(predictor, "version", None)
        threshold     = getattr(predictor, "threshold", None)
        feature_names = getattr(predictor, "feature_names", None)
        feature_count = len(feature_names) if feature_names else None
        return version, threshold, feature_count
    except Exception:
        return None, None, None


def _extract_metadata(card: dict) -> dict[str, Any]:
    """
    Pull the flat metadata fields that both endpoints share from the model card.
    Falls back to safe defaults for every field independently.
    """
    details  = card.get("model_details", {})
    perf     = card.get("model_performance", {})
    metrics  = perf.get("metrics", {})
    dt       = perf.get("decision_threshold", {})

    # Live model may override version / threshold / feature_count
    live_version, live_threshold, live_feature_count = _get_live_model_state()

    version   = live_version   or details.get("version")      or "unknown"
    algorithm = details.get("algorithm")                       or "XGBoost"
    trained_on = details.get("date_trained")                   or None
    threshold = live_threshold if live_threshold is not None \
                else dt.get("value")                           or None

    # Feature count: live predictor > card > sensible default (30 = V1-V28 + Time + Amount)
    feature_count = live_feature_count or None

    return {
        "version":       version,
        "algorithm":     algorithm,
        "trained_on":    trained_on,
        "threshold":     threshold,
        "feature_count": feature_count,
        # Performance metrics — all optional, None if not in card
        "auc":       metrics.get("roc_auc"),
        "precision": metrics.get("precision"),
        "recall":    metrics.get("recall"),
        "f1_score":  metrics.get("f1_score"),
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /model/version
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/version",
    summary="Get model version and identity",
    description=(
        "Returns the current model's version string, algorithm family, training date, "
        "decision threshold, and feature count. "
        "Data is sourced from docs/model_card.json with live model state as override. "
        "Falls back to safe placeholder values if the model card is unavailable."
    ),
)
def get_model_version(
    _: None = Depends(verify_api_key),
):
    """
    Model identity metadata.

    Returns:
        model_version – semantic version string (e.g. "2.0.0")
        algorithm     – model family / algorithm name
        trained_on    – training date or period (e.g. "2025-04")
        threshold     – decision threshold for fraud classification (0–1)
        feature_count – number of input features the model expects, or null
    """
    card = _load_model_card()
    meta = _extract_metadata(card)

    logger.info(f"Model version requested: v{meta['version']}")

    return {
        "model_version": meta["version"],
        "algorithm":     meta["algorithm"],
        "trained_on":    meta["trained_on"],
        "threshold":     meta["threshold"],
        "feature_count": meta["feature_count"],
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /model/performance
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/performance",
    summary="Get model performance metrics",
    description=(
        "Returns the full set of evaluation metrics recorded in the model card: "
        "AUC-ROC, precision, recall, F1-score, decision threshold, and feature count. "
        "Falls back to null for any metric not present in the model card."
    ),
)
def get_model_performance(
    _: None = Depends(verify_api_key),
):
    """
    Model performance metrics from the model card.

    Returns:
        model_version – semantic version string
        algorithm     – model family / algorithm name
        trained_on    – training date or period
        threshold     – decision threshold used at inference time
        auc           – AUC-ROC on held-out test set, or null
        precision     – precision on held-out test set, or null
        recall        – recall on held-out test set, or null
        f1_score      – F1-score on held-out test set, or null
        feature_count – number of input features, or null
    """
    card = _load_model_card()
    meta = _extract_metadata(card)

    logger.info(f"Model performance requested: v{meta['version']}")

    return {
        "model_version": meta["version"],
        "algorithm":     meta["algorithm"],
        "trained_on":    meta["trained_on"],
        "threshold":     meta["threshold"],
        "auc":           meta["auc"],
        "precision":     meta["precision"],
        "recall":        meta["recall"],
        "f1_score":      meta["f1_score"],
        "feature_count": meta["feature_count"],
    }
