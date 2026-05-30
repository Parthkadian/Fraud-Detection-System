"""
api/routes/explainability.py
============================
SHAP-based explainability endpoints.

Routes:
  POST /explain         — per-transaction SHAP explanation (top-N features)
  GET  /explain/global  — global feature importance from model internals

Notes:
  - POST /explain delegates to ShapExplainer.explain_single() from api.main.
  - GET  /explain/global derives global importance from the XGBoost model's
    built-in feature importances (gain-based). No separate global SHAP run
    is needed, keeping latency and memory cost low.
  - All state is read from api.main to avoid circular imports and to stay
    consistent with the live loaded model — no other files are modified.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.middleware import verify_api_key
from api.schemas import TransactionInput, ExplanationResponse, GlobalExplanationResponse

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(tags=["Explainability"])


# ── Helpers: safe access to api.main state ─────────────────────────────────

def _get_shap_explainer():
    """
    Return the live ShapExplainer from api.main.
    Raises HTTP 503 if the model/explainer is not loaded.
    """
    try:
        from api.main import shap_explainer, MODEL_LOADED, MODEL_ERROR  # type: ignore[attr-defined]
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "Service unavailable", "details": str(exc)},
        ) from exc

    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail={
                "error":   "Model not loaded",
                "details": f"Cannot explain: {MODEL_ERROR}",
            },
        )
    if shap_explainer is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error":   "SHAP explainer unavailable",
                "details": "ShapExplainer failed to initialise at startup.",
            },
        )
    return shap_explainer


def _get_predictor_state() -> tuple[Any, str | None]:
    """
    Return (predictor_object, model_version) from api.main.
    Returns (None, None) on any failure — callers handle gracefully.
    """
    try:
        from api.main import predictor, MODEL_LOADED  # type: ignore[attr-defined]
        if not MODEL_LOADED or predictor is None:
            return None, None
        version = getattr(predictor, "version", None)
        return predictor, version
    except Exception:
        return None, None


def _build_global_importance(predictor) -> tuple[list[dict], str]:
    """
    Extract global feature importances from the XGBoost model stored in predictor.

    Strategy (in priority order):
      1. model.feature_importances_  (sklearn API — gain-based, normalised)
      2. model.get_booster().get_score(importance_type='gain')
      3. shap_explainer.feature_order with equal weight (last-resort fallback)

    Returns (features_list, data_source) where data_source is one of:
      "real"       — importances extracted from live model
      "fallback"   — feature names with equal placeholder weight
      "unavailable"— no feature information available at all
    """
    model = getattr(predictor, "model", None) or getattr(predictor, "_model", None)

    # ── Strategy 1: sklearn feature_importances_ ──────────────────────────
    if model is not None and hasattr(model, "feature_importances_"):
        try:
            importances = model.feature_importances_
            names: list[str] = []
            if hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
            elif hasattr(model, "get_booster") and model.get_booster().feature_names:
                names = model.get_booster().feature_names
            else:
                names = [f"feature_{i}" for i in range(len(importances))]

            pairs = sorted(
                zip(names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
            features = [
                {"feature": name, "importance": round(float(imp), 6), "rank": rank + 1}
                for rank, (name, imp) in enumerate(pairs)
            ]
            return features, "real"
        except Exception as exc:
            logger.warning(f"feature_importances_ extraction failed: {exc}")

    # ── Strategy 2: XGBoost booster get_score ────────────────────────────
    if model is not None and hasattr(model, "get_booster"):
        try:
            score_dict = model.get_booster().get_score(importance_type="gain")
            if score_dict:
                total = sum(score_dict.values()) or 1.0
                pairs_sorted = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
                features = [
                    {
                        "feature":    name,
                        "importance": round(gain / total, 6),
                        "rank":       rank + 1,
                    }
                    for rank, (name, gain) in enumerate(pairs_sorted)
                ]
                return features, "real"
        except Exception as exc:
            logger.warning(f"get_booster().get_score() failed: {exc}")

    # ── Strategy 3: feature_names from predictor with equal weight ────────
    feature_names: list[str] = getattr(predictor, "feature_names", None) or []
    if feature_names:
        equal_weight = round(1.0 / len(feature_names), 6)
        features = [
            {"feature": name, "importance": equal_weight, "rank": rank + 1}
            for rank, name in enumerate(feature_names)
        ]
        return features, "fallback"

    # ── Strategy 4: nothing available ────────────────────────────────────
    return [], "unavailable"


# ══════════════════════════════════════════════════════════════════════════ #
#  POST /explain
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="SHAP explanation for a single transaction",
    description=(
        "Generates a SHAP-based explanation for a single transaction, returning "
        "the top contributing features and their SHAP values. "
        "Requires the model and SHAP explainer to be loaded (HTTP 503 otherwise)."
    ),
)
def explain_single(
    transaction: TransactionInput,
    _: None = Depends(verify_api_key),
):
    """
    Per-transaction SHAP feature attribution.

    Returns:
        top_features – list of {feature, shap_value} dicts sorted by |shap_value| desc
    """
    explainer = _get_shap_explainer()

    try:
        result = explainer.explain_single(transaction.model_dump(), top_n=10)
        return ExplanationResponse(top_features=result["top_features"])
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"SHAP explain_single failed: {exc}")
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "Explanation failed",
                "details": str(exc),
            },
        )


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /explain/global
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/explain/global",
    response_model=GlobalExplanationResponse,
    summary="Global feature importance across all predictions",
    description=(
        "Returns the model's global feature importances derived from XGBoost's "
        "gain-based scores. This is a fast, stateless alternative to running SHAP "
        "across many samples. Falls back to equal-weight placeholders if the model "
        "internals are inaccessible, and marks data_source accordingly."
    ),
)
def explain_global(
    _: None = Depends(verify_api_key),
):
    """
    Global feature importance from model internals.

    Returns:
        top_global_features  – ranked list of {feature, importance, rank}
        explanation_method   – "feature_importance" | "unavailable"
        model_version        – live model version string, or null
        data_source          – "real" | "fallback" | "unavailable"

    Notes:
        - data_source="real"        → importances from live XGBoost model
        - data_source="fallback"    → equal weights assigned to feature names
        - data_source="unavailable" → model not loaded or no feature info found
    """
    predictor, model_version = _get_predictor_state()

    if predictor is None:
        logger.warning("Global explanation requested but model is not loaded — returning unavailable.")
        return GlobalExplanationResponse(
            top_global_features=[],
            explanation_method="unavailable",
            model_version=None,
            data_source="unavailable",
        )

    features, data_source = _build_global_importance(predictor)

    explanation_method = "feature_importance" if data_source in ("real", "fallback") else "unavailable"

    logger.info(
        f"Global explanation: {len(features)} features, "
        f"method={explanation_method}, source={data_source}."
    )

    return GlobalExplanationResponse(
        top_global_features=features,
        explanation_method=explanation_method,
        model_version=model_version,
        data_source=data_source,
    )
