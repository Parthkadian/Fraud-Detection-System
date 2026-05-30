"""
api/routes/review.py
====================
Human review queue endpoints.

Routes:
  GET   /review/queue               — all OPEN + UNDER_REVIEW cases, newest first
  PATCH /review/{case_id}/decision  — analyst submits final decision
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.middleware import verify_api_key
from api.schemas import ReviewDecisionRequest
import api.database as db

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(prefix="/review", tags=["Review Queue"])

# Statuses that belong in the human review queue
_REVIEW_STATUSES = {"OPEN", "UNDER_REVIEW"}

# Analyst decision values accepted by the PATCH endpoint.
# APPROVED      → case is closed as legitimate (maps to CLOSED in DB)
# BLOCKED       → case is closed as confirmed fraud (maps to CONFIRMED_FRAUD in DB)
# CONFIRMED_FRAUD → direct confirmation of fraud
# FALSE_POSITIVE  → was flagged incorrectly
_ANALYST_DECISIONS = {"APPROVED", "BLOCKED", "CONFIRMED_FRAUD", "FALSE_POSITIVE"}

# Map analyst decision → canonical DB case status
_DECISION_TO_STATUS: dict[str, str] = {
    "APPROVED":       "CLOSED",
    "BLOCKED":        "CONFIRMED_FRAUD",
    "CONFIRMED_FRAUD": "CONFIRMED_FRAUD",
    "FALSE_POSITIVE":  "FALSE_POSITIVE",
}


# ── GET /review/queue ──────────────────────────────────────────────────────

@router.get(
    "/queue",
    summary="Get human review queue",
    description=(
        "Returns all cases with status OPEN or UNDER_REVIEW, combined and sorted "
        "newest first. Use this endpoint to drive the analyst review dashboard."
    ),
)
def review_queue(
    _: None = Depends(verify_api_key),
):
    """
    Fetch all cases awaiting human review.

    Returns:
        queue      – combined list of OPEN and UNDER_REVIEW cases (newest first)
        total      – total count in the queue
        open       – count of OPEN cases
        in_review  – count of UNDER_REVIEW cases
    """
    open_cases   = db.list_cases(status="OPEN",         limit=200)
    review_cases = db.list_cases(status="UNDER_REVIEW", limit=200)

    queue = open_cases + review_cases
    # Sort combined list newest first by created_at timestamp
    queue.sort(key=lambda c: c.get("created_at", ""), reverse=True)

    logger.info(
        f"Review queue fetched: {len(open_cases)} OPEN, "
        f"{len(review_cases)} UNDER_REVIEW cases."
    )

    return {
        "queue":     queue,
        "total":     len(queue),
        "open":      len(open_cases),
        "in_review": len(review_cases),
    }


# ── PATCH /review/{case_id}/decision ──────────────────────────────────────

@router.patch(
    "/{case_id}/decision",
    summary="Submit analyst decision for a case",
    description=(
        "Analyst resolves a case in the review queue by submitting a decision. "
        "Accepted decisions: APPROVED, BLOCKED, CONFIRMED_FRAUD, FALSE_POSITIVE. "
        "An optional analyst note can be attached."
    ),
)
def submit_review_decision(
    case_id: str,
    body: ReviewDecisionRequest,
    _: None = Depends(verify_api_key),
):
    """
    Submit a final analyst decision for a queued case.

    Decision values:
        APPROVED        → marks case as CLOSED (legitimate transaction)
        BLOCKED         → marks case as CONFIRMED_FRAUD and blocks entity
        CONFIRMED_FRAUD → marks case as CONFIRMED_FRAUD
        FALSE_POSITIVE  → marks case as FALSE_POSITIVE (model error)

    Rules:
        - Only OPEN or UNDER_REVIEW cases can receive a decision.
        - If the case does not exist, HTTP 404 is returned.
        - If an invalid decision is provided, HTTP 422 is returned.
        - If a note is provided it is persisted against the case.
        - If analyst_id is provided and the case is unassigned, it is auto-assigned.
    """
    # ── 1. Validate decision value ─────────────────────────────────────────
    if body.decision not in _ANALYST_DECISIONS:
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "Invalid decision",
                "details": (
                    f"'{body.decision}' is not a valid analyst decision. "
                    f"Must be one of: {sorted(_ANALYST_DECISIONS)}"
                ),
            },
        )

    # ── 2. Look up the case ────────────────────────────────────────────────
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(
            status_code=404,
            detail={
                "error":   "Case not found",
                "details": f"No case with id '{case_id}' exists.",
            },
        )

    # ── 3. Confirm the case is in the review queue ─────────────────────────
    current_status = case.get("status", "")
    if current_status not in _REVIEW_STATUSES:
        raise HTTPException(
            status_code=409,
            detail={
                "error":   "Case not in review queue",
                "details": (
                    f"Case '{case_id}' has status '{current_status}'. "
                    "Only OPEN or UNDER_REVIEW cases can receive a decision."
                ),
            },
        )

    # ── 4. Auto-assign analyst if case is unassigned ───────────────────────
    if body.analyst_id and not case.get("assigned_to"):
        db.assign_case(case_id, body.analyst_id)
        logger.info(f"Case {case_id} auto-assigned to analyst '{body.analyst_id}'.")

    # ── 5. Map analyst decision → DB status and update ─────────────────────
    new_status = _DECISION_TO_STATUS[body.decision]
    db.update_case_status(case_id, new_status)
    logger.info(
        f"Case {case_id} resolved: decision='{body.decision}' "
        f"→ status='{new_status}' by analyst='{body.analyst_id}'."
    )

    # ── 6. Persist analyst note if provided ───────────────────────────────
    note_id: str | None = None
    if body.notes and body.notes.strip():
        note_id = db.add_case_note(
            case_id=case_id,
            note=body.notes.strip(),
            analyst_id=body.analyst_id,
        )
        logger.info(f"Note {note_id} added to case {case_id}.")

    # ── 7. Return clean JSON ───────────────────────────────────────────────
    return {
        "case_id":     case_id,
        "decision":    body.decision,
        "new_status":  new_status,
        "analyst_id":  body.analyst_id,
        "note_id":     note_id,
        "note_added":  note_id is not None,
        "message":     (
            f"Case '{case_id}' resolved as '{body.decision}' "
            f"(status → '{new_status}')."
        ),
    }
