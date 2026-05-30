"""
api/routes/cases.py
===================
Case management endpoints.

Routes:
  POST   /cases                      — create a case manually
  GET    /cases                      — list all cases (filterable by status)
  GET    /cases/{case_id}            — get single case with notes
  PATCH  /cases/{case_id}/status     — update case status
  PATCH  /cases/{case_id}/assign     — assign case to analyst
  POST   /cases/{case_id}/notes      — add a note to a case
  POST   /cases/{case_id}/report     — get a structured case report

All routes are protected by API key authentication.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.middleware import verify_api_key
from api.schemas import (
    CaseCreateRequest,
    CaseResponse,
    CaseStatusUpdate,
    CaseAssignRequest,
    CaseNoteRequest,
    ErrorResponse,
)
import api.database as db

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(prefix="/cases", tags=["Cases"])

# Valid statuses — mirrors db.CASE_STATUSES
_VALID_STATUSES = {"OPEN", "UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"}


# ── Helper ─────────────────────────────────────────────────────────────────
def _fetch_or_404(case_id: str) -> dict:
    """Return case dict or raise 404."""
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(
            status_code=404,
            detail={"error": "Case not found", "details": f"No case with id '{case_id}'"},
        )
    return case


# ══════════════════════════════════════════════════════════════════════════ #
#  POST /cases
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "",
    response_model=CaseResponse,
    status_code=201,
    summary="Create a fraud case manually",
)
def create_case(
    body: CaseCreateRequest,
    _: None = Depends(verify_api_key),
):
    """
    Manually open a new fraud investigation case.
    Cases are also created automatically by /predict for REVIEW/BLOCK decisions.
    """
    try:
        case_id = db.create_case(body.model_dump())
        case    = db.get_case(case_id)
        return CaseResponse(**case, notes=[])
    except Exception as exc:
        logger.error(f"create_case error: {exc}")
        raise HTTPException(status_code=500, detail={"error": "Case creation failed", "details": str(exc)})


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /cases
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "",
    summary="List fraud cases",
    description="Returns cases ordered newest-first. Filter by status with ?status=OPEN etc.",
)
def list_cases(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: OPEN | UNDER_REVIEW | CONFIRMED_FRAUD | FALSE_POSITIVE | CLOSED",
    ),
    limit: int = Query(default=100, ge=1, le=500),
    _: None = Depends(verify_api_key),
):
    if status and status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "Invalid status",
                "details": f"Must be one of {sorted(_VALID_STATUSES)}",
            },
        )
    cases = db.list_cases(status=status, limit=limit)
    return {"cases": cases, "total": len(cases), "status_filter": status}


# ══════════════════════════════════════════════════════════════════════════ #
#  GET /cases/{case_id}
# ══════════════════════════════════════════════════════════════════════════ #

@router.get(
    "/{case_id}",
    response_model=CaseResponse,
    summary="Get a single fraud case",
)
def get_case(
    case_id: str,
    _: None = Depends(verify_api_key),
):
    """Returns the case record including all analyst notes."""
    case  = _fetch_or_404(case_id)
    notes = db.get_case_notes(case_id)
    return CaseResponse(**case, notes=notes)


# ══════════════════════════════════════════════════════════════════════════ #
#  PATCH /cases/{case_id}/status
# ══════════════════════════════════════════════════════════════════════════ #

@router.patch(
    "/{case_id}/status",
    summary="Update case status",
)
def update_case_status(
    case_id: str,
    body: CaseStatusUpdate,
    _: None = Depends(verify_api_key),
):
    """
    Transition a case to a new status.
    Valid statuses: OPEN, UNDER_REVIEW, CONFIRMED_FRAUD, FALSE_POSITIVE, CLOSED
    """
    _fetch_or_404(case_id)   # ensure it exists

    try:
        updated = db.update_case_status(case_id, body.status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "Invalid status", "details": str(exc)})

    if not updated:
        raise HTTPException(status_code=500, detail={"error": "Update failed"})

    return {
        "case_id": case_id,
        "status":  body.status,
        "message": f"Case {case_id} status updated to {body.status}.",
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  PATCH /cases/{case_id}/assign
# ══════════════════════════════════════════════════════════════════════════ #

@router.patch(
    "/{case_id}/assign",
    summary="Assign case to an analyst",
)
def assign_case(
    case_id: str,
    body: CaseAssignRequest,
    _: None = Depends(verify_api_key),
):
    """
    Assign a case to an analyst. Automatically moves status from
    OPEN → UNDER_REVIEW if the case has not already been progressed.
    """
    _fetch_or_404(case_id)

    updated = db.assign_case(case_id, body.assigned_to)
    if not updated:
        raise HTTPException(status_code=500, detail={"error": "Assignment failed"})

    return {
        "case_id":     case_id,
        "assigned_to": body.assigned_to,
        "message":     f"Case {case_id} assigned to {body.assigned_to}.",
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  POST /cases/{case_id}/notes
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "/{case_id}/notes",
    status_code=201,
    summary="Add a note to a case",
)
def add_note(
    case_id: str,
    body: CaseNoteRequest,
    _: None = Depends(verify_api_key),
):
    """Append an analyst note to an existing case."""
    _fetch_or_404(case_id)

    try:
        note_id = db.add_case_note(
            case_id=case_id,
            note=body.note,
            analyst_id=body.analyst_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "Note creation failed", "details": str(exc)})

    return {
        "note_id":     note_id,
        "case_id":     case_id,
        "analyst_id":  body.analyst_id,
        "message":     "Note added successfully.",
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  POST /cases/{case_id}/report
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "/{case_id}/report",
    summary="Generate a structured case report",
)
def case_report(
    case_id: str,
    _: None = Depends(verify_api_key),
):
    """
    Generate a structured investigation report for a case.
    Includes the case record, all notes, and a summary verdict.
    """
    case  = _fetch_or_404(case_id)
    notes = db.get_case_notes(case_id)

    # Verdict mapping
    verdict_map = {
        "CONFIRMED_FRAUD":  "⚠️  FRAUD CONFIRMED — escalate for chargeback and account action.",
        "FALSE_POSITIVE":   "✅  FALSE POSITIVE — transaction cleared. No further action.",
        "CLOSED":           "🔒  CASE CLOSED — investigation complete.",
        "UNDER_REVIEW":     "🔍  UNDER REVIEW — awaiting analyst decision.",
        "OPEN":             "📋  OPEN — not yet assigned.",
    }
    verdict = verdict_map.get(case.get("status", "OPEN"), "Unknown status.")

    return {
        "report_id":       f"RPT-{case_id}",
        "case":            case,
        "notes":           notes,
        "note_count":      len(notes),
        "verdict":         verdict,
        "generated_at":    db._now(),
    }
