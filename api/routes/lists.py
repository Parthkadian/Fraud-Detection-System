"""
api/routes/lists.py
===================
Blacklist and whitelist management endpoints.

Routes:
  POST   /blacklist                               — add entity to blacklist
  GET    /blacklist                               — list all blacklisted entities
  DELETE /blacklist/{entity_type}/{entity_value}  — remove entity from blacklist

  POST   /whitelist                               — add entity to whitelist
  GET    /whitelist                               — list all whitelisted entities
  DELETE /whitelist/{entity_type}/{entity_value}  — remove entity from whitelist

Valid entity_type values: customer, merchant, device, ip, country
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.middleware import verify_api_key
from api.schemas import ListEntityRequest, ListEntityResponse
import api.database as db

logger = logging.getLogger("fraud_detection_logger")

router = APIRouter(tags=["Blacklist & Whitelist"])

# Canonical set of allowed entity types (mirrors db.ENTITY_TYPES)
_VALID_ENTITY_TYPES = {"customer", "merchant", "device", "ip", "country"}


# ── Helper ─────────────────────────────────────────────────────────────────

def _validate_entity_type(entity_type: str) -> None:
    """
    Raise HTTP 422 if entity_type is not one of the supported values.
    Called by both blacklist and whitelist endpoints.
    """
    if entity_type not in _VALID_ENTITY_TYPES:
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "Invalid entity_type",
                "details": (
                    f"'{entity_type}' is not a valid entity type. "
                    f"Must be one of: {sorted(_VALID_ENTITY_TYPES)}"
                ),
            },
        )


# ══════════════════════════════════════════════════════════════════════════ #
#  BLACKLIST
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "/blacklist",
    status_code=201,
    summary="Add entity to blacklist",
    description=(
        "Block a specific entity (customer, merchant, device, IP address, or country). "
        "Duplicate entries are silently ignored — the existing record is retained."
    ),
)
def add_to_blacklist(
    body: ListEntityRequest,
    _: None = Depends(verify_api_key),
):
    """
    Add an entity to the blacklist.

    Returns the new (or existing) record ID and a confirmation message.
    Raises HTTP 422 if entity_type is invalid.
    """
    _validate_entity_type(body.entity_type)

    entry_id = db.add_blacklist(
        entity_type=body.entity_type,
        entity_value=body.entity_value,
        reason=body.reason,
        added_by=body.added_by,
    )

    logger.info(
        f"Blacklist: added {body.entity_type}='{body.entity_value}' "
        f"(id={entry_id}, added_by={body.added_by})."
    )

    return {
        "id":           entry_id,
        "entity_type":  body.entity_type,
        "entity_value": body.entity_value,
        "reason":       body.reason,
        "added_by":     body.added_by,
        "message":      (
            f"Entity {body.entity_type}='{body.entity_value}' "
            "added to blacklist."
        ),
    }


@router.get(
    "/blacklist",
    summary="List all blacklisted entities",
    description="Returns every entity currently on the blacklist, newest first.",
)
def get_blacklist(
    _: None = Depends(verify_api_key),
):
    """
    Retrieve the full blacklist.

    Returns:
        blacklist – list of entity records
        total     – count of records
    """
    entries = db.list_blacklist()

    logger.info(f"Blacklist fetched: {len(entries)} entries.")

    return {
        "blacklist": entries,
        "total":     len(entries),
    }


@router.delete(
    "/blacklist/{entity_type}/{entity_value}",
    summary="Remove entity from blacklist",
    description=(
        "Un-block a previously blacklisted entity. "
        "Returns HTTP 404 if the entity is not currently on the blacklist."
    ),
)
def remove_from_blacklist(
    entity_type: str,
    entity_value: str,
    _: None = Depends(verify_api_key),
):
    """
    Remove an entity from the blacklist.

    Raises:
        HTTP 422 — if entity_type is not valid
        HTTP 404 — if the entity is not on the blacklist
    """
    _validate_entity_type(entity_type)

    removed = db.remove_blacklist(entity_type, entity_value)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail={
                "error":   "Entity not found",
                "details": (
                    f"No blacklist entry for {entity_type}='{entity_value}'."
                ),
            },
        )

    logger.info(f"Blacklist: removed {entity_type}='{entity_value}'.")

    return {
        "entity_type":  entity_type,
        "entity_value": entity_value,
        "removed":      True,
        "message":      (
            f"Entity {entity_type}='{entity_value}' "
            "removed from blacklist."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  WHITELIST
# ══════════════════════════════════════════════════════════════════════════ #

@router.post(
    "/whitelist",
    status_code=201,
    summary="Add entity to whitelist",
    description=(
        "Trust a specific entity (customer, merchant, device, IP address, or country). "
        "Duplicate entries are silently ignored — the existing record is retained."
    ),
)
def add_to_whitelist(
    body: ListEntityRequest,
    _: None = Depends(verify_api_key),
):
    """
    Add an entity to the whitelist.

    Returns the new (or existing) record ID and a confirmation message.
    Raises HTTP 422 if entity_type is invalid.
    """
    _validate_entity_type(body.entity_type)

    entry_id = db.add_whitelist(
        entity_type=body.entity_type,
        entity_value=body.entity_value,
        reason=body.reason,
        added_by=body.added_by,
    )

    logger.info(
        f"Whitelist: added {body.entity_type}='{body.entity_value}' "
        f"(id={entry_id}, added_by={body.added_by})."
    )

    return {
        "id":           entry_id,
        "entity_type":  body.entity_type,
        "entity_value": body.entity_value,
        "reason":       body.reason,
        "added_by":     body.added_by,
        "message":      (
            f"Entity {body.entity_type}='{body.entity_value}' "
            "added to whitelist."
        ),
    }


@router.get(
    "/whitelist",
    summary="List all whitelisted entities",
    description="Returns every entity currently on the whitelist, newest first.",
)
def get_whitelist(
    _: None = Depends(verify_api_key),
):
    """
    Retrieve the full whitelist.

    Returns:
        whitelist – list of entity records
        total     – count of records
    """
    entries = db.list_whitelist()

    logger.info(f"Whitelist fetched: {len(entries)} entries.")

    return {
        "whitelist": entries,
        "total":     len(entries),
    }


@router.delete(
    "/whitelist/{entity_type}/{entity_value}",
    summary="Remove entity from whitelist",
    description=(
        "Un-trust a previously whitelisted entity. "
        "Returns HTTP 404 if the entity is not currently on the whitelist."
    ),
)
def remove_from_whitelist(
    entity_type: str,
    entity_value: str,
    _: None = Depends(verify_api_key),
):
    """
    Remove an entity from the whitelist.

    Raises:
        HTTP 422 — if entity_type is not valid
        HTTP 404 — if the entity is not on the whitelist
    """
    _validate_entity_type(entity_type)

    removed = db.remove_whitelist(entity_type, entity_value)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail={
                "error":   "Entity not found",
                "details": (
                    f"No whitelist entry for {entity_type}='{entity_value}'."
                ),
            },
        )

    logger.info(f"Whitelist: removed {entity_type}='{entity_value}'.")

    return {
        "entity_type":  entity_type,
        "entity_value": entity_value,
        "removed":      True,
        "message":      (
            f"Entity {entity_type}='{entity_value}' "
            "removed from whitelist."
        ),
    }
