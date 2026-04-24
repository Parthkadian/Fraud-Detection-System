"""
API Router
==========
Reserved for future route separation.  Currently all routes are defined
directly in ``api.main`` for simplicity.  As the API grows, routes can
be migrated here using ``fastapi.APIRouter`` for modular organisation.

Example
-------
    from fastapi import APIRouter
    router = APIRouter(prefix="/api/v2", tags=["v2"])

    @router.get("/status")
    def status():
        return {"status": "ok"}
"""
