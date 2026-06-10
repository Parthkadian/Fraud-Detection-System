"""
api/middleware.py
=================
Production-grade middleware for the Fraud Detection Platform.

Contains:
  1. RequestIDMiddleware     — attaches X-Request-ID to every request/response
  2. RequestTimingMiddleware — logs latency, sets X-Process-Time header
  3. RateLimitMiddleware     — IP-based rate limiting (in-memory, no Redis required)
  4. verify_api_key()        — FastAPI dependency for protecting sensitive endpoints

Environment variables:
  FRAUD_API_KEY          — if set, all protected endpoints require this key
                           in the X-API-Key header. If unset, dev mode (warning only).
  RATE_LIMIT_PER_MINUTE  — max requests per IP per minute (default: 60)
"""

from __future__ import annotations

from collections import defaultdict
import logging
import os
from threading import Lock
import time
from typing import Optional
import uuid

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("fraud_detection_logger")

# ══════════════════════════════════════════════════════════════════════════ #
#  Config from environment
# ══════════════════════════════════════════════════════════════════════════ #

_API_KEY: Optional[str] = os.getenv("FRAUD_API_KEY")
_RATE_LIMIT: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

if not _API_KEY:
    logger.warning(
        "⚠️  FRAUD_API_KEY is not set. Running in development mode — "
        "all endpoints are publicly accessible. Set FRAUD_API_KEY in "
        "production to enable authentication."
    )


# ══════════════════════════════════════════════════════════════════════════ #
#  1. Request ID Middleware  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════ #

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique X-Request-ID header to every request/response
    for distributed tracing and log correlation.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ══════════════════════════════════════════════════════════════════════════ #
#  2. Request Timing Middleware  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════ #

class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Logs the processing time for every request and exposes it via
    the X-Process-Time response header.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        logger.info(
            f"{request.method} {request.url.path} → "
            f"{response.status_code} ({duration_ms:.1f}ms)"
        )
        return response


# ══════════════════════════════════════════════════════════════════════════ #
#  3. Rate Limit Middleware  (new — in-memory, no Redis required)
# ══════════════════════════════════════════════════════════════════════════ #

class _RateLimitBucket:
    """
    Sliding-window counter for a single IP address.
    Thread-safe using a Lock.
    """
    __slots__ = ("count", "window_start", "lock")

    def __init__(self) -> None:
        self.count: int = 0
        self.window_start: float = time.monotonic()
        self.lock: Lock = Lock()

    def check(self, limit: int) -> tuple[bool, int]:
        """
        Returns (allowed: bool, remaining: int).
        Resets the window every 60 seconds.
        """
        now = time.monotonic()
        with self.lock:
            if now - self.window_start >= 60.0:
                # New window
                self.count = 0
                self.window_start = now

            self.count += 1
            remaining = max(0, limit - self.count)
            allowed = self.count <= limit
        return allowed, remaining


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    IP-based sliding-window rate limiter.

    - Reads limit from RATE_LIMIT_PER_MINUTE (default: 60)
    - Returns HTTP 429 with Retry-After header when exceeded
    - Adds X-RateLimit-Limit and X-RateLimit-Remaining to every response
    - Skips rate limiting for /health and /metrics (observability endpoints)
    """

    # Paths that are never rate-limited
    _EXEMPT_PATHS = {"/health", "/metrics", "/", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, limit: int = _RATE_LIMIT) -> None:
        super().__init__(app)
        self._limit = limit
        self._buckets: dict[str, _RateLimitBucket] = defaultdict(_RateLimitBucket)
        self._lock = Lock()

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP, respecting X-Forwarded-For for reverse-proxy setups.
        Falls back to direct client host.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip exempt paths
        if request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        # Thread-safe bucket access
        with self._lock:
            bucket = self._buckets[client_ip]

        allowed, remaining = bucket.check(self._limit)

        if not allowed:
            logger.warning(f"Rate limit exceeded for IP {client_ip} on {request.url.path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error":   "Too Many Requests",
                    "details": f"Rate limit of {self._limit} requests/minute exceeded.",
                    "request_id": getattr(request.state, "request_id", None),
                },
                headers={
                    "X-RateLimit-Limit":     str(self._limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After":           "60",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"]     = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# ══════════════════════════════════════════════════════════════════════════ #
#  4. API Key Authentication Dependency
# ══════════════════════════════════════════════════════════════════════════ #

# FastAPI security scheme — shows a padlock on protected endpoints in /docs
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    """
    FastAPI dependency to protect sensitive endpoints.

    Behaviour:
    - If FRAUD_API_KEY env var is NOT set → development mode, all requests pass
      through with a warning (safe for local dev, never do this in production).
    - If FRAUD_API_KEY IS set → the X-API-Key header must match exactly,
      otherwise HTTP 403 is raised.

    Usage in a route:
        from api.middleware import verify_api_key
        from fastapi import Depends

        @router.post("/predict")
        def predict(
            transaction: TransactionInput,
            _: None = Depends(verify_api_key),
        ):
            ...
    """
    if not _API_KEY:
        # Dev mode: no key configured — allow all but log a warning once per call
        logger.debug("API key check skipped (FRAUD_API_KEY not configured — dev mode)")
        return

    if api_key != _API_KEY:
        raise HTTPException(
            status_code=403,
            detail={
                "error":   "Forbidden",
                "details": "Invalid or missing X-API-Key header.",
            },
        )
