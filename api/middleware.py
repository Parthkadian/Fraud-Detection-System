"""
API Middleware
==============
Production-grade middleware for request tracking, timing, and structured logging.
"""

import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("fraud_detection_logger")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique ``X-Request-ID`` header to every request/response
    for distributed tracing and log correlation.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Logs the processing time for every request and exposes it via
    the ``X-Process-Time`` response header.
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
