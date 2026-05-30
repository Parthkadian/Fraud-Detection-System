"""
Structured Logger
=================
Production-grade logging with:

- Rotating file handler (configurable size & backup count)
- **JSON structured output mode** for log aggregators
  (ELK, Datadog, Grafana Loki, etc.)
- Plain-text console output for development
- Correlation ID injection via thread-local context

Usage
-----
Plain text (default)::

    logger = setup_logger("logs/app.log")

JSON mode (production)::

    logger = setup_logger("logs/app.log", json_format=True)

Setting a correlation ID::

    from src.monitoring.logger import set_correlation_id
    set_correlation_id("req-abc-123")
"""

import json
import logging
import logging.handlers
import os
import threading
from datetime import datetime, timezone
from typing import Optional

# Thread-local store for correlation IDs
_local = threading.local()


def set_correlation_id(cid: str) -> None:
    """Attach a correlation ID to the current thread context."""
    _local.correlation_id = cid


def get_correlation_id() -> str:
    """Retrieve the current thread's correlation ID (or 'N/A')."""
    return getattr(_local, "correlation_id", "N/A")


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": get_correlation_id(),
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _PlainFormatter(logging.Formatter):
    """Human-readable plain-text formatter with correlation ID."""

    def format(self, record: logging.LogRecord) -> str:
        record.correlation_id = get_correlation_id()
        return super().format(record)


def setup_logger(
    log_file: str,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    json_format: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return the project-wide logger.

    Parameters
    ----------
    log_file : str
        Path to the rotating log file.
    max_bytes : int
        Maximum file size before rotation (default 5 MB).
    backup_count : int
        Number of rotated backup files to retain.
    json_format : bool
        If ``True``, emit JSON-structured log lines to the file
        handler (suitable for log aggregators).  The console
        handler always uses plain text for readability.
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance named ``"fraud_detection_logger"``.
    """
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    logger = logging.getLogger("fraud_detection_logger")
    logger.setLevel(level)

    if logger.handlers:
        # Already configured — avoid duplicate handlers
        return logger

    plain_formatter = _PlainFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | [%(correlation_id)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    json_formatter = _JsonFormatter()

    # ── File handler (JSON or plain) ─────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(json_formatter if json_format else plain_formatter)

    # ── Console handler (always plain) ───────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(plain_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(
        f"Logger initialised | file={log_file} | "
        f"format={'json' if json_format else 'plain'} | level={logging.getLevelName(level)}"
    )
    return logger