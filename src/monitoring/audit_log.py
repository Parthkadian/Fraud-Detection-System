"""
Prediction Audit Logger
========================
SQLite-backed audit trail for every prediction made by the API.

Every call to ``/predict`` is logged with:
- Unique prediction ID (UUID)
- UTC timestamp
- SHA-256 hash of input features (privacy-safe)
- Fraud probability and binary prediction
- Risk level
- Whether a business rule was triggered
- Model version
- Inference latency in milliseconds

This satisfies PCI-DSS / financial compliance requirements
that mandate a durable, queryable audit trail for all
automated credit-risk decisions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger("fraud_detection_logger")

_DEFAULT_DB = "data/audit_log.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_audit (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    input_hash      TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    prediction      INTEGER NOT NULL,
    risk_level      TEXT NOT NULL,
    rule_triggered  TEXT,
    model_version   TEXT,
    latency_ms      REAL,
    amount          REAL,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

_INSERT_SQL = """
INSERT INTO prediction_audit
    (id, timestamp, input_hash, fraud_probability, prediction,
     risk_level, rule_triggered, model_version, latency_ms, amount)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class PredictionAuditLogger:
    """
    Logs every prediction to a local SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file.  Created automatically if absent.
    model_version : str
        Tag embedded in every audit record (e.g. ``"2.0.0"``).
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB,
        model_version: str = "2.0.0",
    ) -> None:
        self.db_path = db_path
        self.model_version = model_version
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"AuditLogger initialised → {db_path}")

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE_SQL)

    @staticmethod
    def _hash_input(input_data: dict) -> str:
        """SHA-256 of sorted JSON — deterministic, privacy-safe."""
        raw = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def log(
        self,
        input_data: dict,
        fraud_probability: float,
        prediction: int,
        risk_level: str,
        latency_ms: float = 0.0,
        rule_triggered: Optional[str] = None,
    ) -> str:
        """
        Insert one prediction record and return its UUID.

        Parameters
        ----------
        input_data : dict
            Raw feature dict (will be hashed, not stored as-is).
        fraud_probability : float
            Model output probability.
        prediction : int
            Binary label (0 or 1).
        risk_level : str
            LOW / MEDIUM / HIGH.
        latency_ms : float
            End-to-end inference latency in milliseconds.
        rule_triggered : str, optional
            Name of the business rule that overrode the ML prediction,
            or ``None`` if no rule fired.

        Returns
        -------
        str
            The UUID of the new audit record.
        """
        record_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        input_hash = self._hash_input(input_data)
        amount = float(input_data.get("Amount", 0.0))

        try:
            with self._conn() as conn:
                conn.execute(
                    _INSERT_SQL,
                    (
                        record_id,
                        ts,
                        input_hash,
                        round(fraud_probability, 6),
                        prediction,
                        risk_level,
                        rule_triggered,
                        self.model_version,
                        round(latency_ms, 2),
                        round(amount, 2),
                    ),
                )
        except Exception as exc:
            logger.error(f"Audit log write failed: {exc}")

        return record_id

    def get_recent(self, limit: int = 100) -> list[dict]:
        """Return the most recent *limit* audit records as dicts."""
        sql = """
            SELECT id, timestamp, input_hash, fraud_probability,
                   prediction, risk_level, rule_triggered,
                   model_version, latency_ms, amount
            FROM   prediction_audit
            ORDER  BY created_at DESC
            LIMIT  ?
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(sql, (limit,)).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error(f"Audit log read failed: {exc}")
            return []

    def get_stats(self) -> dict:
        """Return aggregate statistics for the health dashboard."""
        sql = """
            SELECT
                COUNT(*)                                AS total_predictions,
                SUM(prediction)                         AS total_fraud,
                ROUND(AVG(fraud_probability), 4)        AS avg_probability,
                ROUND(AVG(latency_ms), 2)               AS avg_latency_ms,
                ROUND(MIN(latency_ms), 2)               AS min_latency_ms,
                ROUND(MAX(latency_ms), 2)               AS max_latency_ms,
                COUNT(DISTINCT DATE(timestamp))         AS active_days
            FROM prediction_audit
        """
        try:
            with self._conn() as conn:
                row = conn.execute(sql).fetchone()
            return dict(row) if row else {}
        except Exception as exc:
            logger.error(f"Audit stats failed: {exc}")
            return {}
