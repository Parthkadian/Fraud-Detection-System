"""
api/database.py
===============
Dual-adapter persistence layer for the Fraud Detection Platform.

Adapter selection (automatic, via environment variable):
  • PostgreSQL — when DATABASE_URL starts with "postgresql://" (Docker / Railway)
  • SQLite     — fallback for local development and CI (no external DB needed)

Manages 7 tables:
  predictions  – every scored transaction
  cases        – fraud case records
  case_notes   – analyst notes attached to cases
  feedback     – human-labelled ground-truth for retraining
  alerts       – high-severity auto-generated alerts
  blacklist    – blocked entities (customer/merchant/device/ip/country)
  whitelist    – trusted entities

NOTE: The existing audit trail (data/audit_log.db / src/monitoring/audit_log.py)
is NOT replaced. This is an additive layer for case management & intelligence
features. Both can coexist.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, List, Optional

logger = logging.getLogger("fraud_detection_logger")

# ── Adapter detection ──────────────────────────────────────────────────────
DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", "")
_USE_POSTGRES: bool = bool(DATABASE_URL and DATABASE_URL.startswith("postgresql"))

if _USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
        logger.info("🐘 Database adapter: PostgreSQL (psycopg2)")
    except ImportError:
        logger.error("psycopg2 not installed — falling back to SQLite")
        _USE_POSTGRES = False
else:
    logger.info("🗄️  Database adapter: SQLite (development/CI mode)")

# ── SQLite fallback path ───────────────────────────────────────────────────
DB_PATH: str = os.getenv("FRAUD_DB_PATH", "data/fraud_platform.db")


# ── Utility ─────────────────────────────────────────────────────────────────
def _now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "") -> str:
    """Generate a short UUID, optionally prefixed (e.g. 'CASE-')."""
    return f"{prefix}{uuid.uuid4().hex[:12].upper()}"


def _adapt_sql(sql: str) -> str:
    """Convert SQLite-style '?' placeholders to psycopg2-style '%s'."""
    return sql.replace("?", "%s") if _USE_POSTGRES else sql


# ── PostgreSQL connection wrapper ──────────────────────────────────────────
class _PgConn:
    """
    Thin wrapper around a psycopg2 connection that presents the same
    interface as sqlite3.Connection so all downstream code is unchanged:

        conn.execute(sql, params)  → cursor (with fetchone / fetchall)
        conn.commit() / rollback() / close()
    """

    def __init__(self, raw_conn: Any) -> None:
        self._conn = raw_conn
        self._cur = raw_conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        )

    def execute(self, sql: str, params: tuple = ()) -> Any:
        self._cur.execute(_adapt_sql(sql), params)
        return self._cur

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._cur.close()
        self._conn.close()


# ── Connection context manager ─────────────────────────────────────────────
@contextmanager
def get_conn() -> Generator[Any, None, None]:
    """
    Yield a database connection that works identically for both adapters:

    • PostgreSQL  — psycopg2 + RealDictCursor wrapped in _PgConn
    • SQLite      — sqlite3.Connection with sqlite3.Row row_factory

    Usage (same for both adapters):
        with get_conn() as conn:
            rows = conn.execute("SELECT * FROM cases WHERE id = ?", (id,)).fetchall()
    """
    if _USE_POSTGRES:
        conn = _PgConn(psycopg2.connect(DATABASE_URL))
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(DB_PATH, check_same_thread=False)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA journal_mode=WAL;")
        raw.execute("PRAGMA foreign_keys=ON;")
        try:
            yield raw
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()


# ══════════════════════════════════════════════════════════════════════════ #
#  CREATE TABLE statements
# ══════════════════════════════════════════════════════════════════════════ #

_SQL_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  TEXT PRIMARY KEY,
    transaction_id      TEXT,
    customer_id         TEXT,
    merchant_id         TEXT,
    device_id           TEXT,
    ip_address          TEXT,
    country             TEXT,
    channel             TEXT,
    transaction_type    TEXT,
    currency            TEXT DEFAULT 'GBP',
    amount              REAL NOT NULL DEFAULT 0.0,
    fraud_probability   REAL NOT NULL,
    prediction          INTEGER NOT NULL,
    risk_level          TEXT NOT NULL,
    decision            TEXT NOT NULL,
    reason_codes        TEXT,
    rule_triggered      TEXT,
    model_version       TEXT,
    latency_ms          REAL,
    potential_loss      REAL DEFAULT 0.0,
    blocked_loss        REAL DEFAULT 0.0,
    false_positive_cost REAL DEFAULT 0.0,
    case_id             TEXT,
    created_at          TEXT NOT NULL
);
"""

_SQL_CASES = """
CREATE TABLE IF NOT EXISTS cases (
    id                  TEXT PRIMARY KEY,
    transaction_id      TEXT,
    customer_id         TEXT,
    merchant_id         TEXT,
    fraud_probability   REAL,
    risk_level          TEXT,
    decision            TEXT,
    status              TEXT NOT NULL DEFAULT 'OPEN',
    assigned_to         TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
"""

_SQL_CASE_NOTES = """
CREATE TABLE IF NOT EXISTS case_notes (
    id          TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    analyst_id  TEXT,
    note        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

_SQL_FEEDBACK = """
CREATE TABLE IF NOT EXISTS feedback (
    id              TEXT PRIMARY KEY,
    transaction_id  TEXT,
    case_id         TEXT,
    actual_label    TEXT NOT NULL,
    analyst_id      TEXT,
    notes           TEXT,
    created_at      TEXT NOT NULL
);
"""

_SQL_ALERTS = """
CREATE TABLE IF NOT EXISTS alerts (
    id              TEXT PRIMARY KEY,
    transaction_id  TEXT,
    case_id         TEXT,
    severity        TEXT NOT NULL,
    message         TEXT NOT NULL,
    acknowledged    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
"""

_SQL_BLACKLIST = """
CREATE TABLE IF NOT EXISTS blacklist (
    id              TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,
    entity_value    TEXT NOT NULL,
    reason          TEXT,
    added_by        TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(entity_type, entity_value)
);
"""

_SQL_WHITELIST = """
CREATE TABLE IF NOT EXISTS whitelist (
    id              TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,
    entity_value    TEXT NOT NULL,
    reason          TEXT,
    added_by        TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(entity_type, entity_value)
);
"""

_SQL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_predictions_customer   ON predictions(customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_merchant   ON predictions(merchant_id);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_device     ON predictions(device_id);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_ip         ON predictions(ip_address);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_created    ON predictions(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_cases_status           ON cases(status);",
    "CREATE INDEX IF NOT EXISTS idx_cases_customer         ON cases(customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged    ON alerts(acknowledged);",
    "CREATE INDEX IF NOT EXISTS idx_blacklist_lookup       ON blacklist(entity_type, entity_value);",
    "CREATE INDEX IF NOT EXISTS idx_whitelist_lookup       ON whitelist(entity_type, entity_value);",
    "CREATE INDEX IF NOT EXISTS idx_feedback_transaction   ON feedback(transaction_id);",
]


# ══════════════════════════════════════════════════════════════════════════ #
#  init_db — called once at API startup
# ══════════════════════════════════════════════════════════════════════════ #

def init_db() -> None:
    """
    Create all tables and indexes if they do not exist.
    Safe to call multiple times (idempotent).
    Called from api/main.py lifespan on startup.
    """
    logger.info(f"Initialising fraud platform database → {DB_PATH}")
    tables = [
        _SQL_PREDICTIONS, _SQL_CASES, _SQL_CASE_NOTES,
        _SQL_FEEDBACK, _SQL_ALERTS, _SQL_BLACKLIST, _SQL_WHITELIST,
    ]
    with get_conn() as conn:
        for ddl in tables:
            conn.execute(ddl)
        for idx in _SQL_INDEXES:
            conn.execute(idx)
    logger.info("✅ Database tables and indexes ready.")


# ══════════════════════════════════════════════════════════════════════════ #
#  PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════ #

def insert_prediction(data: dict) -> str:
    """
    Insert a scored transaction record.
    ``data`` should contain all prediction response fields.
    Returns the generated prediction ID.
    """
    pred_id = _new_id("PRED-")
    reason_codes = data.get("reason_codes", [])
    # Store reason_codes as JSON string if it's a list
    if isinstance(reason_codes, list):
        reason_codes = json.dumps(reason_codes)

    sql = """
        INSERT INTO predictions (
            id, transaction_id, customer_id, merchant_id, device_id,
            ip_address, country, channel, transaction_type, currency,
            amount, fraud_probability, prediction, risk_level, decision,
            reason_codes, rule_triggered, model_version, latency_ms,
            potential_loss, blocked_loss, false_positive_cost, case_id, created_at
        ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """
    with get_conn() as conn:
        conn.execute(sql, (
            pred_id,
            data.get("transaction_id"),
            data.get("customer_id"),
            data.get("merchant_id"),
            data.get("device_id"),
            data.get("ip_address"),
            data.get("country"),
            data.get("channel"),
            data.get("transaction_type"),
            data.get("currency", "GBP"),
            float(data.get("amount", 0.0)),
            float(data.get("fraud_probability", 0.0)),
            int(data.get("prediction", 0)),
            data.get("risk_level", "LOW"),
            data.get("decision", "APPROVE"),
            reason_codes,
            data.get("rule_triggered"),
            data.get("model_version"),
            data.get("latency_ms"),
            float(data.get("potential_loss", 0.0)),
            float(data.get("blocked_loss", 0.0)),
            float(data.get("false_positive_cost", 0.0)),
            data.get("case_id"),
            _now(),
        ))
    return pred_id


def get_recent_predictions(limit: int = 100) -> list[dict]:
    """Return the most recent *limit* prediction records as dicts."""
    sql = """
        SELECT * FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    result = []
    for r in rows:
        row = dict(r)
        # Deserialise reason_codes back to list
        if row.get("reason_codes"):
            try:
                row["reason_codes"] = json.loads(row["reason_codes"])
            except (ValueError, TypeError):
                row["reason_codes"] = []
        result.append(row)
    return result


# ══════════════════════════════════════════════════════════════════════════ #
#  CASES
# ══════════════════════════════════════════════════════════════════════════ #

# Valid case statuses
CASE_STATUSES = {"OPEN", "UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"}


def create_case(data: dict) -> str:
    """
    Insert a new case record.
    Returns the generated case ID (e.g. 'CASE-AB12CD34EF56').
    """
    case_id = _new_id("CASE-")
    now = _now()
    sql = """
        INSERT INTO cases (
            id, transaction_id, customer_id, merchant_id,
            fraud_probability, risk_level, decision,
            status, assigned_to, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """
    with get_conn() as conn:
        conn.execute(sql, (
            case_id,
            data.get("transaction_id"),
            data.get("customer_id"),
            data.get("merchant_id"),
            data.get("fraud_probability"),
            data.get("risk_level"),
            data.get("decision"),
            data.get("status", "OPEN"),
            data.get("assigned_to"),
            now,
            now,
        ))
    return case_id


def get_case(case_id: str) -> Optional[dict]:
    """Return a single case by ID, or None if not found."""
    sql = "SELECT * FROM cases WHERE id = ?"
    with get_conn() as conn:
        row = conn.execute(sql, (case_id,)).fetchone()
    return dict(row) if row else None


def list_cases(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    """
    Return cases, optionally filtered by status.
    Ordered newest first.
    """
    if status:
        sql = "SELECT * FROM cases WHERE status = ? ORDER BY created_at DESC LIMIT ?"
        params = (status, limit)
    else:
        sql = "SELECT * FROM cases ORDER BY created_at DESC LIMIT ?"
        params = (limit,)
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def update_case_status(case_id: str, status: str) -> bool:
    """
    Update the status of a case.
    Returns True if a row was updated, False if case not found.
    Raises ValueError for invalid status values.
    """
    if status not in CASE_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {CASE_STATUSES}")
    sql = "UPDATE cases SET status = ?, updated_at = ? WHERE id = ?"
    with get_conn() as conn:
        cursor = conn.execute(sql, (status, _now(), case_id))
    return cursor.rowcount > 0


def assign_case(case_id: str, assigned_to: str) -> bool:
    """
    Assign a case to an analyst.
    Also sets status to UNDER_REVIEW if currently OPEN.
    Returns True if updated.
    """
    sql = """
        UPDATE cases
        SET assigned_to = ?,
            status = CASE WHEN status = 'OPEN' THEN 'UNDER_REVIEW' ELSE status END,
            updated_at = ?
        WHERE id = ?
    """
    with get_conn() as conn:
        cursor = conn.execute(sql, (assigned_to, _now(), case_id))
    return cursor.rowcount > 0


# ══════════════════════════════════════════════════════════════════════════ #
#  CASE NOTES
# ══════════════════════════════════════════════════════════════════════════ #

def add_case_note(
    case_id: str,
    note: str,
    analyst_id: Optional[str] = None,
) -> str:
    """
    Append a note to an existing case.
    Returns the new note ID.
    """
    note_id = _new_id("NOTE-")
    sql = """
        INSERT INTO case_notes (id, case_id, analyst_id, note, created_at)
        VALUES (?,?,?,?,?)
    """
    with get_conn() as conn:
        conn.execute(sql, (note_id, case_id, analyst_id, note, _now()))
        # Touch the case updated_at
        conn.execute(
            "UPDATE cases SET updated_at = ? WHERE id = ?",
            (_now(), case_id),
        )
    return note_id


def get_case_notes(case_id: str) -> list[dict]:
    """Return all notes for a case, oldest first."""
    sql = """
        SELECT * FROM case_notes
        WHERE case_id = ?
        ORDER BY created_at ASC
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (case_id,)).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════ #
#  FEEDBACK
# ══════════════════════════════════════════════════════════════════════════ #

def insert_feedback(data: dict) -> str:
    """
    Store analyst ground-truth label for a transaction.
    Returns the feedback record ID.
    """
    fb_id = _new_id("FB-")
    sql = """
        INSERT INTO feedback (id, transaction_id, case_id, actual_label,
                              analyst_id, notes, created_at)
        VALUES (?,?,?,?,?,?,?)
    """
    with get_conn() as conn:
        conn.execute(sql, (
            fb_id,
            data.get("transaction_id"),
            data.get("case_id"),
            data.get("actual_label", "LEGIT"),
            data.get("analyst_id"),
            data.get("notes"),
            _now(),
        ))
    return fb_id


def get_feedback(limit: int = 200) -> list[dict]:
    """Return recent feedback records (for retraining pipeline use)."""
    sql = "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?"
    with get_conn() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════ #
#  ALERTS
# ══════════════════════════════════════════════════════════════════════════ #

def create_alert(data: dict) -> str:
    """
    Create a new alert record.
    Returns the alert ID.
    """
    alert_id = _new_id("ALRT-")
    sql = """
        INSERT INTO alerts (id, transaction_id, case_id, severity,
                            message, acknowledged, created_at)
        VALUES (?,?,?,?,?,0,?)
    """
    with get_conn() as conn:
        conn.execute(sql, (
            alert_id,
            data.get("transaction_id"),
            data.get("case_id"),
            data.get("severity", "HIGH"),
            data.get("message", "Fraud alert triggered"),
            _now(),
        ))
    return alert_id


def get_recent_alerts(limit: int = 100) -> list[dict]:
    """Return most recent alerts, unacknowledged first."""
    sql = """
        SELECT * FROM alerts
        ORDER BY acknowledged ASC, created_at DESC
        LIMIT ?
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: str) -> bool:
    """
    Mark an alert as acknowledged.
    Returns True if the alert was found and updated.
    """
    sql = "UPDATE alerts SET acknowledged = 1 WHERE id = ?"
    with get_conn() as conn:
        cursor = conn.execute(sql, (alert_id,))
    return cursor.rowcount > 0


# ══════════════════════════════════════════════════════════════════════════ #
#  BLACKLIST
# ══════════════════════════════════════════════════════════════════════════ #

# Valid entity types for both lists
ENTITY_TYPES = {"customer", "merchant", "device", "ip", "country"}


def add_blacklist(
    entity_type: str,
    entity_value: str,
    reason: Optional[str] = None,
    added_by: Optional[str] = None,
) -> str:
    """
    Add an entity to the blacklist.
    Returns the record ID, or raises ValueError for invalid entity_type.
    Silently ignores duplicates (uses INSERT OR IGNORE).
    """
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Invalid entity_type '{entity_type}'. Must be one of {ENTITY_TYPES}")
    entry_id = _new_id("BL-")
    sql = """
        INSERT OR IGNORE INTO blacklist
            (id, entity_type, entity_value, reason, added_by, created_at)
        VALUES (?,?,?,?,?,?)
    """
    with get_conn() as conn:
        conn.execute(sql, (entry_id, entity_type, entity_value, reason, added_by, _now()))
    return entry_id


def remove_blacklist(entity_type: str, entity_value: str) -> bool:
    """Remove an entity from the blacklist. Returns True if removed."""
    sql = "DELETE FROM blacklist WHERE entity_type = ? AND entity_value = ?"
    with get_conn() as conn:
        cursor = conn.execute(sql, (entity_type, entity_value))
    return cursor.rowcount > 0


def list_blacklist() -> list[dict]:
    """Return all blacklist entries, newest first."""
    sql = "SELECT * FROM blacklist ORDER BY created_at DESC"
    with get_conn() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def is_blacklisted(entity_type: str, entity_value: str) -> bool:
    """Return True if the entity is on the blacklist."""
    sql = "SELECT 1 FROM blacklist WHERE entity_type = ? AND entity_value = ? LIMIT 1"
    with get_conn() as conn:
        row = conn.execute(sql, (entity_type, entity_value)).fetchone()
    return row is not None


# ══════════════════════════════════════════════════════════════════════════ #
#  WHITELIST
# ══════════════════════════════════════════════════════════════════════════ #

def add_whitelist(
    entity_type: str,
    entity_value: str,
    reason: Optional[str] = None,
    added_by: Optional[str] = None,
) -> str:
    """
    Add an entity to the whitelist.
    Returns the record ID, or raises ValueError for invalid entity_type.
    Silently ignores duplicates (uses INSERT OR IGNORE).
    """
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Invalid entity_type '{entity_type}'. Must be one of {ENTITY_TYPES}")
    entry_id = _new_id("WL-")
    sql = """
        INSERT OR IGNORE INTO whitelist
            (id, entity_type, entity_value, reason, added_by, created_at)
        VALUES (?,?,?,?,?,?)
    """
    with get_conn() as conn:
        conn.execute(sql, (entry_id, entity_type, entity_value, reason, added_by, _now()))
    return entry_id


def remove_whitelist(entity_type: str, entity_value: str) -> bool:
    """Remove an entity from the whitelist. Returns True if removed."""
    sql = "DELETE FROM whitelist WHERE entity_type = ? AND entity_value = ?"
    with get_conn() as conn:
        cursor = conn.execute(sql, (entity_type, entity_value))
    return cursor.rowcount > 0


def list_whitelist() -> list[dict]:
    """Return all whitelist entries, newest first."""
    sql = "SELECT * FROM whitelist ORDER BY created_at DESC"
    with get_conn() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def is_whitelisted(entity_type: str, entity_value: str) -> bool:
    """Return True if the entity is on the whitelist."""
    sql = "SELECT 1 FROM whitelist WHERE entity_type = ? AND entity_value = ? LIMIT 1"
    with get_conn() as conn:
        row = conn.execute(sql, (entity_type, entity_value)).fetchone()
    return row is not None


# ══════════════════════════════════════════════════════════════════════════ #
#  ANALYTICS — Dashboard Summary
# ══════════════════════════════════════════════════════════════════════════ #

def get_dashboard_summary() -> dict:
    """
    Aggregate stats for GET /dashboard/summary.
    Counts predictions made today (UTC), case statuses, and average latency.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with get_conn() as conn:
        # Today's prediction counts by risk level
        row = conn.execute("""
            SELECT
                COUNT(*)                                          AS total_predictions_today,
                SUM(CASE WHEN risk_level='HIGH'   THEN 1 ELSE 0 END) AS high_risk_today,
                SUM(CASE WHEN risk_level='MEDIUM' THEN 1 ELSE 0 END) AS medium_risk_today,
                SUM(CASE WHEN risk_level='LOW'    THEN 1 ELSE 0 END) AS low_risk_today,
                ROUND(AVG(latency_ms), 2)                         AS average_latency_ms
            FROM predictions
            WHERE DATE(created_at) = ?
        """, (today,)).fetchone()

        totals = dict(row) if row else {}

        # Review queue size (OPEN + UNDER_REVIEW cases)
        queue_row = conn.execute("""
            SELECT COUNT(*) AS review_queue_size
            FROM cases
            WHERE status IN ('OPEN', 'UNDER_REVIEW')
        """).fetchone()

        # Confirmed fraud and false positive counts (all time)
        status_row = conn.execute("""
            SELECT
                SUM(CASE WHEN status='CONFIRMED_FRAUD'  THEN 1 ELSE 0 END) AS confirmed_fraud_count,
                SUM(CASE WHEN status='FALSE_POSITIVE'   THEN 1 ELSE 0 END) AS false_positive_count
            FROM cases
        """).fetchone()

    return {
        "total_predictions_today": totals.get("total_predictions_today", 0),
        "high_risk_today":         totals.get("high_risk_today", 0),
        "medium_risk_today":       totals.get("medium_risk_today", 0),
        "low_risk_today":          totals.get("low_risk_today", 0),
        "average_latency_ms":      totals.get("average_latency_ms") or 0.0,
        "review_queue_size":       dict(queue_row).get("review_queue_size", 0) if queue_row else 0,
        "confirmed_fraud_count":   dict(status_row).get("confirmed_fraud_count", 0) if status_row else 0,
        "false_positive_count":    dict(status_row).get("false_positive_count", 0) if status_row else 0,
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  ANALYTICS — Customer Risk Profile
# ══════════════════════════════════════════════════════════════════════════ #

def get_customer_risk_profile(customer_id: str) -> dict:
    """
    Aggregate prediction and case data for a single customer.
    Used by GET /customers/{customer_id}/risk-profile.
    """
    with get_conn() as conn:
        pred_row = conn.execute("""
            SELECT
                COUNT(*)                                              AS total_transactions,
                SUM(CASE WHEN risk_level='HIGH' THEN 1 ELSE 0 END)   AS high_risk_count,
                ROUND(AVG(amount), 2)                                 AS average_amount,
                MAX(created_at)                                       AS last_transaction_time,
                COUNT(DISTINCT device_id)                             AS linked_devices,
                COUNT(DISTINCT merchant_id)                           AS linked_merchants
            FROM predictions
            WHERE customer_id = ?
        """, (customer_id,)).fetchone()

        fraud_row = conn.execute("""
            SELECT COUNT(*) AS confirmed_fraud_count
            FROM cases
            WHERE customer_id = ? AND status = 'CONFIRMED_FRAUD'
        """, (customer_id,)).fetchone()

    p = dict(pred_row) if pred_row else {}
    f = dict(fraud_row) if fraud_row else {}

    total = p.get("total_transactions", 0) or 0
    high  = p.get("high_risk_count", 0) or 0

    # Risk tier: CRITICAL > HIGH > MEDIUM > LOW based on high-risk ratio
    ratio = (high / total) if total > 0 else 0.0
    if ratio >= 0.5:
        risk_tier = "CRITICAL"
    elif ratio >= 0.3:
        risk_tier = "HIGH"
    elif ratio >= 0.1:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    return {
        "customer_id":           customer_id,
        "total_transactions":    total,
        "high_risk_count":       high,
        "confirmed_fraud_count": f.get("confirmed_fraud_count", 0) or 0,
        "average_amount":        p.get("average_amount") or 0.0,
        "last_transaction_time": p.get("last_transaction_time"),
        "risk_tier":             risk_tier,
        "linked_devices":        p.get("linked_devices", 0) or 0,
        "linked_merchants":      p.get("linked_merchants", 0) or 0,
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  ANALYTICS — Merchant Risk Profile
# ══════════════════════════════════════════════════════════════════════════ #

def get_merchant_risk_profile(merchant_id: str) -> dict:
    """
    Aggregate prediction data for a single merchant.
    Used by GET /merchants/{merchant_id}/risk-profile.
    """
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                              AS total_transactions,
                SUM(CASE WHEN risk_level='HIGH' THEN 1 ELSE 0 END)   AS high_risk_count,
                SUM(CASE WHEN prediction=1      THEN 1 ELSE 0 END)   AS fraud_count,
                ROUND(AVG(amount), 2)                                 AS average_amount,
                COUNT(DISTINCT customer_id)                           AS linked_customers
            FROM predictions
            WHERE merchant_id = ?
        """, (merchant_id,)).fetchone()

    r = dict(row) if row else {}
    total     = r.get("total_transactions", 0) or 0
    fraud_cnt = r.get("fraud_count", 0) or 0
    high      = r.get("high_risk_count", 0) or 0

    fraud_rate = round((fraud_cnt / total) * 100, 2) if total > 0 else 0.0
    ratio      = (high / total) if total > 0 else 0.0

    if ratio >= 0.5 or fraud_rate >= 40:
        risk_tier = "CRITICAL"
    elif ratio >= 0.3 or fraud_rate >= 20:
        risk_tier = "HIGH"
    elif ratio >= 0.1 or fraud_rate >= 5:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    return {
        "merchant_id":        merchant_id,
        "total_transactions": total,
        "high_risk_count":    high,
        "fraud_rate":         fraud_rate,
        "average_amount":     r.get("average_amount") or 0.0,
        "risk_tier":          risk_tier,
        "linked_customers":   r.get("linked_customers", 0) or 0,
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  ANALYTICS — Velocity Checks
# ══════════════════════════════════════════════════════════════════════════ #

def get_velocity_stats(entity_type: str, entity_value: str) -> dict:
    """
    Return velocity signals for a given entity.
    Used by GET /velocity/{entity_type}/{entity_value}.

    Checks:
    - Transaction count in last 1 hour and last 24 hours
    - Unique customers / devices sharing the same IP or device
    - Max transaction amount in last 24 hours
    - Average fraud probability in last 24 hours
    """
    # Map entity type to the correct column name in predictions
    col_map = {
        "customer": "customer_id",
        "merchant": "merchant_id",
        "device":   "device_id",
        "ip":       "ip_address",
        "country":  "country",
    }
    if entity_type not in col_map:
        raise ValueError(f"Invalid entity_type '{entity_type}'. Must be one of {set(col_map)}")

    col = col_map[entity_type]

    with get_conn() as conn:
        # Counts in last 1h and 24h
        counts = conn.execute(f"""
            SELECT
                COUNT(*)                                                   AS count_24h,
                SUM(CASE WHEN created_at >= datetime('now','-1 hour')
                         THEN 1 ELSE 0 END)                                AS count_1h,
                ROUND(MAX(amount), 2)                                      AS max_amount_24h,
                ROUND(AVG(fraud_probability), 4)                           AS avg_fraud_prob_24h,
                COUNT(DISTINCT customer_id)                                AS unique_customers,
                COUNT(DISTINCT device_id)                                  AS unique_devices,
                COUNT(DISTINCT ip_address)                                 AS unique_ips
            FROM predictions
            WHERE {col} = ?
              AND created_at >= datetime('now','-24 hours')
        """, (entity_value,)).fetchone()

    c = dict(counts) if counts else {}

    count_1h  = c.get("count_1h", 0) or 0
    count_24h = c.get("count_24h", 0) or 0
    avg_prob  = c.get("avg_fraud_prob_24h", 0.0) or 0.0

    # Simple velocity risk score
    velocity_risk = "LOW"
    signals: list[str] = []

    if count_1h >= 10:
        signals.append("HIGH_FREQUENCY_1H")
        velocity_risk = "HIGH"
    elif count_1h >= 5:
        signals.append("ELEVATED_FREQUENCY_1H")
        velocity_risk = "MEDIUM"

    if count_24h >= 50:
        signals.append("HIGH_VOLUME_24H")
        velocity_risk = "HIGH"

    if avg_prob >= 0.7:
        signals.append("HIGH_FRAUD_PROBABILITY_PATTERN")
        velocity_risk = "HIGH"

    unique_customers = c.get("unique_customers", 0) or 0
    unique_devices   = c.get("unique_devices", 0) or 0

    if entity_type == "ip" and unique_customers >= 5:
        signals.append("SHARED_IP_MULTIPLE_CUSTOMERS")
        velocity_risk = "HIGH"

    if entity_type == "device" and unique_customers >= 3:
        signals.append("SHARED_DEVICE_MULTIPLE_CUSTOMERS")
        velocity_risk = "HIGH"

    return {
        "entity_type":         entity_type,
        "entity_value":        entity_value,
        "count_last_1h":       count_1h,
        "count_last_24h":      count_24h,
        "max_amount_24h":      c.get("max_amount_24h") or 0.0,
        "avg_fraud_prob_24h":  avg_prob,
        "unique_customers":    unique_customers,
        "unique_devices":      unique_devices,
        "unique_ips":          c.get("unique_ips", 0) or 0,
        "velocity_risk":       velocity_risk,
        "signals":             signals,
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  ANALYTICS — Business Impact / Fraud Cost
# ══════════════════════════════════════════════════════════════════════════ #

def get_business_impact() -> dict:
    """
    Financial impact summary for GET /business/impact.
    Uses the predictions table for today and all-time case data.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with get_conn() as conn:
        pred_row = conn.execute("""
            SELECT
                ROUND(SUM(potential_loss), 2)      AS total_potential_loss,
                ROUND(SUM(blocked_loss), 2)        AS blocked_loss_today,
                ROUND(SUM(false_positive_cost), 2) AS estimated_false_positive_cost
            FROM predictions
            WHERE DATE(created_at) = ?
        """, (today,)).fetchone()

        fraud_row = conn.execute("""
            SELECT ROUND(SUM(p.amount), 2) AS confirmed_fraud_loss
            FROM predictions p
            INNER JOIN cases c ON p.case_id = c.id
            WHERE c.status = 'CONFIRMED_FRAUD'
        """).fetchone()

    p = dict(pred_row) if pred_row else {}
    f = dict(fraud_row) if fraud_row else {}

    return {
        "total_potential_loss":           p.get("total_potential_loss") or 0.0,
        "blocked_loss_today":             p.get("blocked_loss_today") or 0.0,
        "estimated_false_positive_cost":  p.get("estimated_false_positive_cost") or 0.0,
        "confirmed_fraud_loss":           f.get("confirmed_fraud_loss") or 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════ #
#  ANALYTICS — Attack / Fraud Pattern Detection
# ══════════════════════════════════════════════════════════════════════════ #

def get_attack_signals() -> dict:
    """
    Detect coordinated fraud patterns from the predictions table.
    Used by GET /attack/detection.

    Patterns detected:
    - Card testing: many LOW-amount HIGH-risk txns from same device
    - Fraud spike: sudden increase in HIGH-risk predictions this hour vs avg
    - IP abuse: single IP used by many customers
    - Device abuse: single device used by many customers
    """
    with get_conn() as conn:
        # Fraud spike: compare last 1h HIGH-risk count to prior 23h hourly avg
        spike_row = conn.execute("""
            SELECT
                SUM(CASE WHEN created_at >= datetime('now','-1 hour')
                         AND risk_level='HIGH' THEN 1 ELSE 0 END)     AS high_last_1h,
                SUM(CASE WHEN risk_level='HIGH' THEN 1 ELSE 0 END)    AS high_last_24h
            FROM predictions
            WHERE created_at >= datetime('now','-24 hours')
        """).fetchone()

        # Card testing: device with many low-amount HIGH-risk txns in 1h
        card_test_row = conn.execute("""
            SELECT device_id, COUNT(*) AS cnt
            FROM predictions
            WHERE amount < 10
              AND risk_level = 'HIGH'
              AND created_at >= datetime('now','-1 hour')
              AND device_id IS NOT NULL
            GROUP BY device_id
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()

        # IP abuse: IP used by 5+ distinct customers in 24h
        ip_abuse_row = conn.execute("""
            SELECT ip_address, COUNT(DISTINCT customer_id) AS unique_customers
            FROM predictions
            WHERE created_at >= datetime('now','-24 hours')
              AND ip_address IS NOT NULL
            GROUP BY ip_address
            HAVING unique_customers >= 5
            ORDER BY unique_customers DESC
            LIMIT 1
        """).fetchone()

        # Device abuse: device used by 3+ distinct customers in 24h
        dev_abuse_row = conn.execute("""
            SELECT device_id, COUNT(DISTINCT customer_id) AS unique_customers
            FROM predictions
            WHERE created_at >= datetime('now','-24 hours')
              AND device_id IS NOT NULL
            GROUP BY device_id
            HAVING unique_customers >= 3
            ORDER BY unique_customers DESC
            LIMIT 1
        """).fetchone()

    attacks: list[dict] = []
    attack_detected = False

    # ── Fraud spike check ─────────────────────────────────────────────────
    if spike_row:
        s = dict(spike_row)
        high_1h  = s.get("high_last_1h", 0) or 0
        high_24h = s.get("high_last_24h", 0) or 0
        hourly_avg = (high_24h - high_1h) / 23 if high_24h > high_1h else 0
        if high_1h > max(hourly_avg * 3, 5):   # 3x spike threshold
            attack_detected = True
            attacks.append({
                "attack_type":        "FRAUD_SPIKE",
                "severity":           "HIGH",
                "affected_entities":  [],
                "detail":             f"{high_1h} high-risk txns in last 1h vs avg {hourly_avg:.1f}/h",
                "recommendation":     "Review high-risk transactions immediately and consider temporary block rules.",
            })

    # ── Card testing check ────────────────────────────────────────────────
    if card_test_row:
        ct = dict(card_test_row)
        if (ct.get("cnt") or 0) >= 5:
            attack_detected = True
            attacks.append({
                "attack_type":       "CARD_TESTING",
                "severity":          "CRITICAL",
                "affected_entities": [ct.get("device_id")],
                "detail":            f"Device {ct.get('device_id')} made {ct.get('cnt')} low-amount high-risk txns in 1h.",
                "recommendation":    "Block device and review linked accounts immediately.",
            })

    # ── IP abuse check ────────────────────────────────────────────────────
    if ip_abuse_row:
        ia = dict(ip_abuse_row)
        attack_detected = True
        attacks.append({
            "attack_type":       "IP_ABUSE",
            "severity":          "HIGH",
            "affected_entities": [ia.get("ip_address")],
            "detail":            f"IP {ia.get('ip_address')} linked to {ia.get('unique_customers')} customers in 24h.",
            "recommendation":    "Block IP and audit associated accounts.",
        })

    # ── Device abuse check ────────────────────────────────────────────────
    if dev_abuse_row:
        da = dict(dev_abuse_row)
        attack_detected = True
        attacks.append({
            "attack_type":       "DEVICE_ABUSE",
            "severity":          "HIGH",
            "affected_entities": [da.get("device_id")],
            "detail":            f"Device {da.get('device_id')} linked to {da.get('unique_customers')} customers in 24h.",
            "recommendation":    "Flag device for manual review.",
        })

    overall_severity = "NONE"
    if attacks:
        if any(a["severity"] == "CRITICAL" for a in attacks):
            overall_severity = "CRITICAL"
        else:
            overall_severity = "HIGH"

    return {
        "attack_detected":  attack_detected,
        "overall_severity": overall_severity,
        "attacks":          attacks,
        "checked_at":       _now(),
    }
