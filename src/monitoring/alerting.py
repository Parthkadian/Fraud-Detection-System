"""
Alerting Module
===============
Threshold-based alerting system for batch prediction results.
Detects anomalous fraud rates and high-risk concentrations,
logs alerts, and maintains an in-memory alert history.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger("fraud_detection_logger")


class AlertRule:
    """A single configurable alert rule."""

    def __init__(self, name: str, metric: str, threshold: float, direction: str = "above"):
        self.name = name
        self.metric = metric
        self.threshold = threshold
        self.direction = direction  # "above" or "below"

    def evaluate(self, value: float) -> bool:
        if self.direction == "above":
            return value > self.threshold
        return value < self.threshold


class FraudAlertManager:
    """
    Evaluates batch prediction results against configurable alert rules
    and maintains an alert history for dashboard display.
    """

    def __init__(
        self,
        fraud_rate_threshold: float = 0.05,
        high_risk_threshold: float = 0.10,
    ):
        self.rules = [
            AlertRule(
                "High Fraud Rate",
                "fraud_rate",
                fraud_rate_threshold,
                "above",
            ),
            AlertRule(
                "Excessive High-Risk Transactions",
                "high_risk_rate",
                high_risk_threshold,
                "above",
            ),
        ]
        self.alert_history: list[dict] = []

    def evaluate_batch(self, result_df: pd.DataFrame) -> list[dict]:
        """
        Evaluate a scored batch DataFrame and return triggered alerts.

        Parameters
        ----------
        result_df : pd.DataFrame
            Must contain columns ``prediction`` and ``risk_level``.

        Returns
        -------
        list[dict]
            Triggered alerts with metadata.
        """
        total = len(result_df)
        if total == 0:
            return []

        fraud_count = int(result_df["prediction"].sum())
        high_risk_count = int((result_df["risk_level"] == "HIGH").sum())

        metric_values = {
            "fraud_rate": fraud_count / total,
            "high_risk_rate": high_risk_count / total,
        }

        triggered: list[dict] = []
        for rule in self.rules:
            value = metric_values.get(rule.metric, 0.0)
            if rule.evaluate(value):
                alert = {
                    "alert_name": rule.name,
                    "metric": rule.metric,
                    "value": round(value, 4),
                    "threshold": rule.threshold,
                    "severity": "CRITICAL" if value > rule.threshold * 2 else "WARNING",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "batch_size": total,
                }
                triggered.append(alert)
                logger.warning(
                    f"🚨 ALERT [{alert['severity']}] {rule.name}: "
                    f"{value:.2%} (threshold: {rule.threshold:.2%})"
                )

        self.alert_history.extend(triggered)
        return triggered

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent alerts."""
        return self.alert_history[-limit:]
