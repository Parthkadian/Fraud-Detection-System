"""
Prometheus Metrics
==================
Exposes real-time operational metrics for the Fraud Detection API
in Prometheus text format at the ``/metrics`` endpoint.

Metrics exposed
---------------
fraud_predictions_total : Counter
    Total number of predictions served (labelled by risk_level).
fraud_flagged_total : Counter
    Total number of transactions predicted as fraud.
rule_triggered_total : Counter
    Total number of predictions where a business rule fired.
prediction_latency_seconds : Histogram
    End-to-end prediction latency distribution.
fraud_probability_gauge : Gauge
    Most recent fraud probability (useful for alerting).
model_info : Info
    Static model metadata (version, algorithm).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("fraud_detection_logger")

# ── Try to import prometheus_client ──────────────────────────────────────────
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed. "
        "Install with: pip install prometheus-client==0.21.0"
    )


class FraudMetrics:
    """
    Thread-safe Prometheus metric collectors for the Fraud Detection API.

    All metrics use a shared global registry so they persist across
    requests in the same process.
    """

    def __init__(self, model_version: str = "2.0.0") -> None:
        self.available = _PROMETHEUS_AVAILABLE

        if not self.available:
            return

        # Guard against duplicate registration (e.g. hot-reload)
        try:
            self.predictions_total = Counter(
                "fraud_predictions_total",
                "Total predictions served, labelled by risk level",
                ["risk_level"],
            )
            self.flagged_total = Counter(
                "fraud_flagged_total",
                "Total transactions predicted as fraudulent",
            )
            self.rule_triggered_total = Counter(
                "fraud_rule_triggered_total",
                "Total predictions where a business rule overrode the model",
                ["rule_name"],
            )
            self.latency = Histogram(
                "fraud_prediction_latency_seconds",
                "End-to-end prediction latency in seconds",
                buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            )
            self.probability_gauge = Gauge(
                "fraud_last_probability",
                "Fraud probability of the most recent prediction",
            )
            self.fraud_rate_gauge = Gauge(
                "fraud_rate_rolling",
                "Rolling fraud rate over last 100 predictions (approx)",
            )
            self.model_info = Info(
                "fraud_model",
                "Static model metadata",
            )
            self.model_info.info({
                "version": model_version,
                "algorithm": "XGBoost",
                "calibration": "isotonic",
                "threshold_strategy": "f1_optimised",
            })
        except ValueError:
            # Metrics already registered (happens with uvicorn --reload)
            pass

    def record_prediction(
        self,
        risk_level: str,
        prediction: int,
        fraud_probability: float,
        latency_seconds: float,
        rule_name: Optional[str] = None,
    ) -> None:
        """Update all metrics for one prediction event."""
        if not self.available:
            return
        try:
            self.predictions_total.labels(risk_level=risk_level).inc()
            if prediction == 1:
                self.flagged_total.inc()
            if rule_name:
                self.rule_triggered_total.labels(rule_name=rule_name).inc()
            self.latency.observe(latency_seconds)
            self.probability_gauge.set(fraud_probability)
        except Exception as exc:
            logger.debug(f"Metrics record error: {exc}")

    def get_metrics_output(self) -> tuple[str, str]:
        """
        Generate Prometheus text-format metrics.

        Returns
        -------
        tuple[str, str]
            (metrics_text, content_type)
        """
        if not self.available:
            return "# prometheus_client not installed\n", "text/plain"
        try:
            output = generate_latest(REGISTRY)
            return output.decode("utf-8"), CONTENT_TYPE_LATEST
        except Exception as exc:
            logger.error(f"Metrics generation failed: {exc}")
            return f"# Error: {exc}\n", "text/plain"


# Module-level singleton
_metrics_instance: Optional[FraudMetrics] = None


def get_metrics(model_version: str = "2.0.0") -> FraudMetrics:
    """Return (or create) the module-level FraudMetrics singleton."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = FraudMetrics(model_version=model_version)
    return _metrics_instance
