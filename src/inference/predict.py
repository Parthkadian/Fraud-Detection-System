"""
Fraud Predictor
===============
Production-grade inference engine for single-transaction fraud scoring.

Integrations
------------
- **FeatureEngineering** — applies the same transformations used in training
- **BusinessRuleEngine** — evaluates YAML-defined compliance rules after ML
- **PredictionAuditLogger** — logs every call to SQLite (PCI-DSS compliance)
- **FraudMetrics** — records Prometheus counters and latency histograms
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import pandas as pd

from src.utils.config_loader import load_yaml_file
from src.utils.common import load_object, load_json
from src.features.feature_engineering import FeatureEngineering
from src.rules.rule_engine import BusinessRuleEngine
from src.monitoring.audit_log import PredictionAuditLogger
from src.monitoring.prometheus_metrics import get_metrics

logger = logging.getLogger("fraud_detection_logger")


class FraudPredictor:
    """
    End-to-end inference pipeline for a single credit-card transaction.

    Parameters loaded from ``configs/config.yaml``:
    - model path (``paths.model_dir``)
    - threshold path (``paths.threshold_file``)

    Post-ML processing
    ------------------
    1. Business rules evaluated — may override risk level
    2. Result logged to SQLite audit trail
    3. Prometheus metrics updated
    """

    def __init__(self) -> None:
        self.config = load_yaml_file("configs/config.yaml")
        self.version = self.config.get("project", {}).get("version", "2.0.0")

        # ── Model ────────────────────────────────────────────────────────
        self.model = load_object(
            f"{self.config['paths']['model_dir']}/fraud_model.pkl"
        )
        threshold_data = load_json(self.config["paths"]["threshold_file"])
        self.threshold = threshold_data["best_threshold"]

        # ── Feature engineering ──────────────────────────────────────────
        self.fe = FeatureEngineering()
        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
        else:
            self.feature_order = self.model.get_booster().feature_names

        # ── Integrations ─────────────────────────────────────────────────
        self.rule_engine = BusinessRuleEngine()
        self.audit_logger = PredictionAuditLogger(model_version=self.version)
        self.metrics = get_metrics(model_version=self.version)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def predict(self, input_data: dict) -> dict:
        """
        Score a single transaction end-to-end.

        Parameters
        ----------
        input_data : dict
            Raw transaction features (Time, V1-V28, Amount).

        Returns
        -------
        dict
            Keys: fraud_probability, prediction, risk_level,
                  rule_triggered, latency_ms
        """
        t_start = time.perf_counter()

        # 1 ── Feature engineering
        df = pd.DataFrame([input_data])
        df = self.fe.transform(df, is_train=False)

        # Ensure all required columns exist
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_order]

        # 2 ── ML inference
        prob = float(self.model.predict_proba(df)[:, 1][0])
        prediction = int(prob >= self.threshold)

        # 3 ── Risk level from probability
        risk_level = self._risk_level(prob)

        # 4 ── Business rules (may override)
        engineered_features = df.iloc[0].to_dict()
        engineered_features["Amount"] = input_data.get("Amount", 0.0)
        rule_result = self.rule_engine.evaluate(engineered_features, prob)

        rule_triggered: Optional[str] = None
        if rule_result.triggered:
            rule_triggered = rule_result.rule_name
            if rule_result.action == "override_high":
                prediction = 1
                risk_level = "HIGH"
                logger.info(
                    f"Rule '{rule_result.rule_name}' overrode ML → prediction=1, risk=HIGH"
                )

        latency_ms = (time.perf_counter() - t_start) * 1000

        # 5 ── Audit log
        self.audit_logger.log(
            input_data=input_data,
            fraud_probability=prob,
            prediction=prediction,
            risk_level=risk_level,
            latency_ms=latency_ms,
            rule_triggered=rule_triggered,
        )

        # 6 ── Prometheus metrics
        self.metrics.record_prediction(
            risk_level=risk_level,
            prediction=prediction,
            fraud_probability=prob,
            latency_seconds=latency_ms / 1000,
            rule_name=rule_triggered,
        )

        return {
            "fraud_probability": round(prob, 4),
            "prediction": prediction,
            "risk_level": risk_level,
            "rule_triggered": rule_triggered,
            "latency_ms": round(latency_ms, 2),
        }

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob >= 0.8:
            return "HIGH"
        if prob >= 0.4:
            return "MEDIUM"
        return "LOW"