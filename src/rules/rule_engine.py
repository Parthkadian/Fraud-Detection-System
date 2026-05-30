"""
Business Rule Engine
====================
Evaluates a set of configurable, YAML-defined business rules
against feature-engineered transaction data.

Rules are loaded from ``configs/business_rules.yaml`` and evaluated
**after** the ML model produces its score.  A rule can either:

- ``flag``          — keep ML prediction but mark as rule-triggered
- ``override_high`` — force prediction=1, risk_level=HIGH regardless of ML

This replicates how real financial fraud systems work: the ML model
provides a probabilistic score, but compliance rules provide hard
guardrails that the model cannot override.

Usage
-----
>>> from src.rules.rule_engine import BusinessRuleEngine
>>> engine = BusinessRuleEngine()
>>> result = engine.evaluate(features, fraud_probability=0.03)
>>> # {'triggered': True, 'rule_name': 'Large Amount Threshold', ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config_loader import load_yaml_file

logger = logging.getLogger("fraud_detection_logger")

_RULES_CONFIG = "configs/business_rules.yaml"


@dataclass
class RuleResult:
    """Result of evaluating all business rules against one transaction."""
    triggered: bool = False
    rule_name: Optional[str] = None
    severity: Optional[str] = None
    action: Optional[str] = None
    description: Optional[str] = None
    all_triggered: list[str] = field(default_factory=list)


class BusinessRuleEngine:
    """
    Evaluates YAML-configured business rules against engineered features.

    Parameters
    ----------
    rules_path : str
        Path to the business_rules.yaml config file.

    Examples
    --------
    >>> engine = BusinessRuleEngine()
    >>> result = engine.evaluate({"Amount": 9999, ...}, fraud_probability=0.02)
    >>> result.triggered       # True
    >>> result.rule_name       # "Large Amount Threshold"
    >>> result.action          # "override_high"
    """

    def __init__(self, rules_path: str = _RULES_CONFIG) -> None:
        self.rules_path = rules_path
        self._rules: list[dict] = []
        self._load_rules()

    def _load_rules(self) -> None:
        """Load and validate rules from YAML config."""
        try:
            config = load_yaml_file(self.rules_path)
            self._rules = [r for r in config.get("rules", []) if r.get("active", True)]
            logger.info(f"BusinessRuleEngine loaded {len(self._rules)} active rule(s).")
        except Exception as exc:
            logger.error(f"Failed to load business rules: {exc}. Running with no rules.")
            self._rules = []

    def get_rules(self) -> list[dict]:
        """Return the list of active rules (for API exposure)."""
        return self._rules

    def evaluate(
        self,
        features: dict,
        fraud_probability: float = 0.0,
    ) -> RuleResult:
        """
        Evaluate all active rules against a single transaction's features.

        Parameters
        ----------
        features : dict
            Feature dict — may include both raw inputs and engineered
            features (e.g. ``is_night_transaction``, ``v_features_magnitude``).
        fraud_probability : float
            ML model output (used in future rule conditions).

        Returns
        -------
        RuleResult
            The first CRITICAL/HIGH rule that fires, or the first
            rule of any severity, or an un-triggered result.
        """
        triggered_rules: list[dict] = []

        for rule in self._rules:
            if self._evaluate_single_rule(rule, features):
                triggered_rules.append(rule)

        if not triggered_rules:
            return RuleResult(triggered=False)

        # Prioritise by severity: CRITICAL > HIGH > MEDIUM > LOW
        _severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        triggered_rules.sort(
            key=lambda r: _severity_order.get(r.get("severity", "LOW"), 99)
        )

        top_rule = triggered_rules[0]
        all_names = [r["name"] for r in triggered_rules]

        logger.warning(
            f"🔴 Business rule triggered: [{top_rule['severity']}] "
            f"{top_rule['name']} | Amount={features.get('Amount', '?')}"
        )

        return RuleResult(
            triggered=True,
            rule_name=top_rule["name"],
            severity=top_rule.get("severity", "MEDIUM"),
            action=top_rule.get("action", "flag"),
            description=top_rule.get("description", ""),
            all_triggered=all_names,
        )

    # ------------------------------------------------------------------ #
    #  Internal evaluation logic
    # ------------------------------------------------------------------ #
    def _evaluate_single_rule(self, rule: dict, features: dict) -> bool:
        """Return True if the rule condition is satisfied."""
        try:
            field_val = features.get(rule["field"])
            if field_val is None:
                return False

            primary_match = self._compare(
                float(field_val),
                rule["operator"],
                float(rule["value"]),
            )

            # Optional secondary condition (AND logic)
            if not primary_match:
                return False

            if "secondary_field" in rule:
                sec_val = features.get(rule["secondary_field"])
                if sec_val is None:
                    return False
                secondary_match = self._compare(
                    float(sec_val),
                    rule["secondary_operator"],
                    float(rule["secondary_value"]),
                )
                return secondary_match

            return True

        except (KeyError, TypeError, ValueError) as exc:
            logger.debug(f"Rule evaluation skipped ({rule.get('name', '?')}): {exc}")
            return False

    @staticmethod
    def _compare(value: float, operator: str, threshold: float) -> bool:
        """Evaluate a single comparison."""
        ops = {
            ">":  value > threshold,
            ">=": value >= threshold,
            "<":  value < threshold,
            "<=": value <= threshold,
            "==": value == threshold,
            "!=": value != threshold,
        }
        return ops.get(operator, False)
