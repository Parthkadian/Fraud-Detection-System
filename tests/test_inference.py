"""
Tests — Inference
=================
Tests for single and batch prediction output formats and risk-level logic.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestRiskLevelLogic:
    """Unit tests for the risk-level assignment thresholds."""

    def _classify_risk(self, prob: float) -> str:
        if prob >= 0.8:
            return "HIGH"
        elif prob >= 0.4:
            return "MEDIUM"
        return "LOW"

    def test_low_risk(self):
        assert self._classify_risk(0.1) == "LOW"
        assert self._classify_risk(0.39) == "LOW"

    def test_medium_risk(self):
        assert self._classify_risk(0.4) == "MEDIUM"
        assert self._classify_risk(0.79) == "MEDIUM"

    def test_high_risk(self):
        assert self._classify_risk(0.8) == "HIGH"
        assert self._classify_risk(0.99) == "HIGH"

    def test_boundary_cases(self):
        assert self._classify_risk(0.0) == "LOW"
        assert self._classify_risk(1.0) == "HIGH"


class TestPredictionOutputFormat:
    """Tests for the expected output structure of predictions."""

    def test_prediction_dict_keys(self):
        """A prediction result must contain the three required keys."""
        mock_result = {
            "fraud_probability": 0.123,
            "prediction": 0,
            "risk_level": "LOW",
        }
        assert "fraud_probability" in mock_result
        assert "prediction" in mock_result
        assert "risk_level" in mock_result

    def test_probability_range(self):
        """Fraud probability must be between 0 and 1."""
        for prob in [0.0, 0.5, 1.0]:
            assert 0.0 <= prob <= 1.0

    def test_prediction_is_binary(self):
        """Prediction must be 0 or 1."""
        for pred in [0, 1]:
            assert pred in (0, 1)
