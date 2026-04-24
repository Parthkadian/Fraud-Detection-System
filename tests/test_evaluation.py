"""
Tests — Evaluation
==================
Tests for the ModelEvaluator class.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from src.evaluation.evaluate import ModelEvaluator


class TestModelEvaluator:
    """Tests for the ModelEvaluator class."""

    def test_returns_expected_keys(self, sample_predictions):
        y_true, y_pred, y_prob = sample_predictions
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_prob)

        expected_keys = [
            "accuracy", "precision", "recall", "f1_score",
            "roc_auc", "pr_auc", "mcc", "cohen_kappa",
            "confusion_matrix", "classification_report",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["mcc"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 0.0
        assert metrics["mcc"] == -1.0

    def test_metrics_are_serialisable(self, sample_predictions):
        y_true, y_pred, y_prob = sample_predictions
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_prob)

        # Must not raise
        json_str = json.dumps(metrics, indent=2)
        assert len(json_str) > 0

    def test_save_metrics(self, sample_predictions, tmp_path):
        y_true, y_pred, y_prob = sample_predictions
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_prob)

        file_path = str(tmp_path / "metrics.json")
        evaluator.save_metrics(metrics, file_path)

        assert Path(file_path).exists()
        loaded = json.loads(Path(file_path).read_text())
        assert loaded["accuracy"] == metrics["accuracy"]

    def test_classification_report_string(self, sample_predictions):
        y_true, y_pred, y_prob = sample_predictions
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_prob)

        report = metrics["classification_report"]
        assert "Not Fraud" in report
        assert "Fraud" in report
