"""
Model Evaluation Module
=======================
Comprehensive classification metrics for imbalanced fraud detection,
including Matthews Correlation Coefficient and Cohen's Kappa — both
considered gold-standard metrics for imbalanced binary classification.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
)


class ModelEvaluator:
    """
    Evaluates binary classification models with metrics tailored for
    highly-imbalanced fraud datasets.
    """

    def evaluate(
        self,
        y_true,
        y_pred,
        y_prob,
        class_names: Optional[list[str]] = None,
    ) -> dict:
        """
        Compute a comprehensive set of classification metrics.

        Parameters
        ----------
        y_true : array-like
            Ground-truth binary labels.
        y_pred : array-like
            Predicted binary labels (after thresholding).
        y_prob : array-like
            Predicted probabilities for the positive class.
        class_names : list[str], optional
            Human-readable class names for the classification report.

        Returns
        -------
        dict
            Dictionary of metric name → value.
        """
        if class_names is None:
            class_names = ["Not Fraud", "Fraud"]

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            # --- Standard metrics ---
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "precision": round(
                float(precision_score(y_true, y_pred, zero_division=0)), 6
            ),
            "recall": round(
                float(recall_score(y_true, y_pred, zero_division=0)), 6
            ),
            "f1_score": round(
                float(f1_score(y_true, y_pred, zero_division=0)), 6
            ),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
            "pr_auc": round(
                float(average_precision_score(y_true, y_prob)), 6
            ),
            # --- Advanced imbalanced-classification metrics ---
            "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6),
            "cohen_kappa": round(
                float(cohen_kappa_score(y_true, y_pred)), 6
            ),
            # --- Confusion matrix ---
            "confusion_matrix": cm.tolist(),
            # --- Per-class report (string) ---
            "classification_report": classification_report(
                y_true, y_pred, target_names=class_names, zero_division=0
            ),
        }
        return metrics

    def save_metrics(self, metrics: dict, file_path: str) -> None:
        """
        Persist metrics as JSON.  The ``classification_report`` string is
        included as-is for human readability.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=4)