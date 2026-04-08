import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


class ModelEvaluator:
    def evaluate(self, y_true, y_pred, y_prob) -> dict:
        """
        Compute classification metrics for fraud detection.
        """
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
            "pr_auc": round(float(average_precision_score(y_true, y_prob)), 6),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        return metrics

    def save_metrics(self, metrics: dict, file_path: str) -> None:
        """
        Save metrics to JSON file.
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)