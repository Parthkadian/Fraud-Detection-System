import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class ThresholdOptimizer:
    def __init__(self, metric: str = "f1"):
        self.metric = metric

    def _score(self, y_true, y_pred):
        if self.metric == "f1":
            return f1_score(y_true, y_pred, zero_division=0)
        if self.metric == "precision":
            return precision_score(y_true, y_pred, zero_division=0)
        if self.metric == "recall":
            return recall_score(y_true, y_pred, zero_division=0)
        raise ValueError(f"Unsupported threshold metric: {self.metric}")

    def find_best_threshold(self, y_true, y_prob):
        """
        Search best threshold between 0.1 and 0.95
        """
        best_threshold = 0.5
        best_score = -1.0

        for threshold in np.arange(0.10, 0.96, 0.01):
            y_pred = (y_prob >= threshold).astype(int)
            score = self._score(y_true, y_pred)

            if score > best_score:
                best_score = score
                best_threshold = round(float(threshold), 2)

        return best_threshold, best_score