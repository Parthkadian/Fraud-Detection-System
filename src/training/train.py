"""
Fraud Model Trainer
===================
Orchestrates the full training loop including:

- Dynamic class-weight computation for imbalanced datasets
- Stratified K-Fold cross-validation alongside holdout evaluation
- XGBoost early stopping to prevent overfitting
- Threshold optimisation on the validation set
- Full metric logging to MLflow
- Artefact persistence (model, threshold, metrics)
"""

import time
import logging

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold

from src.utils.config_loader import load_yaml_file
from src.utils.common import save_object, save_json
from src.monitoring.logger import setup_logger
from src.training.model_factory import ModelFactory
from src.training.thresholding import ThresholdOptimizer
from src.evaluation.evaluate import ModelEvaluator

logger = logging.getLogger("fraud_detection_logger")


class FraudModelTrainer:
    """Train, validate, and persist fraud detection models."""

    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")
        self.logger = setup_logger(self.config["paths"]["log_file"])
        self.model_factory = ModelFactory()
        self.evaluator = ModelEvaluator()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calculate_scale_pos_weight(y_train) -> float:
        """Compute the ratio of negatives to positives for XGBoost."""
        negative = int((y_train == 0).sum())
        positive = int((y_train == 1).sum())
        if positive == 0:
            return 1.0
        return negative / positive

    # ------------------------------------------------------------------ #
    #  Cross-validation
    # ------------------------------------------------------------------ #
    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series, model_name: str, n_splits: int = 5
    ) -> dict:
        """
        Run Stratified K-Fold cross-validation and return per-fold metrics.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics: list[dict] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            spw = self._calculate_scale_pos_weight(y_fold_train)
            model = self.model_factory.get_model(model_name, scale_pos_weight=spw)
            model.fit(X_fold_train, y_fold_train)

            y_prob = model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = self.evaluator.evaluate(y_fold_val, y_pred, y_prob)
            fold_metrics.append(metrics)
            self.logger.info(
                f"  Fold {fold}/{n_splits} — "
                f"F1: {metrics['f1_score']:.4f}  "
                f"ROC-AUC: {metrics['roc_auc']:.4f}  "
                f"MCC: {metrics['mcc']:.4f}"
            )

        # Aggregate
        aggregated = {}
        numeric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "mcc", "cohen_kappa"]
        for key in numeric_keys:
            values = [m[key] for m in fold_metrics]
            aggregated[f"cv_{key}_mean"] = round(float(np.mean(values)), 6)
            aggregated[f"cv_{key}_std"] = round(float(np.std(values)), 6)

        return aggregated

    # ------------------------------------------------------------------ #
    #  Main training entry point
    # ------------------------------------------------------------------ #
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """
        Train the fraud detection model end-to-end.

        Steps
        -----
        1. Run Stratified K-Fold cross-validation (logged to MLflow).
        2. Train final model on full training set with early stopping.
        3. Optimise decision threshold on validation set.
        4. Evaluate and persist all artefacts.

        Returns
        -------
        tuple[model, dict]
            Trained model and evaluation metrics dictionary.
        """
        model_name = self.config["training"]["model_name"]
        threshold_metric = self.config["training"]["threshold_metric"]

        mlflow.set_experiment(self.config["project"]["name"])

        with mlflow.start_run(run_name=f"train_{model_name}"):
            start = time.time()

            self.logger.info(f"Selected model: {model_name}")
            mlflow.log_param("model_name", model_name)

            # --- Class balance ---
            scale_pos_weight = self._calculate_scale_pos_weight(y_train)
            self.logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
            mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))

            # --- Cross-validation ---
            self.logger.info("Running 5-fold Stratified Cross-Validation …")
            cv_metrics = self._cross_validate(X_train, y_train, model_name)
            for k, v in cv_metrics.items():
                mlflow.log_metric(k, v)
            self.logger.info(
                f"CV Results — F1: {cv_metrics['cv_f1_score_mean']:.4f} "
                f"(±{cv_metrics['cv_f1_score_std']:.4f})"
            )

            # --- Final model training ---
            model = self.model_factory.get_model(
                model_name=model_name, scale_pos_weight=scale_pos_weight
            )

            self.logger.info("Training final model …")
            if model_name == "xgboost":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            else:
                model.fit(X_train, y_train)

            # --- Threshold optimisation ---
            self.logger.info("Generating validation probabilities …")
            y_val_prob = model.predict_proba(X_val)[:, 1]

            threshold_optimizer = ThresholdOptimizer(metric=threshold_metric)
            best_threshold, best_score = threshold_optimizer.find_best_threshold(
                y_val, y_val_prob
            )

            self.logger.info(
                f"Best threshold ({threshold_metric}): {best_threshold} "
                f"-> score: {best_score:.6f}"
            )

            # --- Evaluation ---
            y_val_pred = (y_val_prob >= best_threshold).astype(int)
            metrics = self.evaluator.evaluate(y_val, y_val_pred, y_val_prob)
            metrics["best_threshold"] = best_threshold
            metrics["threshold_metric"] = threshold_metric
            metrics["threshold_metric_score"] = round(float(best_score), 6)
            metrics["model_name"] = model_name
            metrics["training_time_seconds"] = round(time.time() - start, 2)

            # Merge CV metrics
            metrics.update(cv_metrics)

            # Log numeric metrics to MLflow
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

            # --- Persist artefacts ---
            model_path = f"{self.config['paths']['model_dir']}/fraud_model.pkl"
            metrics_path = self.config["paths"]["metrics_file"]
            threshold_path = self.config["paths"]["threshold_file"]

            save_object(model_path, model)
            save_json(threshold_path, {"best_threshold": best_threshold})
            self.evaluator.save_metrics(metrics, metrics_path)

            # Log model to MLflow
            if model_name == "xgboost":
                mlflow.xgboost.log_model(model, "fraud_model")
            else:
                mlflow.sklearn.log_model(model, "fraud_model")

            self.logger.info(f"Model saved to: {model_path}")
            self.logger.info(f"Threshold saved to: {threshold_path}")
            self.logger.info(f"Metrics saved to: {metrics_path}")
            self.logger.info(
                f"Training completed in {metrics['training_time_seconds']}s"
            )

            return model, metrics