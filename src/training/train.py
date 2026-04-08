import pandas as pd

from src.utils.config_loader import load_yaml_file
from src.utils.common import save_object, save_json
from src.monitoring.logger import setup_logger
from src.training.model_factory import ModelFactory
from src.training.thresholding import ThresholdOptimizer
from src.evaluation.evaluate import ModelEvaluator


class FraudModelTrainer:
    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")
        self.logger = setup_logger(self.config["paths"]["log_file"])
        self.model_factory = ModelFactory()
        self.evaluator = ModelEvaluator()

    def _calculate_scale_pos_weight(self, y_train) -> float:
        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()

        if positive_count == 0:
            return 1.0

        return negative_count / positive_count

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series):
        """
        Train fraud detection model, tune threshold, evaluate on validation set,
        and save artifacts.
        """
        model_name = self.config["training"]["model_name"]
        threshold_metric = self.config["training"]["threshold_metric"]

        self.logger.info(f"Selected model: {model_name}")

        scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        self.logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

        model = self.model_factory.get_model(
            model_name=model_name,
            scale_pos_weight=scale_pos_weight
        )

        self.logger.info("Training model...")
        model.fit(X_train, y_train)

        self.logger.info("Generating validation probabilities...")
        y_val_prob = model.predict_proba(X_val)[:, 1]

        threshold_optimizer = ThresholdOptimizer(metric=threshold_metric)
        best_threshold, best_score = threshold_optimizer.find_best_threshold(y_val, y_val_prob)

        self.logger.info(f"Best threshold based on {threshold_metric}: {best_threshold}")
        self.logger.info(f"Best validation {threshold_metric}: {best_score:.6f}")

        y_val_pred = (y_val_prob >= best_threshold).astype(int)

        metrics = self.evaluator.evaluate(y_val, y_val_pred, y_val_prob)
        metrics["best_threshold"] = best_threshold
        metrics["threshold_metric"] = threshold_metric
        metrics["threshold_metric_score"] = round(float(best_score), 6)
        metrics["model_name"] = model_name

        model_path = f"{self.config['paths']['model_dir']}/fraud_model.pkl"
        metrics_path = self.config["paths"]["metrics_file"]
        threshold_path = self.config["paths"]["threshold_file"]

        save_object(model_path, model)
        save_json(threshold_path, {"best_threshold": best_threshold})
        self.evaluator.save_metrics(metrics, metrics_path)

        self.logger.info(f"Model saved to: {model_path}")
        self.logger.info(f"Threshold saved to: {threshold_path}")
        self.logger.info(f"Metrics saved to: {metrics_path}")

        return model, metrics