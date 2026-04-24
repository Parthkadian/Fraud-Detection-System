"""
Model Factory
=============
Centralised model instantiation from YAML configuration.
Supports XGBoost, Logistic Regression, and Random Forest.
"""

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.utils.config_loader import load_yaml_file


class ModelFactory:
    """Create configured scikit-learn / XGBoost estimators."""

    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")
        self.model_params = load_yaml_file("configs/model_params.yaml")

    def get_model(self, model_name: str, scale_pos_weight: float = 1.0):
        """
        Return a fresh model instance.

        Parameters
        ----------
        model_name : str
            One of ``"xgboost"``, ``"logistic_regression"``, ``"random_forest"``.
        scale_pos_weight : float
            Positive-class weight for XGBoost (ignored by other models).

        Returns
        -------
        estimator
            Unfitted scikit-learn compatible estimator.
        """
        if model_name == "xgboost":
            params = self.model_params["xgboost"].copy()
            params["scale_pos_weight"] = scale_pos_weight
            return XGBClassifier(**params)

        if model_name == "logistic_regression":
            params = self.model_params["logistic_regression"]
            return LogisticRegression(**params)

        if model_name == "random_forest":
            params = self.model_params["random_forest"]
            return RandomForestClassifier(**params)

        raise ValueError(f"Unsupported model: {model_name}")