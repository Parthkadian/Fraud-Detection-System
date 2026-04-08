import pandas as pd
import shap

from src.utils.config_loader import load_yaml_file
from src.utils.common import load_object
from src.features.feature_engineering import FeatureEngineering


class ShapExplainer:
    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")

        self.model = load_object(
            f"{self.config['paths']['model_dir']}/fraud_model.pkl"
        )

        self.fe = FeatureEngineering()

        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
        else:
            self.feature_order = self.model.get_booster().feature_names

        self.explainer = shap.TreeExplainer(self.model)

    def explain_single(self, input_data: dict, top_n: int = 10) -> dict:
        """
        Generate SHAP explanation for a single transaction.
        """
        df = pd.DataFrame([input_data])
        df = self.fe.transform(df)

        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_order]

        shap_values = self.explainer.shap_values(df)

        # binary classification handling
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        feature_contributions = pd.DataFrame({
            "feature": self.feature_order,
            "shap_value": values
        })

        feature_contributions["abs_shap"] = feature_contributions["shap_value"].abs()
        feature_contributions = feature_contributions.sort_values(
            by="abs_shap",
            ascending=False
        ).head(top_n)

        return {
            "top_features": feature_contributions[["feature", "shap_value"]].to_dict(orient="records")
        }