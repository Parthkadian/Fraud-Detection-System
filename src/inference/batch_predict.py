import pandas as pd

from src.utils.config_loader import load_yaml_file
from src.utils.common import load_object, load_json
from src.features.feature_engineering import FeatureEngineering


class BatchPredictor:
    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")

        self.model = load_object(
            f"{self.config['paths']['model_dir']}/fraud_model.pkl"
        )

        threshold_data = load_json(self.config["paths"]["threshold_file"])
        self.threshold = threshold_data["best_threshold"]

        self.fe = FeatureEngineering()

        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
        else:
            self.feature_order = self.model.get_booster().feature_names

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        input_df = df.copy()

        if "Class" in input_df.columns:
            input_df = input_df.drop(columns=["Class"])

        input_df = self.fe.transform(input_df, is_train=False)

        for col in self.feature_order:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[self.feature_order]

        probs = self.model.predict_proba(input_df)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        risk_levels = []
        for prob in probs:
            if prob >= 0.8:
                risk_levels.append("HIGH")
            elif prob >= 0.4:
                risk_levels.append("MEDIUM")
            else:
                risk_levels.append("LOW")

        result_df = df.copy()
        result_df["fraud_probability"] = probs.round(4)
        result_df["prediction"] = preds
        result_df["risk_level"] = risk_levels

        return result_df