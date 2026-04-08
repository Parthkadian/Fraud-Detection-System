import pandas as pd

from src.utils.config_loader import load_yaml_file
from src.utils.common import load_object, load_json
from src.features.feature_engineering import FeatureEngineering


class FraudPredictor:
    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")

        self.model = load_object(
            f"{self.config['paths']['model_dir']}/fraud_model.pkl"
        )

        threshold_data = load_json(self.config["paths"]["threshold_file"])
        self.threshold = threshold_data["best_threshold"]

        self.fe = FeatureEngineering()

        # Get feature order used during training
        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
        else:
            self.feature_order = self.model.get_booster().feature_names

    def predict(self, input_data: dict) -> dict:
        """
        Predict fraud for a single transaction.
        """
        df = pd.DataFrame([input_data])

        # Apply same feature engineering as training
        df = self.fe.transform(df)

        # Ensure all required columns exist
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training order
        df = df[self.feature_order]

        prob = self.model.predict_proba(df)[:, 1][0]
        prediction = int(prob >= self.threshold)

        if prob >= 0.8:
            risk = "HIGH"
        elif prob >= 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "fraud_probability": round(float(prob), 4),
            "prediction": prediction,
            "risk_level": risk
        }