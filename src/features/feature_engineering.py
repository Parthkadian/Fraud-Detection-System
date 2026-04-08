import pandas as pd
import numpy as np


class FeatureEngineering:
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional fraud-relevant features.
        """
        X = X.copy()

        # Log transform for Amount
        if "Amount" in X.columns:
            X["log_amount"] = np.log1p(X["Amount"])

        # Time-based feature
        if "Time" in X.columns:
            X["hour"] = (X["Time"] // 3600) % 24

        return X