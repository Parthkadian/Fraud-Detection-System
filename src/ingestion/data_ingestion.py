import pandas as pd
from src.utils.config_loader import load_yaml_file
from src.monitoring.logger import setup_logger


class DataIngestion:
    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")
        self.logger = setup_logger(self.config["paths"]["log_file"])

    def load_data(self) -> pd.DataFrame:
        """
        Load raw dataset from path and generate synthetic NLP features.
        """
        path = self.config["paths"]["raw_data"]
        self.logger.info(f"Loading data from: {path}")

        df = pd.read_csv(path)

        # Ensure 'Class' column exists before relying on it for synthetic generation
        if "Class" in df.columns:
            import numpy as np
            np.random.seed(42)
            
            # Legitimate transactions memos
            legit_memos = ["Amazon electronics", "Starbucks coffee", "Uber ride", "Grocery store", "Netflix subscription", "Gas station"]
            # Fraudulent transactions memos
            fraud_memos = ["Unrecognized overseas transfer", "Large cryptocurrency buy", "Luxury watch purchase", "Suspicious wire transfer", "High-value gift cards"]
            
            # Assign randomly based on class
            df["transaction_memo"] = np.where(
                df["Class"] == 1,
                np.random.choice(fraud_memos, size=len(df)),
                np.random.choice(legit_memos, size=len(df))
            )
        else:
            self.logger.warning("Class column not found, unable to generate realistic NLP features. Using defaults.")
            df["transaction_memo"] = "Standard transaction"

        self.logger.info(f"Data shape after reading and synthetic NLP feature injection: {df.shape}")
        return df