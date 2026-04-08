import pandas as pd
from src.utils.config_loader import load_yaml_file
from src.monitoring.logger import setup_logger


class DataIngestion:
    def __init__(self):
        self.config = load_yaml_file("configs/config.yaml")
        self.logger = setup_logger(self.config["paths"]["log_file"])

    def load_data(self) -> pd.DataFrame:
        """
        Load raw dataset from path.
        """
        path = self.config["paths"]["raw_data"]
        self.logger.info(f"Loading data from: {path}")

        df = pd.read_csv(path)

        self.logger.info(f"Data shape: {df.shape}")
        return df