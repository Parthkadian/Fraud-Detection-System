from src.ingestion.data_ingestion import DataIngestion
from src.preprocessing.data_cleaning import DataCleaning
from src.preprocessing.splitter import DataSplitter
from src.features.feature_engineering import FeatureEngineering
from src.training.train import FraudModelTrainer
from src.utils.common import create_directories
from src.monitoring.logger import setup_logger
from src.utils.config_loader import load_yaml_file


def run_pipeline():
    config = load_yaml_file("configs/config.yaml")
    logger = setup_logger(config["paths"]["log_file"])

    logger.info("Starting pipeline...")

    create_directories([
        "data/processed",
        "models/trained",
        "models/artifacts",
        "logs",
        "reports"
    ])

    ingestion = DataIngestion()
    df = ingestion.load_data()

    cleaner = DataCleaning()
    df = cleaner.clean(df)

    fe = FeatureEngineering()
    df = fe.transform(df)

    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(df)

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Validation shape: {X_val.shape}")
    logger.info(f"Test shape: {X_test.shape}")

    trainer = FraudModelTrainer()
    model, metrics = trainer.train(X_train, y_train, X_val, y_val)

    logger.info("Training pipeline completed successfully.")
    logger.info(f"Validation Metrics: {metrics}")


if __name__ == "__main__":
    run_pipeline()