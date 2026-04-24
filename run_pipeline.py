"""
Pipeline Runner
================
Orchestrates the end-to-end fraud detection pipeline from data ingestion
through training and final test-set evaluation.

Usage
-----
    python run_pipeline.py                 # default settings
    python run_pipeline.py --model xgboost # specify model
    python run_pipeline.py --skip-cv       # skip cross-validation (faster)
"""

import argparse
import time
import logging

from src.ingestion.data_ingestion import DataIngestion
from src.preprocessing.data_cleaning import DataCleaning
from src.preprocessing.data_validator import DataValidator
from src.preprocessing.splitter import DataSplitter
from src.features.feature_engineering import FeatureEngineering
from src.training.train import FraudModelTrainer
from src.evaluation.evaluate import ModelEvaluator
from src.utils.common import create_directories, load_object, load_json
from src.monitoring.logger import setup_logger
from src.utils.config_loader import load_yaml_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fraud Detection Training Pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to train (xgboost | logistic_regression | random_forest). "
             "Defaults to config value.",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation for faster iteration.",
    )
    return parser.parse_args()


def run_pipeline():
    args = parse_args()
    config = load_yaml_file("configs/config.yaml")
    logger = setup_logger(config["paths"]["log_file"])

    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION PIPELINE — START")
    logger.info("=" * 60)

    # ── Directories ──────────────────────────────────────────────
    create_directories([
        "data/processed",
        "models/trained",
        "models/artifacts",
        "logs",
        "reports",
        "reports/figures",
    ])

    # ── 1. Data Ingestion ────────────────────────────────────────
    t0 = time.time()
    logger.info("[1/6] Data Ingestion …")
    ingestion = DataIngestion()
    df = ingestion.load_data()
    logger.info(f"  -> Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    # ── 2. Data Validation ───────────────────────────────────────
    t0 = time.time()
    logger.info("[2/6] Data Validation …")
    validator = DataValidator(strict=False)
    df = validator.validate(df)
    logger.info(f"  -> Validated in {time.time() - t0:.1f}s")

    # ── 3. Data Cleaning ─────────────────────────────────────────
    t0 = time.time()
    logger.info("[3/6] Data Cleaning …")
    cleaner = DataCleaning()
    df = cleaner.clean(df)
    logger.info(f"  -> Cleaned in {time.time() - t0:.1f}s")

    # ── 4. Feature Engineering ───────────────────────────────────
    t0 = time.time()
    logger.info("[4/6] Feature Engineering …")
    fe = FeatureEngineering()
    df = fe.transform(df, is_train=True)
    logger.info(f"  -> Engineered {len(df.columns)} features in {time.time() - t0:.1f}s")

    # ── 5. Train / Val / Test Split ──────────────────────────────
    t0 = time.time()
    logger.info("[5/6] Splitting data …")
    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(df)
    logger.info(f"  -> Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    logger.info(f"  -> Split in {time.time() - t0:.1f}s")

    # ── 6. Training ──────────────────────────────────────────────
    t0 = time.time()
    logger.info("[6/6] Training model …")

    if args.model:
        config["training"]["model_name"] = args.model

    trainer = FraudModelTrainer()
    model, val_metrics = trainer.train(X_train, y_train, X_val, y_val)
    logger.info(f"  -> Trained in {time.time() - t0:.1f}s")

    # ── 7. Test-Set Evaluation ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("HELD-OUT TEST SET EVALUATION")
    logger.info("=" * 60)

    threshold = load_json(config["paths"]["threshold_file"])["best_threshold"]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)

    evaluator = ModelEvaluator()
    test_metrics = evaluator.evaluate(y_test, y_test_pred, y_test_prob)

    logger.info(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Test Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  Test F1:        {test_metrics['f1_score']:.4f}")
    logger.info(f"  Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    logger.info(f"  Test PR-AUC:    {test_metrics['pr_auc']:.4f}")
    logger.info(f"  Test MCC:       {test_metrics['mcc']:.4f}")
    logger.info(f"  Test Kappa:     {test_metrics['cohen_kappa']:.4f}")
    logger.info(f"\n{test_metrics['classification_report']}")

    total_time = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE — Total time: {total_time:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()