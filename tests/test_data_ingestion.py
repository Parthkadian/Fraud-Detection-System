"""
Tests — Data Ingestion
======================
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


class TestDataIngestion:
    """Tests for the DataIngestion class."""

    @patch("src.ingestion.data_ingestion.pd.read_csv")
    @patch("src.ingestion.data_ingestion.load_yaml_file")
    def test_load_data_with_class_column(self, mock_config, mock_read_csv):
        """Verify synthetic NLP memos are generated when Class column exists."""
        mock_config.return_value = {
            "paths": {"raw_data": "dummy.csv", "log_file": "logs/test.log"},
        }
        mock_read_csv.return_value = pd.DataFrame({
            "Time": [1.0, 2.0, 3.0],
            "Amount": [10.0, 20.0, 30.0],
            "V1": [0.1, 0.2, 0.3],
            "Class": [0, 1, 0],
        })

        from src.ingestion.data_ingestion import DataIngestion

        ingestion = DataIngestion()
        df = ingestion.load_data()

        assert "transaction_memo" in df.columns
        assert len(df) == 3

    @patch("src.ingestion.data_ingestion.pd.read_csv")
    @patch("src.ingestion.data_ingestion.load_yaml_file")
    def test_load_data_without_class_column(self, mock_config, mock_read_csv):
        """Verify default memo is used when Class column is absent."""
        mock_config.return_value = {
            "paths": {"raw_data": "dummy.csv", "log_file": "logs/test.log"},
        }
        mock_read_csv.return_value = pd.DataFrame({
            "Time": [1.0],
            "Amount": [10.0],
        })

        from src.ingestion.data_ingestion import DataIngestion

        ingestion = DataIngestion()
        df = ingestion.load_data()

        assert df["transaction_memo"].iloc[0] == "Standard transaction"
