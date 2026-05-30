"""
Tests — Preprocessing
=====================
Tests for DataCleaning and DataValidator.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.data_cleaning import DataCleaning
from src.preprocessing.data_validator import DataValidator, DataValidationError


class TestDataCleaning:
    """Tests for the DataCleaning class."""

    def test_removes_duplicates(self):
        df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
        cleaner = DataCleaning()
        result = cleaner.clean(df)
        assert len(result) == 2

    def test_fills_missing_numeric_with_median(self):
        df = pd.DataFrame({"Amount": [10.0, np.nan, 30.0]})
        cleaner = DataCleaning()
        result = cleaner.clean(df)
        assert not result["Amount"].isnull().any()
        assert result["Amount"].iloc[1] == 20.0  # median of 10, 30

    def test_fills_missing_text_with_unknown(self):
        df = pd.DataFrame({"memo": ["coffee", None, "gas"]})
        cleaner = DataCleaning()
        result = cleaner.clean(df)
        assert result["memo"].iloc[1] == "Unknown"

    def test_cleaning_report_populated(self):
        df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
        cleaner = DataCleaning()
        cleaner.clean(df)
        assert cleaner.cleaning_report["duplicates_removed"] == 1

    def test_outlier_detection_reports(self):
        df = pd.DataFrame({"Amount": [10, 20, 30, 40, 50, 10000]})
        cleaner = DataCleaning(outlier_factor=1.5)
        cleaner.clean(df)
        assert "Amount" in cleaner.cleaning_report["outlier_summary"]


class TestDataValidator:
    """Tests for the DataValidator class."""

    def test_passes_valid_data(self, sample_dataframe):
        validator = DataValidator(strict=True)
        result = validator.validate(sample_dataframe)
        assert len(result) == len(sample_dataframe)

    def test_fails_on_missing_columns(self):
        df = pd.DataFrame({"Time": [1.0], "Amount": [10.0]})  # Missing V1-V28
        validator = DataValidator(strict=True)
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validator.validate(df)

    def test_non_strict_mode_warns(self):
        df = pd.DataFrame({"Time": [1.0], "Amount": [10.0]})
        validator = DataValidator(strict=False)
        result = validator.validate(df)
        assert len(validator.validation_report["missing_columns"]) > 0

    def test_negative_amount_detection(self, sample_dataframe):
        sample_dataframe.loc[0, "Amount"] = -100
        validator = DataValidator(strict=False)
        validator.validate(sample_dataframe)
        assert validator.validation_report["negative_amount_count"] == 1
