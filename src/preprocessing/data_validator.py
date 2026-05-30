"""
Data Validator Module
=====================
Provides schema validation, type enforcement, and range checks for incoming
transaction data.  Designed to run before any preprocessing or feature
engineering step so that downstream code can trust the shape and types of
its inputs.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("fraud_detection_logger")

# --------------------------------------------------------------------------- #
#  Expected schema for the raw credit-card transaction dataset.
# --------------------------------------------------------------------------- #
EXPECTED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

COLUMN_TYPES = {
    "Time": "float64",
    "Amount": "float64",
    **{f"V{i}": "float64" for i in range(1, 29)},
}


class DataValidationError(Exception):
    """Raised when data fails validation checks."""


class DataValidator:
    """
    Validates raw transaction DataFrames against the expected schema.

    Checks performed
    ----------------
    1. Required columns are present.
    2. Column dtypes are numeric (coercible to float64).
    3. No fully-null columns.
    4. Amount is non-negative.
    5. Reports summary statistics for auditing.
    """

    def __init__(self, strict: bool = True):
        """
        Parameters
        ----------
        strict : bool
            If *True*, raise ``DataValidationError`` on failures.
            If *False*, log warnings and continue.
        """
        self.strict = strict
        self.validation_report: dict = {}

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all validation checks and return a cleaned copy."""
        df = df.copy()
        self.validation_report = {}

        self._check_required_columns(df)
        self._check_numeric_types(df)
        self._check_null_columns(df)
        self._check_amount_range(df)
        self._summarise(df)

        return df

    # ------------------------------------------------------------------ #
    #  Individual checks
    # ------------------------------------------------------------------ #
    def _check_required_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        self.validation_report["missing_columns"] = missing
        if missing:
            msg = f"Missing required columns: {missing}"
            if self.strict:
                raise DataValidationError(msg)
            logger.warning(msg)

    def _check_numeric_types(self, df: pd.DataFrame) -> None:
        non_numeric = []
        for col in EXPECTED_COLUMNS:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                non_numeric.append(col)
        self.validation_report["non_numeric_columns"] = non_numeric
        if non_numeric:
            msg = f"Non-numeric columns detected (expected float64): {non_numeric}"
            if self.strict:
                raise DataValidationError(msg)
            logger.warning(msg)

    def _check_null_columns(self, df: pd.DataFrame) -> None:
        # Only check columns that actually exist to avoid KeyError in non-strict mode
        present_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
        if not present_cols:
            self.validation_report["fully_null_columns"] = []
            self.validation_report["null_percentages"] = {}
            return
        null_pct = df[present_cols].isnull().mean()
        fully_null = list(null_pct[null_pct == 1.0].index)
        self.validation_report["fully_null_columns"] = fully_null
        self.validation_report["null_percentages"] = null_pct[null_pct > 0].to_dict()
        if fully_null:
            msg = f"Fully-null columns detected: {fully_null}"
            if self.strict:
                raise DataValidationError(msg)
            logger.warning(msg)


    def _check_amount_range(self, df: pd.DataFrame) -> None:
        if "Amount" not in df.columns:
            return
        neg_count = int((df["Amount"] < 0).sum())
        self.validation_report["negative_amount_count"] = neg_count
        if neg_count > 0:
            msg = f"{neg_count} transactions have negative Amount values."
            if self.strict:
                raise DataValidationError(msg)
            logger.warning(msg)

    def _summarise(self, df: pd.DataFrame) -> None:
        self.validation_report["total_rows"] = len(df)
        self.validation_report["total_columns"] = len(df.columns)
        self.validation_report["duplicate_rows"] = int(df.duplicated().sum())
        logger.info(
            f"Validation summary — rows: {len(df):,}, "
            f"cols: {len(df.columns)}, "
            f"duplicates: {self.validation_report['duplicate_rows']:,}"
        )
