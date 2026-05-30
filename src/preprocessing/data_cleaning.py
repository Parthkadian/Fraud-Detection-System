"""
Data Cleaning Module
====================
Production-grade data cleaning with outlier detection reporting,
column-specific imputation strategies, and audit logging.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("fraud_detection_logger")


class DataCleaning:
    """
    Cleans raw transaction DataFrames for the fraud-detection pipeline.

    Steps
    -----
    1. Remove exact duplicate rows.
    2. Handle missing values with column-aware strategies.
    3. Flag statistical outliers (IQR method) — reported, **not** removed.
    4. Coerce types to float64 for numeric columns.
    """

    def __init__(self, outlier_factor: float = 3.0):
        """
        Parameters
        ----------
        outlier_factor : float
            Multiplier for the IQR fence.  Default ``3.0`` (conservative for
            fraud data where genuine outliers are informative).
        """
        self.outlier_factor = outlier_factor
        self.cleaning_report: dict = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline.  Returns a new DataFrame.
        """
        df = df.copy()
        initial_rows = len(df)

        # 1 — Duplicates
        dup_count = int(df.duplicated().sum())
        df = df.drop_duplicates()
        logger.info(f"Removed {dup_count:,} duplicate rows.")

        # 2 — Missing values
        missing_before = df.isnull().sum()
        missing_cols = missing_before[missing_before > 0]
        if not missing_cols.empty:
            logger.info(f"Missing values detected:\n{missing_cols.to_string()}")

        # Column-specific strategies
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"  {col}: filled {int(missing_before[col])} nulls with median ({median_val:.4f})")

        # Non-numeric fallback
        object_cols = df.select_dtypes(include="object").columns
        for col in object_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna("Unknown")

        # 3 — Outlier detection (report only — do NOT drop for fraud data)
        outlier_summary = self._detect_outliers(df, numeric_cols)

        # 4 — Build cleaning report
        self.cleaning_report = {
            "initial_rows": initial_rows,
            "final_rows": len(df),
            "duplicates_removed": dup_count,
            "missing_values_filled": int(missing_cols.sum()) if not missing_cols.empty else 0,
            "outlier_summary": outlier_summary,
        }
        logger.info(
            f"Cleaning complete -- {initial_rows:,} -> {len(df):,} rows "
            f"({dup_count} duplicates removed)."
        )
        return df

    def _detect_outliers(self, df: pd.DataFrame, numeric_cols) -> dict:
        """
        Flag outliers using the IQR method.  Returns counts per column.
        Outliers are *not* removed because extreme values are often the
        most informative signal in fraud detection.
        """
        outlier_counts: dict = {}
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.outlier_factor * iqr
            upper = q3 + self.outlier_factor * iqr
            n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_outliers > 0:
                outlier_counts[col] = n_outliers

        if outlier_counts:
            logger.info(
                f"Outlier report (IQR×{self.outlier_factor}): "
                f"{len(outlier_counts)} columns with outliers."
            )
        return outlier_counts