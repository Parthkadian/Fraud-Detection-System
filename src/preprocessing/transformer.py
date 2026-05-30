"""
Data Transformer
================
Provides a reusable, sklearn-compatible transformer wrapper that
composes StandardScaler with optional feature selection into a
single ``fit`` / ``transform`` interface.

This sits alongside ``FeatureEngineering`` for cases where you
want a conventional sklearn ``Pipeline`` — e.g. for grid-search
or cross-validation with scikit-learn utilities.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger("fraud_detection_logger")


class FraudTransformerPipeline(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that applies:

    1. **VarianceThreshold** — drops near-zero-variance columns
       (configurable threshold).
    2. **StandardScaler** — zero-mean, unit-variance scaling on
       the remaining numeric columns.

    Designed to slot cleanly into a ``sklearn.pipeline.Pipeline``
    for hyperparameter search or reproducible cross-validation.

    Parameters
    ----------
    variance_threshold : float
        Minimum variance required to keep a feature (default 0.0
        removes only constant columns).
    scale : bool
        Whether to apply StandardScaler after thresholding.

    Examples
    --------
    >>> from src.preprocessing.transformer import FraudTransformerPipeline
    >>> t = FraudTransformerPipeline(variance_threshold=0.01, scale=True)
    >>> X_train_t = t.fit_transform(X_train)
    >>> X_val_t   = t.transform(X_val)
    """

    def __init__(
        self,
        variance_threshold: float = 0.0,
        scale: bool = True,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.scale = scale

        self._var_selector: Optional[VarianceThreshold] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names_out: Optional[list[str]] = None

    # ------------------------------------------------------------------ #
    #  sklearn interface
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y=None) -> "FraudTransformerPipeline":
        """
        Fit the variance selector and (optionally) the scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix (numeric only).
        y : ignored
        """
        numeric_X = X.select_dtypes(include="number")

        # Step 1 — Variance filtering
        self._var_selector = VarianceThreshold(
            threshold=self.variance_threshold
        )
        self._var_selector.fit(numeric_X)
        selected_mask = self._var_selector.get_support()
        self._feature_names_out = [
            col for col, keep in zip(numeric_X.columns, selected_mask) if keep
        ]
        n_removed = len(numeric_X.columns) - len(self._feature_names_out)
        logger.info(
            f"VarianceThreshold (threshold={self.variance_threshold}) "
            f"removed {n_removed} feature(s). Kept {len(self._feature_names_out)}."
        )

        # Step 2 — Scaling
        if self.scale:
            filtered = numeric_X[self._feature_names_out]
            self._scaler = StandardScaler()
            self._scaler.fit(filtered)

        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """
        Apply the fitted variance selector and scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to transform.

        Returns
        -------
        np.ndarray
            Transformed feature array.
        """
        if self._var_selector is None or self._feature_names_out is None:
            raise RuntimeError(
                "FraudTransformerPipeline has not been fitted yet. "
                "Call fit() before transform()."
            )

        numeric_X = X.select_dtypes(include="number")
        filtered = numeric_X[self._feature_names_out]

        if self.scale and self._scaler is not None:
            return self._scaler.transform(filtered)

        return filtered.to_numpy()

    def get_feature_names_out(self) -> list[str]:
        """Return the list of selected feature names after fitting."""
        if self._feature_names_out is None:
            raise RuntimeError("Call fit() first.")
        return self._feature_names_out
