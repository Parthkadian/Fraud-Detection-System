"""
Tests — FraudTransformerPipeline
=================================
Unit tests for the sklearn-compatible transformer wrapper.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.transformer import FraudTransformerPipeline


class TestFraudTransformerPipeline:
    """Tests for the FraudTransformerPipeline class."""

    def _make_df(self, n: int = 20) -> pd.DataFrame:
        np.random.seed(42)
        df = pd.DataFrame({
            "Amount": np.random.uniform(0, 1000, n),
            "Time": np.random.uniform(0, 172800, n),
        })
        for i in range(1, 6):
            df[f"V{i}"] = np.random.randn(n)
        # Add a constant column to test variance threshold
        df["constant_col"] = 5.0
        return df

    def test_fit_transform_returns_ndarray(self):
        df = self._make_df()
        t = FraudTransformerPipeline(variance_threshold=0.0, scale=True)
        result = t.fit_transform(df)
        assert isinstance(result, np.ndarray)

    def test_removes_constant_columns(self):
        df = self._make_df()
        t = FraudTransformerPipeline(variance_threshold=0.01, scale=False)
        t.fit(df)
        # 'constant_col' should be removed
        assert "constant_col" not in t.get_feature_names_out()

    def test_scaling_applied(self):
        df = self._make_df()
        t = FraudTransformerPipeline(variance_threshold=0.0, scale=True)
        result = t.fit_transform(df)
        # After scaling, column means should be near 0
        assert abs(result.mean(axis=0)).max() < 1.0

    def test_transform_without_fit_raises(self):
        df = self._make_df()
        t = FraudTransformerPipeline()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            t.transform(df)

    def test_get_feature_names_without_fit_raises(self):
        t = FraudTransformerPipeline()
        with pytest.raises(RuntimeError, match="Call fit"):
            t.get_feature_names_out()

    def test_train_val_consistency(self):
        df_train = self._make_df(50)
        df_val = self._make_df(10)
        t = FraudTransformerPipeline(variance_threshold=0.0, scale=True)
        train_result = t.fit_transform(df_train)
        val_result = t.transform(df_val)
        assert train_result.shape[1] == val_result.shape[1]
